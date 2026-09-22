"""Warmed single-layer native/PTO ACLGraph performance comparison.

This is a fixed decode-step microbenchmark, not a progressing serving request.
Cross-implementation numerical parity is deliberately not a performance gate.
"""

from __future__ import annotations

import json
import statistics
import time


def summarize(samples):
    return {
        "unit": "us_per_forward",
        "round_means": samples,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def benchmark(args, cfg, attention, record):
    # Import only after the caller initializes Ascend and the attention module.
    import torch
    import torch_npu
    from impl_forward_ab import build_metadata, build_payload
    from vllm.forward_context import get_forward_context

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention import pto_attn

    if (args.batch, args.seq, args.start_pos, args.steps) != (4, 6, 131066, 1):
        raise ValueError("This benchmark is scoped to B4/S6/end-position 128K/one fixed step")
    if args.perf_iterations < 1 or args.perf_rounds < 2:
        raise ValueError("Use positive iterations and at least two alternating rounds")

    def checkpoint(stage):
        record["stage"] = stage
        (args.out_dir / "result.json").write_text(json.dumps(record, indent=2))
        print(f"[PERF] {stage}", flush=True)

    payload = build_payload(args)
    metadata = build_metadata(args, payload, args.start_pos)
    impl = attention.dsa_attn.dsa_attn.impl
    layer = attention.dsa_attn.dsa_attn.layer_name
    pto_attn.audit_shared_pool_ownership(metadata, payload["pto"], args.batch, args.seq)
    original_positions = metadata[0].decode.input_positions.cpu().clone()
    original_device = tuple(t.to(payload["x"].device) for t in payload["initial"])
    op = pto_attn._registered()
    prepared, _ = pto_attn.build_args(
        impl,
        payload["x"],
        payload["pto"],
        metadata,
        args.seq,
        layer,
        output=payload["pto_out"],
    )
    holders = {}

    def native():
        impl.forward(layer, payload["x"], payload["native"], metadata, output=payload["native_out"])

    def prepare(name):
        bound, _ = pto_attn.build_args(
            impl,
            payload["x"],
            payload["pto"],
            metadata,
            args.seq,
            layer,
            output=payload["pto_out"],
        )
        holders[name] = bound  # Keep graph-owned argument allocations alive.
        return bound

    def pto_total():
        op(*prepare("pto_total"))

    def pto_kernel():
        op(*prepared)

    def pto_prepare():
        prepare("pto_prepare_only")

    variants = {
        "native_total": native,
        "pto_total": pto_total,
        "pto_kernel_prebound": pto_kernel,
        "pto_prepare_only": pto_prepare,
    }

    def restore(name):
        roots = payload["native_roots"] if name == "native_total" else payload["pto_roots"]
        for dst, src in zip(roots, original_device):
            dst.copy_(src)
        torch.npu.synchronize()

    record["performance"] = {
        "boundary": "CSA impl.forward versus build_args plus registered PTO attention",
        "execution": "warmed NPUGraph replay of a fixed step; not multi-step inference",
        "timing": "NPU events around a replay batch; alternating variant order per round",
        "excluded": [
            "initial compilation",
            "weight preparation",
            "metadata construction",
            "cache restoration",
            "graph capture",
            "profiling",
        ],
        "iterations_per_round": args.perf_iterations,
        "rounds": args.perf_rounds,
        "warmup_calls": 3,
        "native_multistream_dsa_preprocess": impl.multistream_dsa_preprocess,
        "native_multistream_dsv4_dsa_overlap": impl.multistream_dsv4_dsa_overlap,
        "numerical_parity": "not required by this performance-only run; prior mismatch remains",
        "decomposition_note": "Independent graphs include their own launch overhead; do not sum them",
        "sanity": {},
    }
    graphs = {}
    stream = torch.npu.Stream()
    # Warm the same stream used for capture, including every native op and
    # the real PyPTO specialization. Never compile inside graph capture.
    with (
        set_ascend_forward_context({layer: metadata}, cfg, num_tokens=args.batch * args.seq),
        torch.npu.stream(stream),
    ):
        for name, call in variants.items():
            checkpoint(f"warmup_{name}")
            restore(name)
            for _ in range(3):
                call()
            torch.npu.synchronize()

        for name, call in variants.items():
            checkpoint(f"capture_{name}")
            restore(name)
            graph = torch_npu.npu.NPUGraph()
            context = get_forward_context()
            context.capturing = True
            try:
                with torch_npu.npu.graph(graph, stream=stream):
                    call()
            finally:
                context.capturing = False
            torch.npu.synchronize()
            graphs[name] = graph
            restore(name)
            if name == "pto_prepare_only":
                valid_index = pto_attn.ARG_ORDER.index("token_valid")
                graph_valid = holders[name][valid_index]
                graph_valid.zero_()
            else:
                output = payload["native_out"] if name == "native_total" else payload["pto_out"]
                output.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            if name == "pto_prepare_only":
                executed = bool((graph_valid.cpu() == 1).all())
            else:
                executed = bool(torch.isfinite(output).all().cpu())
            sanity = {
                "replay_wrote_output": executed,
                "positions_unchanged": torch.equal(metadata[0].decode.input_positions.cpu(), original_positions),
                "output_guard_unchanged": bool((payload["pto_output_guard"].cpu() == -12.5).all()),
            }
            record["performance"]["sanity"][name] = sanity
            if not all(sanity.values()):
                raise RuntimeError(f"Graph replay integrity failed: {name}: {sanity}")

        device_samples = {name: [] for name in variants}
        wall_samples = {name: [] for name in variants}
        names = list(variants)
        checkpoint("timing_aclgraph")
        # All variants run on the same leased NPU. Reverse the order on
        # alternating rounds to reduce systematic warm-cache/thermal bias.
        for round_id in range(args.perf_rounds):
            order = names if round_id % 2 == 0 else list(reversed(names))
            for name in order:
                restore(name)
                for _ in range(3):
                    graphs[name].replay()
                torch.npu.synchronize()
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                started = time.perf_counter()
                start.record(stream)
                for _ in range(args.perf_iterations):
                    graphs[name].replay()
                end.record(stream)
                end.synchronize()
                device_us = start.elapsed_time(end) * 1000 / args.perf_iterations
                wall_us = (time.perf_counter() - started) * 1e6 / args.perf_iterations
                device_samples[name].append(device_us)
                wall_samples[name].append(wall_us)
                print(
                    f"[PERF] round={round_id + 1} {name}: device={device_us:.3f} us wall={wall_us:.3f} us",
                    flush=True,
                )

        record["performance"]["device"] = {name: summarize(v) for name, v in device_samples.items()}
        record["performance"]["host_wall"] = {name: summarize(v) for name, v in wall_samples.items()}
        record["performance"]["pto_over_native"] = statistics.median(device_samples["pto_total"]) / statistics.median(
            device_samples["native_total"]
        )
        checkpoint("timing_complete")

        if args.perf_profile:
            # Profiling is a separate untimed pass, never a latency sample.
            checkpoint("profile_aclgraph")
            profile_dir = args.out_dir / "profile"
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
                record_shapes=True,
                with_stack=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                ),
            ) as prof:
                for name in ("native_total", "pto_total", "pto_prepare_only"):
                    for _ in range(5):
                        with torch.profiler.record_function(f"CSA_AB::{name}"):
                            graphs[name].replay()
                        torch.npu.synchronize()
                    prof.step()
            record["performance"]["profile_dir"] = str(profile_dir)

    torch.npu.synchronize()
    record["performance"]["final_positions_unchanged"] = torch.equal(
        metadata[0].decode.input_positions.cpu(),
        original_positions,
    )
    record["performance"]["final_output_guard_unchanged"] = bool(
        (payload["pto_output_guard"].cpu() == -12.5).all(),
    )
    record["ok"] = (
        record["performance"]["final_positions_unchanged"] and record["performance"]["final_output_guard_unchanged"]
    )
    record["ok_scope"] = "performance execution and buffer integrity only, not numerical equivalence"
    checkpoint("performance_complete")
