#!/usr/bin/env python3
"""Single-device native CSA impl.forward versus the native-layout PTO entry.

Only one checkpoint attention module is loaded, not a model or serving engine.
Cache allocation snapshots are independent between the two execution paths.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path


def initialize(args, stack):
    import torch
    import torch_npu  # noqa: F401 -- Registers torch.npu before selecting the device.

    import vllm_ascend

    torch.npu.set_device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(62)
    vllm_ascend.__path__.append(str(args.extension_dir.resolve()))
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("Native vLLM Ascend operators could not be loaded")
    from vllm.platforms import current_platform

    current_platform.pre_register_and_update()
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs

    from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
    from vllm_ascend.utils import adapt_patch, register_ascend_customop, set_weight_prefetch_method

    engine_args = EngineArgs(
        model=str(args.model),
        dtype="bfloat16",
        quantization="compressed-tensors",
        tensor_parallel_size=1,
        enforce_eager=True,
        skip_tokenizer_init=True,
        max_model_len=max(4096, args.start_pos + args.seq * args.steps + 128),
        max_num_seqs=max(4, args.batch),
        max_num_batched_tokens=512,
        enable_prefix_caching=False,
        additional_config={
            "multistream_dsa_preprocess": False,
            "multistream_dsv4_dsa_overlap": False,
            "prefill_comm_compute_overlap": False,
        },
    )
    cfg = engine_args.create_engine_config()
    stack.enter_context(set_current_vllm_config(cfg))
    init_ascend_config(cfg)
    register_ascend_customop(cfg)
    adapt_patch()
    rendezvous = Path(tempfile.mkdtemp(prefix="tp1-", dir=args.out_dir)) / "rendezvous"
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=args.device,
        distributed_init_method=rendezvous.as_uri(),
        backend="hccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, backend="hccl")
    set_weight_prefetch_method(get_ascend_config().weight_prefetch_config)
    return cfg


def load_attention(args, cfg):
    import torch
    from safetensors import safe_open
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    from vllm_ascend.models.deepseek_v4 import DeepseekV4Attention
    from vllm_ascend.quantization.compressed_tensors_config import AscendCompressedTensorsConfig

    checkpoint_config = json.loads((args.model / "config.json").read_text())
    quant_desc = checkpoint_config["quantization_config"]
    quant_desc["ignore"] = [
        ("model." + name if not name.startswith("model.") else name).replace(
            ".attn.",
            ".self_attn.",
        )
        for name in quant_desc["ignore"]
    ]
    quant_cfg = AscendCompressedTensorsConfig.from_config(quant_desc)
    hf = cfg.model_config.hf_config
    with torch.device(f"npu:{args.device}"):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            attention = DeepseekV4Attention(
                cfg,
                hf,
                max_position_embeddings=hf.rope_parameters["original_max_position_embeddings"],
                cache_config=cfg.cache_config,
                quant_config=quant_cfg,
                prefix=f"model.layers.{args.layer}.self_attn",
            )
        finally:
            torch.set_default_dtype(previous_dtype)

    parameters = dict(attention.named_parameters())
    prefix = f"layers.{args.layer}.attn."
    weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    selected = {name: shard for name, shard in weight_map.items() if name.startswith(prefix)}
    loaded = set()
    for shard in sorted(set(selected.values())):
        with safe_open(str(args.model / shard), framework="pt", device="cpu") as handle:
            for source, source_shard in selected.items():
                if source_shard != shard:
                    continue
                name = source[len(prefix) :]
                if name.endswith(".scale"):
                    name = name[:-6] + ".weight_scale"
                if name not in parameters:
                    raise KeyError(f"Checkpoint parameter {source} does not map to attention: {name}")
                param = parameters[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                if name == "wo_a.weight":
                    # Use Ascend's grouped-output loader; the generic V2 loader
                    # attached by compressed-tensors only fills the flat weight.
                    loader = attention.wo_a.weight_loader
                weight = handle.get_tensor(source)
                # The checkpoint stores per-channel scales flat, whereas the
                # native compressed-tensors loader allocates [out_features, 1].
                if name.endswith(".weight_scale") and weight.numel() == param.numel():
                    weight = weight.reshape(param.shape)
                print(f"[weights] {name}: {tuple(weight.shape)} -> {tuple(param.shape)}", flush=True)
                loader(param, weight)
                loaded.add(name)
    # Quantization-specific offset buffers can be initialized by the quantizer;
    # every mathematical weight, norm and APE must have a checkpoint source.
    missing = set(parameters) - loaded
    if any(not name.endswith("weight_offset") for name in missing):
        raise RuntimeError(f"Unloaded attention parameters: {sorted(missing)}")
    for name in missing:
        parameters[name].data.zero_()
    for module in attention.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)
    return attention, {
        "checkpoint": str(args.model),
        "layer": args.layer,
        "loaded": sorted(loaded),
        "initialized_auxiliary": sorted(missing),
        "parameter_shapes": {name: [list(p.shape), str(p.dtype)] for name, p in attention.named_parameters()},
    }


def cache_views(roots):
    """Recreate vLLM's six views without losing shared-allocation aliasing."""
    import torch

    main, raw, index = roots
    main_state = main.as_strided((main.shape[0], 8, 1, 2048), (32768, 2048, 2048, 1))
    cmp_kv = main.view(torch.bfloat16).view(-1, 128, 1, 512)
    inner_state = index.view(torch.float32).as_strided(
        (index.shape[0], 8, 1, 512),
        (4160, 512, 512, 1),
    )
    key = index.as_strided((index.shape[0], 128, 1, 128), (16640, 128, 128, 1))
    scale = index.view(torch.float16).as_strided(
        (index.shape[0], 128, 1, 1),
        (8320, 1, 1, 1),
        storage_offset=8192,
    )
    return cmp_kv, raw, main_state, inner_state, key, scale


def build_payload(args):
    """Use absolute native page tables and disjoint writable request pages.

    Weights are real; hidden states and historical caches are reproducible
    synthetic inputs. This is an actual impl.forward test, not a full request.
    """
    import torch

    first, end = args.start_pos, args.start_pos + args.seq * args.steps
    state = torch.zeros(args.batch, (end + 7) // 8, dtype=torch.int32)
    raw = torch.zeros(args.batch, (end + 127) // 128, dtype=torch.int32)
    cmp = torch.zeros(args.batch, (end + 511) // 512, dtype=torch.int32)
    next_page, next_raw = 1, 1
    state_pages, cmp_pages = [], []
    for request in range(args.batch):
        for col in range(cmp.shape[1]):
            cmp[request, col] = next_page
            cmp_pages.append(next_page)
            next_page += 1
        for col in range(max(0, first - 7) // 8, state.shape[1]):
            state[request, col] = next_page
            state_pages.append(next_page)
            next_page += 1
        for col in range(max(0, first - 127) // 128, raw.shape[1]):
            raw[request, col] = next_raw
            next_raw += 1
    roots = (
        torch.zeros(next_page, 16, 2048, dtype=torch.float32),
        torch.randn(next_raw, 128, 1, 512, dtype=torch.bfloat16) * 0.1,
        torch.zeros(next_page, 130, 128, dtype=torch.int8),
    )
    cmp_kv, _, main_state, inner_state, key, scale = cache_views(roots)
    for page in cmp_pages:
        cmp_kv[page].copy_(torch.randn_like(cmp_kv[page]) * 0.1)
        key[page].copy_(torch.randint(-100, 101, key[page].shape, dtype=torch.int8))
        scale[page].fill_(0.01)
    for page in state_pages:
        main_state[page].copy_(torch.randn_like(main_state[page]) * 0.1)
        inner_state[page].copy_(torch.randn_like(inner_state[page]) * 0.1)
    device = f"npu:{args.device}"
    native_roots = tuple(t.to(device) for t in roots)
    pto_roots = tuple(t.to(device) for t in roots)
    tables_cpu = (cmp, state, state.clone(), cmp.clone(), raw)
    # The public output exposes only real token rows. Guard the next tile so a
    # missing tail mask fails explicitly instead of corrupting another input.
    pto_out_storage = torch.full(
        (args.batch * args.seq + 8, 4096),
        -12.5,
        dtype=torch.bfloat16,
        device=device,
    )
    return {
        "initial": roots,
        "native_roots": native_roots,
        "pto_roots": pto_roots,
        "native": cache_views(native_roots),
        "pto": cache_views(pto_roots),
        "tables_cpu": tables_cpu,
        "tables": tuple(t.to(device) for t in tables_cpu),
        "state_pages": state_pages,
        "cmp_pages": cmp_pages,
        "x": torch.randn(args.batch * args.seq, 4096, dtype=torch.bfloat16).to(device),
        "native_out": torch.empty(args.batch * args.seq, 4096, dtype=torch.bfloat16, device=device),
        "pto_out": pto_out_storage[: args.batch * args.seq],
        "pto_output_guard": pto_out_storage[args.batch * args.seq :],
    }


def build_metadata(args, payload, start_pos):
    import torch

    from vllm_ascend.attention.dsa_v1 import AscendDSADecodeMetadata, AscendDSAMetadata
    from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa

    device = payload["x"].device
    b, s = args.batch, args.seq
    t, end = b * s, start_pos + s
    pos_cpu = torch.arange(start_pos, end, dtype=torch.int64).repeat(b)
    pos = pos_cpu.to(device)
    qstarts_cpu = torch.arange(0, t + 1, s, dtype=torch.int32)
    qstarts = qstarts_cpu.to(device)
    lengths = torch.full((b,), end, dtype=torch.int32, device=device)
    starts = torch.full((b,), start_pos, dtype=torch.int32, device=device)
    cos, sin = get_cos_and_sin_dsa(pos)
    boundaries = (pos_cpu + 1) % 4 == 0
    boundary_positions = pos_cpu[boundaries] + 1 - 4
    rope_capacity = min(t, t // 4 + b)
    padded_boundaries = torch.zeros(rope_capacity, dtype=torch.int64)
    padded_boundaries[: boundary_positions.numel()] = boundary_positions
    cmp_cos, cmp_sin = get_cos_and_sin_dsa({"c4": padded_boundaries.to(device)})
    sas = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        cu_seqlens_q=qstarts,
        seqused_kv=lengths,
        max_seqlen_q=s,
        max_seqlen_kv=end,
        batch_size=b,
        cmp_topk=512,
        cmp_ratio=4,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=True,
        device=str(device),
    )
    qli = torch.ops._C_ascend.npu_quant_lightning_indexer_metadata(
        actual_seq_lengths_query=qstarts[1:].clone(),
        actual_seq_lengths_key=lengths.clone(),
        num_heads_q=64,
        num_heads_k=1,
        head_dim=128,
        query_quant_mode=0,
        key_quant_mode=0,
        batch_size=b,
        max_seqlen_q=s,
        max_seqlen_k=end,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=512,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        cmp_ratio=4,
        device=str(device),
    )
    hadamard = torch.ones(1, 1)
    while hadamard.shape[0] < 128:
        hadamard = torch.cat((torch.cat((hadamard, hadamard), 1), torch.cat((hadamard, -hadamard), 1)), 0)
    hadamard = hadamard.to(device=device, dtype=torch.bfloat16)
    metadata = []
    request_ids = torch.arange(b).repeat_interleave(s)
    for group, (table_cpu, table) in enumerate(zip(payload["tables_cpu"], payload["tables"])):
        rows = pos_cpu // 4 if group in (0, 3) else pos_cpu
        page_size = 8 if group in (1, 2) else 128
        # AscendDSAMetadataBuilder converts flat allocator slots into native
        # scatter_nd coordinates. A flat [N] tensor instead means ONE N-axis
        # index to ScatterNdUpdateV2 and can write outside the intended row.
        slots = torch.stack(
            (table_cpu[request_ids, rows // page_size].long(), rows % page_size),
            dim=-1,
        )
        if group in (0, 3):
            slots = slots[boundaries]
        slots = slots.to(device=device, dtype=torch.int64)
        decode = AscendDSADecodeMetadata(
            input_positions=pos,
            block_table=table,
            seq_lens=lengths,
            max_seqlen_kv=end,
            max_seqlen_q=s,
            seq_lens_list=[end] * b,
            max_seq_lens=end,
            slot_mapping=slots,
            query_start_loc=qstarts,
            query_start_loc_cpu=qstarts_cpu,
            sin=sin,
            cos=cos,
            compress_sin=cmp_sin,
            compress_cos=cmp_cos,
            start_pos=starts,
            sas_metadata=sas,
            qli_metadata=qli,
        )
        metadata.append(
            AscendDSAMetadata(
                num_actual_tokens=t,
                slot_mapping=slots,
                query_start_loc=qstarts,
                seq_lens=lengths,
                block_tables=table,
                sin=sin,
                cos=cos,
                num_decodes=b,
                num_decode_tokens=t,
                num_prefills=0,
                num_input_tokens=t,
                query_lens=[s] * b,
                decode=decode,
                hadamard=hadamard,
                start_pos=starts,
            )
        )
    return metadata


def difference(actual, expected):
    import torch

    a, e = actual.detach().float().cpu().reshape(-1), expected.detach().float().cpu().reshape(-1)
    diff = (a - e).abs()
    relative = diff / torch.maximum(a.abs(), e.abs()).clamp_min(1e-3)
    # Large unchanged cache pools need wide reductions: FP32 accumulation can
    # otherwise report a cosine above one even for identical BF16 histories.
    a_sq = a.square().sum(dtype=torch.float64)
    e_sq = e.square().sum(dtype=torch.float64)
    d_sq = diff.square().sum(dtype=torch.float64)
    cosine = (a * e).sum(dtype=torch.float64) / (a_sq * e_sq).sqrt().clamp_min(1e-24)
    return {
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(e).all()),
        "shape": list(actual.shape),
        "max_abs": diff.max().item(),
        "mean_abs": diff.sum(dtype=torch.float64).item() / diff.numel(),
        "relative_l2": (d_sq.sqrt() / e_sq.sqrt().clamp_min(1e-12)).item(),
        "cosine": cosine.item(),
        "frac_rdiff_gt_5e-3": (relative > 5e-3).count_nonzero().item() / relative.numel(),
        "frac_rdiff_gt_1e-2": (relative > 1e-2).count_nonzero().item() / relative.numel(),
        "allclose_1e-2": bool(torch.allclose(a, e, rtol=1e-2, atol=1e-2)),
    }


def numerical_ab(args, cfg, attention, record):
    import torch

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention import pto_attn

    payload = build_payload(args)
    x_before = payload["x"].cpu().clone()
    metadata = build_metadata(args, payload, args.start_pos)
    impl, layer = attention.dsa_attn.dsa_attn.impl, attention.dsa_attn.dsa_attn.layer_name
    print(f"[A/B] layer={layer} B={args.batch} S={args.seq} start={args.start_pos}", flush=True)
    pto_attn.audit_shared_pool_ownership(metadata, payload["pto"], args.batch, args.seq)
    for md in metadata:
        if md.decode.slot_mapping.ndim != 2 or md.decode.slot_mapping.shape[1] != 2:
            raise ValueError("Native ScatterNdUpdateV2 requires [N, 2] (page, offset) coordinates")
    if args.diagnostics:
        record["metadata_before_native"] = {
            "positions": metadata[0].decode.input_positions.cpu().tolist(),
            "raw_slots": metadata[4].decode.slot_mapping.cpu().tolist(),
            "cmp_slots": metadata[0].decode.slot_mapping.cpu().tolist(),
        }
    record["stage"] = "native_forward"
    with set_ascend_forward_context({layer: metadata}, cfg, num_tokens=args.batch * args.seq):
        impl.forward(layer, payload["x"], payload["native"], metadata, output=payload["native_out"])
        torch.npu.synchronize()
        print("[A/B] native impl.forward completed", flush=True)
        if args.diagnostics:
            native_raw_snapshot = payload["native"][1].cpu().clone()
            record["positions_after_native"] = metadata[0].decode.input_positions.cpu().tolist()
        record["stage"] = "pto_forward"
        call_args, _ = pto_attn.build_args(
            impl,
            payload["x"],
            payload["pto"],
            metadata,
            args.seq,
            layer,
            output=payload["pto_out"],
        )
        if args.diagnostics:
            record["positions_after_build_args"] = metadata[0].decode.input_positions.cpu().tolist()
        op = pto_attn._registered()
        if args.diagnostics:
            record["positions_after_registration"] = metadata[0].decode.input_positions.cpu().tolist()
        op(*call_args)
        torch.npu.synchronize()
        if args.diagnostics:
            record["positions_after_pto"] = metadata[0].decode.input_positions.cpu().tolist()
    record["comparisons"] = {"output": difference(payload["pto_out"], payload["native_out"])}
    for i, name in enumerate(("cmp_kv", "raw_kv", "main_state", "inner_state", "index_key", "index_scale")):
        pages = payload["state_pages"] if i in (2, 3) else payload["cmp_pages"]
        a, e = payload["pto"][i], payload["native"][i]
        if i != 1:
            ids = torch.tensor(pages, device=a.device)
            a, e = a[ids], e[ids]
        record["comparisons"][name] = difference(a, e)
    if args.diagnostics:
        # Check the earliest independently visible result without changing any
        # production kernel or relying on a later attention output comparison.
        native_raw = payload["native"][1].cpu().view(-1, 512)
        pto_raw = payload["pto"][1].cpu().view(-1, 512)
        initial_raw = payload["initial"][1].view(-1, 512)
        # Use the fixture's immutable host positions to select reference rows,
        # independently of the input-integrity observations above.
        positions = torch.tensor(record["metadata_before_native"]["positions"], dtype=torch.int64)
        request_ids = torch.arange(args.batch).repeat_interleave(args.seq)
        raw_table = payload["tables_cpu"][4]
        rows = raw_table[request_ids, positions // 128].long() * 128 + positions % 128
        weights = pto_attn.prepare_weights(impl)
        projected = x_before.float() @ weights["wkv"].float().cpu()
        gamma = weights["gamma_ckv"].float().cpu()
        cos, sin = pto_attn._native_rope_tables(layer)
        cos, sin = cos[positions.to(cos.device)].cpu(), sin[positions.to(sin.device)].cpu()
        references = {}
        for name, values in (("fp32_projection", projected), ("bf16_projection", projected.bfloat16().float())):
            normed = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + impl.eps) * gamma
            if name == "bf16_projection":
                normed = normed.bfloat16().float()
            rotated = normed.clone()
            tail = normed[:, -64:]
            swapped = tail.reshape(-1, 32, 2).flip(-1).reshape(-1, 64)
            sign = torch.tensor([-1.0, 1.0]).repeat(32)
            rotated[:, -64:] = tail * cos + swapped * sin * sign
            references[name] = rotated.bfloat16()
        record["diagnostics"] = {
            "hidden_states_unchanged": torch.equal(payload["x"].cpu(), x_before),
            "reference_positions": positions.tolist(),
            "native_cache_unchanged_by_pto": torch.equal(payload["native"][1].cpu(), native_raw_snapshot),
            "expected_raw_rows": rows.tolist(),
            "native_changed_raw_rows": (native_raw != initial_raw).any(-1).nonzero().flatten().tolist(),
            "pto_changed_raw_rows": (pto_raw != initial_raw).any(-1).nonzero().flatten().tolist(),
            "raw_written_rows": difference(pto_raw[rows], native_raw[rows]),
            "raw_nope": difference(pto_raw[rows, :448], native_raw[rows, :448]),
            "raw_rope": difference(pto_raw[rows, 448:], native_raw[rows, 448:]),
            "native_vs_reference": {name: difference(native_raw[rows], ref) for name, ref in references.items()},
            "pto_vs_reference": {name: difference(pto_raw[rows], ref) for name, ref in references.items()},
            "output_abs_mean": {
                "native": payload["native_out"].float().abs().mean().item(),
                "pto": payload["pto_out"].float().abs().mean().item(),
            },
        }
        torch.save(
            {"native_raw": native_raw[rows], "pto_raw": pto_raw[rows], **references},
            args.out_dir / "raw_kv_diagnostics.pt",
        )
    record["stage"] = "numerical_comparison_complete"
    record["ok"] = all(v["allclose_1e-2"] for v in record["comparisons"].values())
    record["output_guard_unchanged"] = bool((payload["pto_output_guard"].cpu() == -12.5).all())
    record["ok"] = record["ok"] and record["output_guard_unchanged"]
    if args.diagnostics:
        original_positions = record["metadata_before_native"]["positions"]
        record["position_inputs_unchanged"] = all(
            record[name] == original_positions
            for name in (
                "positions_after_native",
                "positions_after_build_args",
                "positions_after_registration",
                "positions_after_pto",
            )
        )
        record["ok"] = record["ok"] and record["position_inputs_unchanged"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/data/models/dsv4-flash-0731-dspark-w8a8"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, choices=(1, 6), default=1)
    parser.add_argument("--start-pos", type=int, default=127)
    parser.add_argument(
        "--steps",
        type=int,
        choices=(1,),
        default=1,
        help="One fixed decode step; --performance benchmarks its ACLGraph replay, not multi-step inference.",
    )
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument(
        "--diagnostics", action="store_true", help="Audit published raw rows against a CPU projection reference."
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Time fixed B4/S6/128K ACLGraph replay; numerical parity is not a gate.",
    )
    parser.add_argument("--perf-iterations", type=int, default=50)
    parser.add_argument("--perf-rounds", type=int, default=6)
    parser.add_argument(
        "--perf-profile", action="store_true", help="Collect a separate untimed NPU profile after benchmarking."
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "scope": "CSA impl.forward only",
        "stage": "initialize",
        "ok": False,
        "workload": {
            "batch": args.batch,
            "seq": args.seq,
            "start_pos": args.start_pos,
            "steps": args.steps,
            "execution": "aclgraph_replay" if args.performance else "eager",
        },
        "inputs": "Real checkpoint attention weights; synthetic hidden states and history caches",
    }
    try:
        with contextlib.ExitStack() as stack:
            cfg = initialize(args, stack)
            import torch
            import vllm_ascend.vllm_ascend_C as native_extension

            record["runtime"] = {
                "python": sys.executable,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "triton-ascend")
                },
                "native_extension": native_extension.__file__,
                "native_vendor": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
                "ptoas_root": os.environ.get("PTOAS_ROOT"),
                "pypto_root": os.environ.get("PYPTO_ROOT"),
            }
            print(json.dumps(record["runtime"], indent=2), flush=True)

            stack.enter_context(torch.inference_mode())
            record["stage"] = "load_one_attention"
            print("[A/B] loading one checkpoint attention", flush=True)
            attention, info = load_attention(args, cfg)
            record["weights"] = info
            record["impl"] = type(attention.dsa_attn.dsa_attn.impl).__name__
            if args.setup_only:
                record["stage"] = "setup_complete"
                record["ok"] = True
            elif args.performance:
                from impl_forward_perf import benchmark

                benchmark(args, cfg, attention, record)
            else:
                numerical_ab(args, cfg, attention, record)
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        (args.out_dir / "result.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2), flush=True)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
