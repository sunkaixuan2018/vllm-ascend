#!/usr/bin/env python3
"""B 档：跑 PTO 版 CSA（我们的代码），出 trace.json。

中间的 CSA 是 pypto-lib 的 `sparse_attn_test`
（models/deepseek_v4_flash_mtp/decode_sparse_attn_csa.py），它比 vendor 的
`npu_sparse_attn_sharedkv` 边界更大——把 RoPE 和输出投影（wo_a/wo_b/wo_b_scale）
融进了同一个 kernel，所以替换点只能在层级，不能做算子级 1:1 对换。

输入用 pypto-lib 自带的 `build_tensor_specs()`，确定性假数据，不需要任何模型权重；
数值正确性由 pypto-lib 自己的 `golden_sparse_attn` 判定。

trace 的取法：`golden.run` 一次调用里既编译又执行，直接套 profiler 会把编译过程
也录进去。所以跑两遍——第一遍编译并留下 work_dir，第二遍用 `runtime_dir` 复用编译、
只在 profiler 里执行，trace 就只剩真正的下发和 kernel。

用法（锁内）:
    export ASCEND_RT_VISIBLE_DEVICES=$TASK_DEVICE
    source env_pto_csa.sh
    $PTO_PY csa_b_tier.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="PTO CSA B 档用例（带 trace）")
    p.add_argument("--out-dir", required=True, help="产物目录（trace.json / result.json）")
    p.add_argument("--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--no-profile", action="store_true", help="只跑功能，不采 trace")
    p.add_argument("--chip-swimlane", type=int, default=0, choices=range(5),
                   help="PTO 侧 chip swimlane 级别（与 torch trace 独立）")
    p.add_argument("--pmu", type=int, default=0, choices=[0, 1, 2, 4])
    p.add_argument("--dep-gen", action="store_true", help="导出 PTO 依赖边 deps.json")
    p.add_argument("--dump-passes", action="store_true")
    p.add_argument("--save-data", action="store_true",
                   help="把 fixture 与 golden 输出落盘到 work_dir/data，供 vendor 侧复用同一份输入")
    p.add_argument("--reuse-compile", action="store_true", default=True,
                   help="第二遍复用第一遍的编译产物（默认开）")
    p.add_argument("--no-reuse-compile", dest="reuse_compile", action="store_false")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prof_dir = out_dir / "prof"
    result: dict = {
        "backend": "pto",
        "kernel": "pypto-lib models/deepseek_v4_flash_mtp/decode_sparse_attn_csa.sparse_attn_test",
        "scope": "CSA + RoPE + output projection fused (PTO kernel boundary)",
        "platform": args.platform,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "(未设置)"),
        "stage": "import",
    }

    def save():
        (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result, ensure_ascii=False), flush=True)

    try:
        import torch  # noqa: F401
        import torch_npu
        from golden import ratio_allclose, run

        import decode_sparse_attn_csa as csa

        result["pypto_lib"] = os.environ.get("PYPTO_LIB_ROOT", "?")
        result["pypto_root"] = os.environ.get("PYPTO_ROOT", "?")
        result["ptoas_root"] = os.environ.get("PTOAS_ROOT", "?")
        result["compress_ratio"] = getattr(csa, "COMPRESS_RATIO", None)
        result["topk"] = getattr(csa, "TOPK", None)
        result["stage"] = "build_specs"
        save()

        specs = csa.build_tensor_specs()
        # RunConfig 的这几个键没有默认值，缺一个就 AttributeError（实测缺
        # enable_chip_swimlane 时报 'RunConfig' object has no attribute ...），
        # 所以照抄 decode_sparse_attn_csa.py 的 __main__ 传全。
        base_config = dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.chip_swimlane,
            enable_dep_gen=args.dep_gen,
            enable_pmu=args.pmu,
        )
        common = dict(
            fn=csa.sparse_attn_test,
            specs=specs,
            golden_fn=csa.golden_sparse_attn,
            rtol=1e-3,
            atol=1e-3,
            compare_fn={"attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
        )

        # 第一遍：编译 + 执行 + 数值校验（不采 trace）
        result["stage"] = "compile_and_validate"
        save()
        t0 = time.perf_counter()
        first = run(config=dict(base_config), save_data=args.save_data, **common)
        result["pass1"] = {
            "passed": bool(first.passed),
            "error": first.error,
            "seconds": round(time.perf_counter() - t0, 2),
            "work_dir": str(first.work_dir) if first.work_dir else None,
        }
        # data/ 里是这次跑用的 fixture 与 golden 输出。vendor 侧必须读它，
        # 因为 specs 的 init_value 是无种子的 torch.rand，重造一份不是同一组数。
        if args.save_data and first.work_dir:
            data_dir = Path(first.work_dir) / "data"
            result["pass1"]["data_dir"] = str(data_dir) if data_dir.is_dir() else None
            (out_dir / "data_dir.txt").write_text(str(data_dir))
        if not first.passed:
            result["ok"] = False
            result["stage"] = "failed_validation"
            return 1

        if args.no_profile:
            result["ok"] = True
            result["stage"] = "complete"
            return 0

        # 第二遍：复用编译，只把执行录进 trace
        result["stage"] = "profile"
        save()
        prof_dir.mkdir(parents=True, exist_ok=True)
        # runtime_dir 是 run() 的参数，不是 RunConfig 的字段（塞进 config 会报
        # TypeError: RunConfig.__init__() got an unexpected keyword argument）。
        second_config = dict(base_config)
        reuse = str(first.work_dir) if (args.reuse_compile and first.work_dir) else None

        experimental = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        )
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(prof_dir)),
            record_shapes=True,
            with_stack=True,
            experimental_config=experimental,
        ) as prof:
            with torch.profiler.record_function("pto_csa"):
                t1 = time.perf_counter()
                second = run(config=second_config, runtime_dir=reuse, **common)
                second_s = time.perf_counter() - t1
            prof.step()

        result["pass2"] = {
            "passed": bool(second.passed),
            "error": second.error,
            "seconds": round(second_s, 2),
            "reused_compile": reuse is not None,
        }

        traces = sorted(prof_dir.rglob("trace_view.json"))
        if traces:
            dst = out_dir / "trace.json"
            shutil.copyfile(traces[-1], dst)
            result["trace"] = {"path": str(dst), "bytes": dst.stat().st_size,
                               "source": str(traces[-1])}
        else:
            result["trace"] = {"path": None,
                               "note": f"{prof_dir} 下没找到 trace_view.json"}

        result["ok"] = bool(second.passed) and traces != []
        result["stage"] = "complete"
    except Exception:
        result["ok"] = False
        result["error"] = traceback.format_exc()
    finally:
        save()
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
