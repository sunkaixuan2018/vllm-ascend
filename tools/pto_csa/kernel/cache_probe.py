#!/usr/bin/env python3
"""问 PyPTO：JIT 编译缓存为什么一次都没落盘。

现象：`$HOME/.cache/pypto/jit` 始终为空，每次调用都重跑编译器前端，
在整网里表现为每批次一次约 3.5 秒的编译，全部计入 TTFT。
缓存默认是开的（`PYPTO_CACHE` 缺省 "1"，root 缺省 `Path.home()/.cache/pypto/jit`），
所以要么策略被什么覆盖了，要么每次都走了 fallback —— `cache_stats()` 里有计数和
最近一次的诊断信息，直接读它，不猜。

用法（锁内）:
    export ASCEND_RT_VISIBLE_DEVICES=$TASK_DEVICE
    python cache_probe.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="PyPTO JIT 缓存诊断")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r: dict = {"stage": "import"}

    def save():
        (out / "result.json").write_text(json.dumps(r, indent=2, ensure_ascii=False, default=str))
        print(json.dumps(r, ensure_ascii=False, default=str), flush=True)

    try:
        import torch
        import torch_npu

        import decode_sparse_attn_csa as csa
        from pypto._cache_config import cache_stats, capture_cache_config
        from pypto.torch import init

        torch_npu.npu.set_device(args.device)

        cfg = capture_cache_config(None)
        r["policy"] = {"enabled": cfg.enabled, "root": str(cfg.root),
                       "readonly": cfg.readonly}
        r["root_exists"] = Path(cfg.root).exists() if cfg.root else None
        r["root_writable"] = os.access(str(Path(cfg.root).parent), os.W_OK) if cfg.root else None
        r["env"] = {k: os.environ.get(k) for k in
                    ("PYPTO_CACHE", "PYPTO_CACHE_DIR", "PYPTO_CACHE_READONLY",
                     "PYPTO_PROG_BUILD_DIR", "HOME")}
        r["stage"] = "init"
        save()

        init()
        specs = csa.build_tensor_specs()
        bufs = [sp.create_tensor().npu() for sp in specs]

        r["stage"] = "call1"
        save()
        t0 = time.perf_counter()
        csa.sparse_attn_test(*bufs)
        torch.npu.synchronize()
        r["call1_s"] = round(time.perf_counter() - t0, 2)
        r["stats_after_1"] = dict(cache_stats()._asdict()) if hasattr(cache_stats(), "_asdict") \
            else vars(cache_stats())
        save()

        # 第二次调用形状完全相同：命中缓存的话应该快一个量级
        r["stage"] = "call2"
        save()
        t0 = time.perf_counter()
        csa.sparse_attn_test(*bufs)
        torch.npu.synchronize()
        r["call2_s"] = round(time.perf_counter() - t0, 2)
        r["stats_after_2"] = dict(cache_stats()._asdict()) if hasattr(cache_stats(), "_asdict") \
            else vars(cache_stats())

        r["root_exists_after"] = Path(cfg.root).exists() if cfg.root else None
        if cfg.root and Path(cfg.root).exists():
            r["root_entries"] = len(list(Path(cfg.root).iterdir()))
        r["ok"] = True
        r["stage"] = "complete"
    except Exception:
        r["ok"] = False
        r["error"] = traceback.format_exc()
    finally:
        save()
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
