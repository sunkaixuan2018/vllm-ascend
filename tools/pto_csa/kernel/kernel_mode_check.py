#!/usr/bin/env python3
"""验证真实 CSA kernel 能否走 kernel 模式，以及能否被 NPUGraph 捕获重放。

按 D:\\PTO\\handoff\\pypto-kernel-mode-torch210.md 的三步与判据来，一步不减：
  1 kernel 模式 eager：拿 NPU 张量直接调 @pl.jit 对象（不是 op.compile(...)(...)）
  2 捕获进 NPUGraph 再 replay，结果正确
  3 原地改输入再 replay，输出必须跟着变

第 3 条最容易漏：capture/replay 如果把值烘死了，replay 会"成功"但结果不更新，
接到推理服务里表现为输出静止不动，很难定位。

与 handoff 原脚本的唯一差别：分支头的 417e5ed76（feat(torch): require explicit
pypto.torch.init）之后，每进程必须先 `pypto.torch.init()` 且在 capture 之外调用，
否则首次 kernel 调用直接报错。判据没动。

用法（锁内）:
    export ASCEND_RT_VISIBLE_DEVICES=$TASK_DEVICE
    python kernel_mode_check.py --out-dir <dir>
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
    p = argparse.ArgumentParser(description="CSA kernel 模式 + NPUGraph 捕获验证")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-2, help="handoff 判据：三项都 < 1e-2")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r: dict = {
        "scope": "真实 CSA kernel 的 kernel 模式 eager / NPUGraph capture / replay 跟随输入",
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "(未设置)"),
        "stage": "import",
    }

    def save():
        (out_dir / "result.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(json.dumps(r, ensure_ascii=False), flush=True)

    try:
        import torch
        import torch_npu

        import decode_sparse_attn_csa as csa  # pypto-lib models/deepseek_v4_flash_mtp

        r["python"] = sys.version.split()[0]
        r["torch"] = torch.__version__
        r["torch_npu"] = torch_npu.__version__
        import pypto
        r["pypto"] = pypto.__file__
        save()

        torch_npu.npu.set_device(args.device)

        # 每进程一次，且必须在首次 kernel 调用之前、graph capture 之外。
        r["stage"] = "pypto_torch_init"
        save()
        from pypto.torch import init

        init()
        r["init"] = "ok"

        r["stage"] = "fixture_and_golden"
        save()
        specs = csa.build_tensor_specs()
        host = {sp.name: sp.create_tensor() for sp in specs}
        golden_in = {k: v.clone() for k, v in host.items()}
        csa.golden_sparse_attn(golden_in)
        ref = golden_in["attn_out"].float()

        order = [sp.name for sp in specs]
        dev = {k: v.npu() for k, v in host.items()}

        # ---- 1. kernel 模式：直接拿 NPU 张量调 jit 对象 ----
        r["stage"] = "1_eager"
        save()
        t0 = time.perf_counter()
        csa.sparse_attn_test(*[dev[k] for k in order])
        torch.npu.synchronize()
        got = dev["attn_out"].cpu().float()
        r["eager"] = {"s": round(time.perf_counter() - t0, 3),
                      "max_abs": (got - ref).abs().max().item()}
        save()

        # 文档要求：捕获之前每个特化都要在捕获之外热身过。
        for _ in range(2):
            csa.sparse_attn_test(*[dev[k] for k in order])
        torch.npu.synchronize()

        # ---- 2. 捕获 + replay ----
        r["stage"] = "2_capture_replay"
        save()
        dev["attn_out"].zero_()
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph):
            csa.sparse_attn_test(*[dev[k] for k in order])
        torch.npu.synchronize()

        dev["attn_out"].zero_()
        graph.replay()
        torch.npu.synchronize()
        got = dev["attn_out"].cpu().float()
        r["replay"] = {"max_abs": (got - ref).abs().max().item()}
        save()

        # ---- 3. 改输入再 replay，输出必须跟着变 ----
        r["stage"] = "3_replay_follows_input"
        save()
        new_q = (host["q"].float() * 0.5).to(host["q"].dtype)
        dev["q"].copy_(new_q.npu())
        h2 = {k: v.clone() for k, v in host.items()}
        h2["q"] = new_q
        csa.golden_sparse_attn(h2)
        ref2 = h2["attn_out"].float()

        dev["attn_out"].zero_()
        graph.replay()
        torch.npu.synchronize()
        got2 = dev["attn_out"].cpu().float()
        r["replay_follows_input"] = {
            "max_abs_vs_new_golden": (got2 - ref2).abs().max().item(),
            "differs_from_first_replay": bool((got2 - got).abs().max().item() > 0),
        }

        # handoff 判据，原样执行
        tol = args.tol
        r["verdict"] = {
            "eager_ok": r["eager"]["max_abs"] < tol,
            "replay_ok": r["replay"]["max_abs"] < tol,
            "follows_ok": (r["replay_follows_input"]["max_abs_vs_new_golden"] < tol
                           and r["replay_follows_input"]["differs_from_first_replay"]),
        }
        r["ok"] = all(r["verdict"].values())
        r["stage"] = "complete"
    except Exception:
        r["ok"] = False
        r["error"] = traceback.format_exc()
    finally:
        save()
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
