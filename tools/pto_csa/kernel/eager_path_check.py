#!/usr/bin/env python3
"""验证 CSA kernel 能不能走 PyPTO 的 **eager 调用路径**，以及能不能被 NPUGraph 捕获。

`JITFunction.__call__` 的文档说它是"把 kernel 挂到当前 torch NPU 流上"，吃 NPU 张量，
并且"图捕获要求每个算子特化先在捕获之外热身"；而 `op.compile(...)(...)` 那条才是
program 执行、吃 CPU 张量的路径。接 vLLM 时我先前用的是后者，于是不得不把 KV 页
搬到 host，也就不可能被 aclgraph 捕获。

这里对**真实的 CSA kernel**（不是玩具 add）把三件事逐一验掉：
  1. 直接用 NPU 张量调 `sparse_attn_test(...)`，结果对不对
  2. 捕获进 NPUGraph 再 replay，结果对不对
  3. replay 是否真的跟着输入变（改 q 再 replay，输出应当跟着变）

第 3 条是关键：capture/replay 如果把某一步的值烘死了，replay 会"成功"但结果不更新，
这种错误在服务里表现为输出静止不动，很难定位。

用法（锁内，cp310 环境）:
    export ASCEND_RT_VISIBLE_DEVICES=$TASK_DEVICE
    python eager_path_check.py --out-dir <dir>
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
    p = argparse.ArgumentParser(description="PyPTO eager 调用路径 + NPUGraph 捕获验证")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r: dict = {
        "question": "CSA kernel 能否走 eager 路径吃 NPU 张量，并被 NPUGraph 捕获重放",
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "(未设置)"),
        "stage": "import",
    }

    def save():
        (out_dir / "result.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(json.dumps(r, ensure_ascii=False), flush=True)

    try:
        import torch
        import torch_npu

        import decode_sparse_attn_csa as csa

        torch.npu.set_device(args.device)
        r["python"] = sys.version.split()[0]
        r["stage"] = "fixture"
        save()

        specs = csa.build_tensor_specs()
        host = {sp.name: sp.create_tensor() for sp in specs}
        golden_in = {k: v.clone() for k, v in host.items()}
        csa.golden_sparse_attn(golden_in)
        ref = golden_in["attn_out"].float()

        order = [sp.name for sp in specs]
        dev = {k: v.npu() for k, v in host.items()}
        r["arg_order"] = order
        r["stage"] = "eager_call"
        save()

        # --- 1. eager 路径：直接拿 NPU 张量调 jit 对象 ---
        t0 = time.perf_counter()
        csa.sparse_attn_test(*[dev[k] for k in order])
        torch.npu.synchronize()
        eager_s = time.perf_counter() - t0
        got = dev["attn_out"].cpu().float()
        d = (got - ref).abs()
        r["eager"] = {"ok": True, "first_call_s": round(eager_s, 3),
                      "max_abs_error": d.max().item(),
                      "mean_abs_error": d.mean().item(),
                      "ref_abs_mean": ref.abs().mean().item()}
        save()

        # 文档要求：捕获之前每个特化都要在捕获之外热身过。
        for _ in range(2):
            csa.sparse_attn_test(*[dev[k] for k in order])
        torch.npu.synchronize()

        # --- 2. 捕获进 NPUGraph ---
        r["stage"] = "capture"
        save()
        dev["attn_out"].zero_()
        graph = torch_npu.npu.NPUGraph()
        t0 = time.perf_counter()
        with torch_npu.npu.graph(graph):
            csa.sparse_attn_test(*[dev[k] for k in order])
        torch.npu.synchronize()
        r["capture"] = {"ok": True, "seconds": round(time.perf_counter() - t0, 3)}
        save()

        # --- 3. replay，并确认 replay 跟着输入走 ---
        r["stage"] = "replay"
        save()
        dev["attn_out"].zero_()
        t0 = time.perf_counter()
        graph.replay()
        torch.npu.synchronize()
        replay_s = time.perf_counter() - t0
        got = dev["attn_out"].cpu().float()
        d = (got - ref).abs()
        r["replay"] = {"ok": True, "seconds": round(replay_s, 4),
                       "max_abs_error": d.max().item(),
                       "mean_abs_error": d.mean().item()}

        # 原地改 q 再 replay：输出必须跟着变，否则说明值被烘进图里了
        new_q = (host["q"].float() * 0.5).to(host["q"].dtype)
        dev["q"].copy_(new_q.npu())
        host2 = {k: v.clone() for k, v in host.items()}
        host2["q"] = new_q
        csa.golden_sparse_attn(host2)
        ref2 = host2["attn_out"].float()
        dev["attn_out"].zero_()
        graph.replay()
        torch.npu.synchronize()
        got2 = dev["attn_out"].cpu().float()
        d2 = (got2 - ref2).abs()
        r["replay_follows_input"] = {
            "max_abs_error_vs_new_golden": d2.max().item(),
            "differs_from_first_replay": bool((got2 - got).abs().max().item() > 0),
        }
        r["ok"] = (r["eager"]["max_abs_error"] < 1e-2
                   and r["replay"]["max_abs_error"] < 1e-2
                   and d2.max().item() < 1e-2
                   and r["replay_follows_input"]["differs_from_first_replay"])
        r["stage"] = "complete"
        r["verdict"] = ("eager 路径可用且可被 NPUGraph 捕获重放 —— 适配器应当改走这条"
                        if r["ok"] else "这条路对 CSA kernel 不成立，见上面各段误差")
    except Exception:
        r["ok"] = False
        r["error"] = traceback.format_exc()
    finally:
        save()
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
