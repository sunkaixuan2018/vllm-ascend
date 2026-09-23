#!/usr/bin/env python3
"""B 档门禁实验：PyPTO 的 dispatch 能否活在一个已被 torch_npu 占住设备的进程里。

替换 vLLM 层里的 CSA，前提是 PTO kernel 跑在 vLLM 的 worker 进程内 —— 那个进程
已经 `torch.npu.set_device()`、已经建了 NPU 上的 KV cache、还在跑 vendor 算子。
`coexist_check.py` 只证明了两边**能同时 import**，没证明 PyPTO 起 ChipWorker 时
不会和 torch_npu 抢设备。本脚本按 vLLM 里真实的先后顺序走一遍：

    1. torch_npu 建设备张量 + 跑一次 vendor CSA          （模拟 vLLM 已在工作）
    2. PyPTO 编译 + dispatch 同一个 CSA kernel            （模拟替换点）
    3. 再跑一次 vendor CSA，并检查步骤 1 的张量没被破坏   （模拟替换之后层继续跑）

三步都过，层级替换才谈得上可行。任何一步挂了，替换就得换成子进程/独立服务的形态。

用法（锁内，cp310 环境）:
    export ASCEND_RT_VISIBLE_DEVICES=$TASK_DEVICE
    python pto_in_vllm_process.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def vendor_csa_once(torch, csa, tensors):
    """按 A 档验证过的参数跑一次 vendor CSA，返回 host 上的输出。"""
    win = tensors["window_swa_indices"]
    pos = tensors["position_ids"][:, 0].to(torch.int64)
    S, T, BS = csa.S, csa.T, csa.BLOCK_SIZE
    WIN = int(win.shape[1])

    nseq = T
    cu = torch.arange(0, nseq + 1, dtype=torch.int32)
    seq = (pos + 1).to(torch.int32)
    nblocks = int(pos.max().item()) // BS + 2
    obt_h = torch.zeros((nseq, nblocks), dtype=torch.int32)
    for t in range(T):
        base = int(pos[t]) - (WIN - 1)
        for j, raw in enumerate(win[t].tolist()):
            slot = int(raw)
            if slot >= 0:
                obt_h[t, (base + j) // BS] = slot // BS

    raw_idx = tensors["idx_topk"][:, :csa.CMP_TOPK].to(torch.int64)
    bound = ((pos + 1) // csa.COMPRESS_RATIO).unsqueeze(1)
    keep = (raw_idx >= 0) & (raw_idx < bound)
    cmp_idx = torch.where(keep, raw_idx, torch.full_like(raw_idx, -1)).to(torch.int32)

    common = dict(cmp_ratio=csa.COMPRESS_RATIO, ori_mask_mode=4, cmp_mask_mode=3,
                  ori_win_left=WIN - 1, ori_win_right=0,
                  layout_q="TND", layout_kv="PA_ND")
    cu_d, seq_d = cu.npu(), seq.npu()
    cidx = cmp_idx.reshape(nseq, 1, -1).npu()
    meta = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=csa.H, num_heads_kv=1, head_dim=csa.HEAD_DIM,
        cu_seqlens_q=cu_d, seqused_kv=seq_d, batch_size=nseq,
        max_seqlen_q=1, max_seqlen_kv=int(seq.max()),
        cmp_topk=int(cidx.shape[-1]), has_ori_kv=True, has_cmp_kv=True, **common)
    out = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
        tensors["q"].npu(), ori_kv=tensors["ori_kv"].npu(), cmp_kv=tensors["cmp_kv"].npu(),
        cmp_sparse_indices=cidx, ori_block_table=obt_h.npu(),
        cmp_block_table=tensors["cmp_block_table"].to(torch.int32).repeat_interleave(S, dim=0).npu(),
        cu_seqlens_q=cu_d, seqused_kv=seq_d, sinks=tensors["attn_sink"].float().npu(),
        metadata=meta, softmax_scale=float(csa.SOFTMAX_SCALE), **common)[0]
    torch.npu.synchronize()
    return out.cpu().float()


def main() -> int:
    p = argparse.ArgumentParser(description="PTO dispatch 与 torch_npu 同进程共存门禁")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--platform", default="a2a3")
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r: dict = {
        "question": "PyPTO dispatch 能否活在已被 torch_npu 占住设备的进程里",
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "(未设置)"),
        "stage": "import",
    }

    def save():
        (out_dir / "result.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(json.dumps(r, ensure_ascii=False), flush=True)

    try:
        import torch
        import torch_npu  # noqa: F401
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env(include_vendor_lib=True)
        import vllm_ascend.vllm_ascend_C  # noqa: F401
        import decode_sparse_attn_csa as csa

        torch.npu.set_device(args.device)
        r["python"] = sys.version.split()[0]
        r["torch"] = torch.__version__

        # --- 1. torch_npu 先占住设备并干活 ---
        r["stage"] = "step1_torch_npu_first"
        save()
        specs = csa.build_tensor_specs()
        tensors = {sp.name: sp.create_tensor() for sp in specs}
        # 一块常驻的 NPU 张量，用来验证 PTO 起落之后 torch 的显存没被踩。
        canary = torch.arange(4096, dtype=torch.float32).npu()
        t0 = time.perf_counter()
        vendor_before = vendor_csa_once(torch, csa, tensors)
        r["step1"] = {"ok": True, "seconds": round(time.perf_counter() - t0, 2),
                      "vendor_out_abs_mean": vendor_before.abs().mean().item(),
                      "npu_mem_allocated_MiB": torch.npu.memory_allocated() / 2**20}

        # --- 2. 同一进程里编译并 dispatch PTO kernel ---
        r["stage"] = "step2_pto_dispatch"
        save()
        from golden import ratio_allclose, run

        t0 = time.perf_counter()
        res = run(
            fn=csa.sparse_attn_test,
            specs=specs,
            golden_fn=csa.golden_sparse_attn,
            config=dict(platform=args.platform, device_id=args.device,
                        enable_chip_swimlane=0, enable_dep_gen=False,
                        enable_pmu=0, dump_passes=False),
            rtol=1e-3, atol=1e-3,
            compare_fn={"attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
        )
        r["step2"] = {"ok": bool(res.passed), "error": res.error,
                      "seconds": round(time.perf_counter() - t0, 2),
                      "work_dir": str(res.work_dir) if res.work_dir else None}
        if not res.passed:
            r["ok"] = False
            r["stage"] = "pto_dispatch_failed"
            return 1

        # --- 3. PTO 落地之后 torch_npu 还能不能继续用 ---
        r["stage"] = "step3_torch_npu_after"
        save()
        canary_ok = bool(torch.equal(canary.cpu(),
                                     torch.arange(4096, dtype=torch.float32)))
        t0 = time.perf_counter()
        vendor_after = vendor_csa_once(torch, csa, tensors)
        d = (vendor_after - vendor_before).abs().max().item()
        r["step3"] = {"ok": True, "seconds": round(time.perf_counter() - t0, 2),
                      "canary_intact": canary_ok,
                      "vendor_rerun_max_abs_diff": d,
                      "vendor_rerun_bitwise_equal": d == 0.0,
                      "npu_mem_allocated_MiB": torch.npu.memory_allocated() / 2**20}
        r["ok"] = canary_ok and d == 0.0
        r["stage"] = "complete"
        r["verdict"] = ("可以在 vLLM worker 进程内就地替换" if r["ok"] else
                        "同进程共存不干净，替换要换成子进程/独立服务形态")
    except Exception:
        r["ok"] = False
        r["error"] = traceback.format_exc()
    finally:
        save()
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
