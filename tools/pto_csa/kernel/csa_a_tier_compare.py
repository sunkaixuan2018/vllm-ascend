#!/usr/bin/env python3
"""A 档：同一份 fixture，vendor 与 PTO 两条 CSA 路径分别对同一个 golden 比。

边界不一致是这件事的核心难点：
  PTO   `sparse_attn_test`           = 注意力 + RoPE + grouped o_proj（融合）
  vendor `npu_sparse_attn_sharedkv`  = 只有注意力
所以两者不能直接比输出。pypto-lib 的 `golden_sparse_attn` 恰好是分阶段写的
（先算 o[T,H,HEAD_DIM]，再 RoPE，最后 o_proj），于是：
  PTO 全路径    ←→ golden 全输出     （csa_b_tier.py --save-data 那一步做）
  vendor 注意力 ←→ golden stage-1 的 o（本脚本做）
两边吃同一份 fixture、对同一个参考，就能说明两条实现在各自边界上等价。

fixture 必须**取自 PTO 那一步落盘的 data/in**，不能各自现造：specs 的 init_value
用的是无种子的 torch.rand，同一份 specs 在两个进程里会给出不同的数。

stage-1 的 o 不手抄——手抄那段量化数学极易出错（本会话已栽过一次）。
改成把 golden 的源码取出来、在 RoPE 之前注入一行捕获，再 exec，
数学部分保持逐字不变。

用法（锁内，cp310 环境）:
    export ASCEND_RT_VISIBLE_DEVICES=$TASK_DEVICE
    python csa_a_tier_compare.py --out-dir <dir> --data-dir <pto work_dir>/data
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import textwrap
import traceback
from pathlib import Path


def load_fixture(data_dir, specs, torch):
    """读 PTO 那一步落盘的 in/*.pt；输出张量没落盘时按 spec 补零。"""
    in_dir = Path(data_dir) / "in"
    out = {}
    for sp in specs:
        f = in_dir / f"{sp.name}.pt"
        if f.exists():
            out[sp.name] = torch.load(f, weights_only=True).to(sp.dtype)
        else:
            out[sp.name] = torch.zeros(sp.shape, dtype=sp.dtype)
    return out


def materialize(specs, torch):
    """没有落盘 fixture 时的退路：让 spec 自己造，dtype 由它负责。"""
    return {sp.name: sp.create_tensor() for sp in specs}


def golden_with_attn_capture(csa, tensors):
    """跑 golden，并额外拿回 RoPE 之前的注意力输出。

    做法是取 golden_sparse_attn 的源码，在 `rope_pair = ...` 那行之前插一行捕获，
    其余逐字不动，然后在原模块的命名空间里 exec。
    """
    src = textwrap.dedent(inspect.getsource(csa.golden_sparse_attn))
    marker = "    rope_pair = o[..., NOPE_DIM:]"
    if marker not in src:
        raise RuntimeError("golden 源码结构变了，找不到 RoPE 起点；请重新核对注入点")
    src = src.replace(marker, '    tensors["__attn_only__"] = o.clone()\n' + marker, 1)
    src = src.replace("def golden_sparse_attn(", "def _golden_capture(", 1)
    ns = dict(vars(csa))
    exec(compile(src, "<golden_capture>", "exec"), ns)
    ns["_golden_capture"](tensors)
    return tensors.pop("__attn_only__")


def main() -> int:
    p = argparse.ArgumentParser(description="CSA A 档：vendor 注意力对 golden stage-1")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data-dir", default="", help="PTO 那一步的 work_dir/data（含 in/ out/）")
    p.add_argument("--atol", type=float, default=5e-3)
    p.add_argument("--rtol", type=float, default=2e-2)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r: dict = {
        "scope": "vendor npu_sparse_attn_sharedkv vs golden_sparse_attn stage-1 (pre-RoPE)",
        "fixture_source": args.data_dir or "(本进程现造，未与 PTO 对齐)",
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

        torch.npu.set_device(0)
        r["python"] = sys.version.split()[0]
        r["torch"] = torch.__version__
        r["shape_consts"] = {k: getattr(csa, k, None) for k in
                             ("T", "B", "S", "H", "HEAD_DIM", "BLOCK_SIZE", "COMPRESS_RATIO",
                              "TOPK", "CMP_TOPK", "PADDED_TOPK", "WIN", "IDX_TOPK",
                              "CMP_STORAGE_BLOCK_SIZE", "NOPE_DIM", "SOFTMAX_SCALE")}
        r["stage"] = "fixture"
        save()

        specs = csa.build_tensor_specs()
        tensors = (load_fixture(args.data_dir, specs, torch) if args.data_dir
                   else materialize(specs, torch))
        r["fixture"] = {k: [list(v.shape), str(v.dtype)] for k, v in tensors.items()}

        r["stage"] = "golden"
        save()
        attn_only = golden_with_attn_capture(csa, tensors)
        r["golden"] = {"attn_only_shape": list(attn_only.shape),
                       "full_out_shape": list(tensors["attn_out"].shape)}

        # 落盘的 out/attn_out.pt 是 PTO 那一步比对用的参考。本进程重算一遍 golden，
        # 两者应当逐位相同——不同就说明注入改动影响了数学，比对失去意义。
        ref = Path(args.data_dir) / "out" / "attn_out.pt" if args.data_dir else None
        if ref and ref.exists():
            saved = torch.load(ref, weights_only=True).float()
            d = (tensors["attn_out"].float() - saved).abs()
            r["golden"]["matches_saved_out"] = {"max_abs_diff": d.max().item(),
                                                "bitwise_equal": bool(d.max().item() == 0.0)}

        # ---- vendor 侧：只喂注意力需要的那部分 ----
        r["stage"] = "vendor_inputs"
        save()
        win = tensors["window_swa_indices"]
        pos = tensors["position_ids"][:, 0].to(torch.int64)
        S, T = csa.S, csa.T
        BS = csa.BLOCK_SIZE
        WIN = int(win.shape[1])

        # fixture 里同一条 batch 的 S 个 token 带**同一个** position_ids，窗口也因此相同
        # （见 init_position_ids / init_window_swa_indices）。vendor 的 TND 布局则把一条
        # 序列里第 i 个 query 摆在 seqused_kv-S+i 上，两者对不上。所以这里把 T 个 token
        # 各当成一条长度为 1 的 decode 序列，query 位置直接由各自的 seqused_kv-1 决定。
        nseq = T
        cu = torch.arange(0, nseq + 1, dtype=torch.int32)
        seq = (pos + 1).to(torch.int32)

        # window_swa_indices 存的是**物理 slot**（block_id*BLOCK_SIZE + intra），内容是
        # 一段连续历史加当前 token，当前 token 常落在另一个物理块，扁平看不连续——这正是
        # vendor 用 ori_block_table 表达的分页。这里从 slot 反推逻辑块→物理块：
        # 窗口第 j 项对应逻辑位置 pos-(WIN-1)+j。
        nblocks = int(pos.max().item()) // BS + 2
        obt_h = torch.zeros((nseq, nblocks), dtype=torch.int32)
        conflicts = []
        for t in range(T):
            base = int(pos[t]) - (WIN - 1)
            for j, raw in enumerate(win[t].tolist()):
                slot = int(raw)
                if slot < 0:
                    continue
                lpos = base + j
                lblk, pblk = lpos // BS, slot // BS
                if lpos % BS != slot % BS:
                    conflicts.append(("intra", t, j, lpos, slot))
                elif int(obt_h[t, lblk]) not in (0, pblk):
                    conflicts.append(("dup", t, lblk, int(obt_h[t, lblk]), pblk))
                else:
                    obt_h[t, lblk] = pblk
        r["block_table_derivation"] = {"sequences": nseq, "logical_blocks": nblocks,
                                       "conflicts": len(conflicts),
                                       "sample_conflicts": conflicts[:4]}
        if conflicts:
            r["ok"] = False
            r["stage"] = "block_table_mismatch"
            r["note"] = ("window_swa_indices 无法一致地映射成 逻辑块→物理块 的块表"
                         "（intra-block 偏移对不上或同一逻辑块映到多个物理块），"
                         "说明它不是分页滑窗语义，vendor 表达不了")
            return 1

        # golden 对压缩槽的掩码：保留 0 <= raw < floor((pos+1)/ratio)，否则 -1。
        raw_idx = tensors["idx_topk"][:, :csa.CMP_TOPK].to(torch.int64)
        bound = ((pos + 1) // csa.COMPRESS_RATIO).unsqueeze(1)
        keep = (raw_idx >= 0) & (raw_idx < bound)
        cmp_idx = torch.where(keep, raw_idx, torch.full_like(raw_idx, -1)).to(torch.int32)

        H, HD = csa.H, csa.HEAD_DIM
        q = tensors["q"].npu()
        ori_kv = tensors["ori_kv"].npu()
        cmp_kv = tensors["cmp_kv"].npu()
        sinks = tensors["attn_sink"].float().npu()
        cu_d = cu.npu()
        seq_d = seq.npu()
        obt = obt_h.npu()
        # cmp_block_table 是按 batch 给的，T 序列视角下每条 batch 的 S 个 token 复用同一行。
        cbt = tensors["cmp_block_table"].to(torch.int32).repeat_interleave(S, dim=0).npu()
        cidx = cmp_idx.reshape(nseq, 1, -1).npu()

        common = dict(cmp_ratio=csa.COMPRESS_RATIO, ori_mask_mode=4, cmp_mask_mode=3,
                      ori_win_left=WIN - 1, ori_win_right=0,
                      layout_q="TND", layout_kv="PA_ND")
        r["vendor_args"] = dict(common, batch_size=nseq, max_seqlen_q=1,
                                max_seqlen_kv=int(seq.max()),
                                cmp_topk=int(cidx.shape[-1]),
                                softmax_scale=float(csa.SOFTMAX_SCALE))

        r["stage"] = "vendor_run"
        save()
        meta = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=H, num_heads_kv=1, head_dim=HD,
            cu_seqlens_q=cu_d, seqused_kv=seq_d, batch_size=nseq,
            max_seqlen_q=1, max_seqlen_kv=int(seq.max()),
            cmp_topk=int(cidx.shape[-1]), has_ori_kv=True, has_cmp_kv=True, **common)
        torch.npu.synchronize()

        vendor_out = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
            q, ori_kv=ori_kv, cmp_kv=cmp_kv, cmp_sparse_indices=cidx,
            ori_block_table=obt, cmp_block_table=cbt, cu_seqlens_q=cu_d,
            seqused_kv=seq_d, sinks=sinks, metadata=meta,
            softmax_scale=float(csa.SOFTMAX_SCALE), **common)[0]
        torch.npu.synchronize()

        a = vendor_out.cpu().float()
        b = attn_only.float()
        r["vendor_vs_golden_attn"] = {"vendor_shape": list(a.shape),
                                      "golden_shape": list(b.shape)}
        if a.shape == b.shape:
            d = (a - b).abs()
            r["vendor_vs_golden_attn"].update({
                "max_abs_error": d.max().item(),
                "mean_abs_error": d.mean().item(),
                "within_tol": bool(torch.allclose(a, b, atol=args.atol, rtol=args.rtol)),
                "per_token_max_abs": d.flatten(1).max(dim=1).values.tolist(),
            })
            r["ok"] = r["vendor_vs_golden_attn"]["within_tol"]
        else:
            r["ok"] = False
            r["note"] = "vendor 与 golden 的注意力输出形状不一致，先对齐 layout 再比数值"
        r["stage"] = "complete"
    except Exception:
        r["ok"] = False
        r["error"] = traceback.format_exc()
    finally:
        save()
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
