#!/usr/bin/env python3
"""适配器离线验收：把 A 档那份 fixture 反造成 vLLM 的约定，过一遍 `PtoCsaRunner`，对 golden。

替换代码真正难的部分不是调用 kernel，而是两边数据约定的换算：
  · 压缩缓存页 32 槽 ↔ 128 槽，块表要跟着重映射
  · 窗口在 PTO 是显式物理槽，在 vLLM 是 块表 + position
  · wo_a 转置、wo_b 从 BF16 量化成 INT8 + per-channel scale
  · decode 批次 n(≤B) 要摊到 kernel 编译期固定的 (B, S)
这些错了，跑整网只会看到"输出不对"，定位不到是哪一步。所以先在这里单独验：
输入是 pypto-lib 自己的 fixture（按 vLLM 约定重新打包），参考是它自己的 golden，
中间完整走一遍 `run_decode`。过了，再谈接到服务里去。

用法（锁内，cp310 环境）:
    export ASCEND_RT_VISIBLE_DEVICES=$TASK_DEVICE
    python csa_adapter_check.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import types
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="PTO CSA vLLM 适配器离线验收")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--atol", type=float, default=2e-2)
    p.add_argument("--rtol", type=float, default=5e-2)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r: dict = {
        "scope": "PtoCsaRunner.run_decode 走 vLLM 约定的入参，对 pypto-lib golden",
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "(未设置)"),
        "stage": "import",
    }

    def save():
        (out_dir / "result.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(json.dumps(r, ensure_ascii=False), flush=True)

    try:
        import torch
        import torch_npu  # noqa: F401

        import decode_sparse_attn_csa as csa
        from vllm_ascend.attention.pto_csa import PtoCsaRunner

        torch.npu.set_device(args.device)
        r["python"] = sys.version.split()[0]
        r["stage"] = "fixture"
        save()

        specs = csa.build_tensor_specs()
        t = {sp.name: sp.create_tensor() for sp in specs}
        T, B, S = csa.T, csa.B, csa.S
        BS, WIN = csa.BLOCK_SIZE, csa.WIN
        CMP_PAGE, RATIO = csa.CMP_STORAGE_BLOCK_SIZE, csa.COMPRESS_RATIO
        PER_ORI = BS // CMP_PAGE

        r["stage"] = "golden"
        save()
        csa.golden_sparse_attn(t)
        golden_out = t["attn_out"].clone()

        # ---- 把 fixture 翻成 vLLM 的约定 ----
        r["stage"] = "to_vllm_layout"
        save()
        # 每条 batch 取 s=0 那个 token 当一条 vLLM decode 序列（S 个槽彼此独立）。
        pick = torch.arange(B, dtype=torch.int64) * S
        n = B

        # 窗口：PTO 给的是物理槽，vLLM 给的是块表 + position，所以反推块表。
        # A 档已经验过这份 fixture 的槽确实是分页滑窗语义（0 冲突）。
        win = t["window_swa_indices"]
        pos = t["position_ids"][:, 0].to(torch.int64)
        ncols = int(pos.max().item()) // BS + 2
        obt = torch.zeros((n, ncols), dtype=torch.int32)
        conflicts = 0
        for i, tk in enumerate(pick.tolist()):
            base = int(pos[tk]) - (WIN - 1)
            for j, raw in enumerate(win[tk].tolist()):
                slot = int(raw)
                if slot < 0:
                    continue
                lpos = base + j
                if lpos % BS != slot % BS:
                    conflicts += 1
                else:
                    obt[i, lpos // BS] = slot // BS
        r["ori_block_table"] = {"shape": list(obt.shape), "conflicts": conflicts}
        if conflicts:
            r["ok"] = False
            r["stage"] = "window_not_paged"
            return 1

        # 压缩缓存：vLLM 一页装 BLOCK_SIZE 个压缩槽，PTO 一页装 CMP_PAGE=32 个，
        # 线性地址相同，所以直接 reshape；块表按 4 合 1（fixture 的表是恒等，天然 4 对齐）。
        cmp_kv_vllm = t["cmp_kv"].reshape(-1, BS, *t["cmp_kv"].shape[2:])
        pto_cbt = t["cmp_block_table"].to(torch.int64)
        v_cols = (pto_cbt.shape[1] + PER_ORI - 1) // PER_ORI
        cbt_vllm = torch.zeros((n, v_cols), dtype=torch.int32)
        for i in range(n):
            for jv in range(v_cols):
                j0 = jv * PER_ORI
                if j0 >= pto_cbt.shape[1]:
                    continue
                phys = int(pto_cbt[i, j0])
                if phys % PER_ORI:
                    conflicts += 1
                cbt_vllm[i, jv] = phys // PER_ORI
        r["cmp_block_table"] = {"shape": list(cbt_vllm.shape), "conflicts": conflicts}
        if conflicts:
            r["ok"] = False
            r["stage"] = "cmp_table_not_4_aligned"
            r["note"] = ("fixture 的压缩块表不是 4 对齐，没法表示成 vLLM 的 128 槽页；"
                         "真实 vLLM 侧天生就是 128 槽页，这只是本用例的构造限制")
            return 1

        # 权重：vLLM 的 wo_a 是 [G, in, r]，wo_b 是 BF16（官方 checkpoint 就是这样）。
        wo_a_vllm = t["wo_a"].transpose(1, 2).contiguous()
        wo_b_vllm = (t["wo_b"].float() * t["wo_b_scale"].unsqueeze(-1)).to(torch.bfloat16)

        impl = types.SimpleNamespace(
            wo_a=types.SimpleNamespace(weight=wo_a_vllm.npu()),
            wo_b=types.SimpleNamespace(weight=wo_b_vllm.npu()),
            attn_sink=t["attn_sink"].npu(),
        )
        impl._pto_csa_stash = {
            "q": t["q"].index_select(0, pick).npu(),
            "ori_kv": t["ori_kv"].npu(),
            "cmp_kv": cmp_kv_vllm.npu(),
            "cmp_sparse_indices": t["idx_topk"].index_select(0, pick).npu(),
            "ori_block_table": obt.npu(),
            "cmp_block_table": cbt_vllm.npu(),
            "seqused_kv": (pos.index_select(0, pick) + 1).to(torch.int32).npu(),
            "positions": pos.index_select(0, pick).npu(),
        }
        # vLLM 的 RoPE 表是 [c0, c0, c1, c1, ...]（interleave 模式），pypto-lib 的是
        # "前半真值 + 后半复制"。这里按 vLLM 的排布造，才能真正验到 `_rope_table`。
        def to_vllm_rope(x):
            half = csa.ROPE_DIM // 2
            uniq = x.index_select(0, pick).float()[:, :half]
            return uniq.repeat_interleave(2, dim=1).reshape(n, 1, 1, -1).npu()

        cos = to_vllm_rope(t["freqs_cos"])
        sin = to_vllm_rope(t["freqs_sin"])

        r["stage"] = "run_decode"
        save()
        runner = PtoCsaRunner()
        got = runner.run_decode(impl, cos=cos, sin=sin)
        r["runner_stats"] = runner.stats
        if got is None:
            r["ok"] = False
            r["stage"] = "runner_fallback"
            r["note"] = "适配器判定不适用并回退，见 runner_stats.fallbacks"
            return 1

        ref = golden_out.index_select(0, pick).float()
        a = got.cpu().float()
        d = (a - ref).abs()
        r["adapter_vs_golden"] = {
            "shape": list(a.shape),
            "max_abs_error": d.max().item(),
            "mean_abs_error": d.mean().item(),
            "golden_abs_mean": ref.abs().mean().item(),
            "within_tol": bool(torch.allclose(a, ref, atol=args.atol, rtol=args.rtol)),
            "per_token_max_abs": d.max(dim=1).values.tolist(),
        }
        r["ok"] = r["adapter_vs_golden"]["within_tol"]
        r["stage"] = "complete"
    except Exception:
        r["ok"] = False
        r["error"] = traceback.format_exc()
    finally:
        save()
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
