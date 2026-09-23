#!/usr/bin/env python3
"""离线重放 live 那一步的替换入参，定位翻译层在哪一段走偏。

服务端只能看到"输出不对"。这里把 dump 下来的**实际喂给 PTO kernel 的那组张量**
交给 pypto-lib 自己的 golden 重算一遍，于是能把问题逐段切开：

  golden(翻译后入参) vs PTO 输出            -> kernel 有没有照着算
  golden stage-1 vs vendor stage-1          -> 窗口/压缩槽的翻译对不对（不含投影）
  golden后半段(喂 vendor stage-1) vs vendor -> 逆 RoPE + o_proj 的建模对不对
  golden(翻译后入参) vs vendor 输出         -> 端到端还差多少

用法（cp310 环境，不需要卡）:
    python analyze_dump.py --dump <dir>
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import textwrap
from pathlib import Path

MARKER = "    rope_pair = o[..., NOPE_DIM:]"


def golden_with_attn_capture(csa, tensors):
    """跑 golden，并额外拿回 RoPE 之前的注意力输出（数学逐字不动，只插一行捕获）。"""
    src = textwrap.dedent(inspect.getsource(csa.golden_sparse_attn))
    if MARKER not in src:
        raise RuntimeError("golden 源码结构变了，找不到 RoPE 起点")
    src = src.replace(MARKER, '    tensors["__attn_only__"] = o.clone()\n' + MARKER, 1)
    src = src.replace("def golden_sparse_attn(", "def _golden_capture(", 1)
    ns = dict(vars(csa))
    exec(compile(src, "<golden_capture>", "exec"), ns)
    ns["_golden_capture"](tensors)
    return tensors.pop("__attn_only__")


def golden_tail(csa, tensors, attn_o):
    """只跑 golden 的后半段：拿现成的 stage-1 注意力，做逆 RoPE + grouped o_proj。

    取 golden 源码里 RoPE 起点之后的部分，前面换成"o 由调用方给定"，数学逐字不动。
    """
    src = textwrap.dedent(inspect.getsource(csa.golden_sparse_attn))
    _, tail = src.split(MARKER, 1)
    prologue = [
        "def _golden_tail_only(tensors, o):",
        "    import torch",
        "    wo_a = tensors['wo_a'].float()",
        "    wo_b_i8 = tensors['wo_b']",
        "    wo_b_scale = tensors['wo_b_scale'].float()",
        "    cos = tensors['freqs_cos'].float()",
        "    sin = tensors['freqs_sin'].float()",
        MARKER,
    ]
    body = "\n".join(prologue) + tail
    ns = dict(vars(csa))
    exec(compile(body, "<golden_tail>", "exec"), ns)
    ns["_golden_tail_only"](tensors, attn_o)
    return tensors["attn_out"]


def cmp(name, a, b):
    d = (a.float() - b.float()).abs()
    return {name: {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
                   "ref_abs_mean": b.float().abs().mean().item()}}


def main() -> int:
    p = argparse.ArgumentParser(description="PTO CSA 替换的 live dump 离线分析")
    p.add_argument("--dump", required=True)
    args = p.parse_args()

    import torch

    import decode_sparse_attn_csa as csa

    d = Path(args.dump)
    blob = torch.load(d / "step0.pt", weights_only=False)
    vendor = torch.load(d / "vendor0.pt", weights_only=False).float()
    g = {k[len("derived."):]: v for k, v in blob.items() if k.startswith("derived.")}

    n = int(g["n"]) if not hasattr(g["n"], "shape") else int(g["n"])
    S = csa.S
    take = torch.arange(n, dtype=torch.int64) * S
    r: dict = {"n": n}

    def build():
        return {
            "q": g["q_t"].to(torch.bfloat16),
            "ori_kv": g["ori_small"].to(torch.bfloat16),
            "window_swa_indices": g["win_new"].to(torch.int32),
            "cmp_kv": g["cmp_small"].to(torch.bfloat16),
            "cmp_block_table": g["cmp_bt"].to(torch.int32),
            "idx_topk": g["idx_t"].to(torch.int32),
            "position_ids": g["pos_t"].to(torch.int32).reshape(csa.T, 1),
            "attn_sink": g["sink"].float(),
            "freqs_cos": g["cos_t"].to(torch.bfloat16),
            "freqs_sin": g["sin_t"].to(torch.bfloat16),
            "wo_a": g["wo_a"].to(torch.bfloat16),
            "wo_b": g["wo_b"].to(torch.int8),
            "wo_b_scale": g["wo_b_scale"].float(),
            "attn_out": torch.zeros((csa.T, csa.D), dtype=torch.bfloat16),
        }

    tensors = build()
    attn_only = golden_with_attn_capture(csa, tensors)
    golden_rows = tensors["attn_out"].index_select(0, take).float()

    r.update(cmp("golden(翻译后入参) vs PTO 输出", golden_rows, g["pto_out"].float()))

    va = d / "vendor_attn0.pt"
    if va.exists():
        vattn = torch.load(va, weights_only=False).float()
        gattn = attn_only.index_select(0, take).float()
        r["stage1_shapes"] = {"golden": list(gattn.shape), "vendor": list(vattn.shape)}
        if gattn.shape == vattn.shape:
            r.update(cmp("golden stage-1 vs vendor stage-1（只有注意力）", gattn, vattn))
            # 逐 head 的差：个别 head 偏指向 sink/mask，全 head 偏指向 KV 选错
            ph = (gattn - vattn).abs().amax(dim=2).amax(dim=0)
            r["stage1_per_head_max_abs"] = {
                "worst8": sorted(ph.tolist(), reverse=True)[:8],
                "best8": sorted(ph.tolist())[:8],
            }
            # 按 head_dim 切：前 NOPE_DIM 不过 RoPE，后 ROPE_DIM 过
            dn = (gattn - vattn).abs()
            r["stage1_split"] = {
                "nope_max_abs": dn[..., :csa.NOPE_DIM].max().item(),
                "rope_max_abs": dn[..., csa.NOPE_DIM:].max().item(),
            }
            # 把 vendor 的 stage-1 接到 golden 的后半段，隔离逆 RoPE + o_proj
            t2 = build()
            full = torch.zeros((csa.T, csa.H, csa.HEAD_DIM))
            full[take] = vattn
            golden_tail(csa, t2, full)
            r.update(cmp("golden后半段(喂 vendor stage-1) vs vendor 输出",
                         t2["attn_out"].index_select(0, take).float(), vendor))

    r.update(cmp("golden(翻译后入参) vs vendor 输出", golden_rows, vendor))
    r.update(cmp("PTO 输出 vs vendor 输出", g["pto_out"].float(), vendor))

    win = g["win_new"]
    idx = g["idx_t"]
    r["inputs"] = {
        "window_valid_per_token": (win >= 0).sum(dim=1).tolist(),
        "window_slot_range": [int(win[win >= 0].min()), int(win.max())],
        "ori_small_rows": int(g["ori_small"].shape[0]),
        "idx_valid_per_token": (idx >= 0).sum(dim=1).tolist(),
        "idx_range": [int(idx[idx >= 0].min()), int(idx.max())],
        "cmp_bt_shape": list(g["cmp_bt"].shape),
        "cmp_small_rows": int(g["cmp_small"].shape[0]),
        "positions": g["pos_t"].reshape(-1).tolist(),
    }
    print(json.dumps(r, indent=2, ensure_ascii=False))
    (d / "analysis.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
