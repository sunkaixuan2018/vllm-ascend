#!/usr/bin/env python3
"""Single-device functional gate for the TP1, S=6 native-layout CSA entry."""

from __future__ import annotations

import argparse
import os
import sys

import torch

TARGET_BATCHES = (4, 8, 16, 24, 32, 40)
START_POS = 131_066


def _add_model_helpers() -> None:
    # Alternatively provide pypto-lib and its model helpers through PYTHONPATH.
    model_dir = os.environ.get("PYPTO_LIB_MODEL_DIR")
    if model_dir and model_dir not in sys.path:
        sys.path.insert(0, model_dir)


def _tables(kernel, batch: int, seq: int, start_pos: int):
    positions = torch.arange(
        start_pos,
        start_pos + seq,
        dtype=torch.int64,
    ).repeat(batch, 1)
    max_position = int(positions.max())
    state_width = max_position // kernel.VLLM_COMPRESS_STATE_LIVE_ROWS + 1
    raw_width = max_position // kernel.VLLM_KV_PAGE_ROWS + 1
    compressed_rows = (max_position + 1) // kernel.COMPRESS_RATIO
    compressed_width = max(1, (compressed_rows + kernel.VLLM_KV_PAGE_ROWS - 1) // kernel.VLLM_KV_PAGE_ROWS)

    main = torch.zeros(batch, state_width, dtype=torch.int32)
    raw = torch.zeros(batch, raw_width, dtype=torch.int32)
    compressed = torch.zeros(batch, compressed_width, dtype=torch.int32)
    inner = torch.zeros(batch, state_width, dtype=torch.int32)
    index = torch.zeros(batch, compressed_width, dtype=torch.int32)

    first_cmp_page = max(0, (start_pos // 4 - 1) // 128)
    next_page = first_cmp_page + 1
    raw_next = 1
    for request in range(batch):
        history = torch.arange(1, first_cmp_page + 1, dtype=torch.int32)
        compressed[request, :first_cmp_page] = history
        index[request, :first_cmp_page] = history
        for column in range(first_cmp_page, compressed_width):
            compressed[request, column] = next_page
            index[request, column] = next_page
            next_page += 1
        for column in range(max(0, start_pos - 7) // 8, state_width):
            main[request, column] = next_page
            inner[request, column] = next_page
            next_page += 1
        for column in range(max(0, start_pos - 127) // 128, raw_width):
            raw[request, column] = raw_next
            raw_next += 1

    return {
        "positions": positions.reshape(-1).contiguous(),
        "main": main,
        "raw": raw,
        "compressed": compressed,
        "inner": inner,
        "index": index,
        "main_pages": next_page,
        "raw_pages": raw_next,
        "compressed_pages": next_page,
        "shared_pages": next_page,
        "seq": seq,
        "start_pos": start_pos,
    }


def _shared_pages(kernel, layout):
    pages = torch.zeros(
        layout["shared_pages"],
        kernel.VLLM_INDEX_PAGE_ROWS,
        kernel.IDX_HEAD_DIM,
        dtype=torch.int8,
    )
    scales = (
        pages[
            :,
            kernel.VLLM_KV_PAGE_ROWS : kernel.VLLM_INDEX_PAGE_ROWS,
        ]
        .reshape(layout["shared_pages"], -1)
        .view(torch.float16)
    )
    scales.fill_(1.0)
    positions = layout["positions"].reshape(-1, layout["seq"])
    for request in range(positions.shape[0]):
        for position in positions[request].tolist():
            inner_page = int(layout["inner"][request, position // 8])
            inner_row = position % 8
            inner_state_dim = 2 * kernel.INNER_OUT_DIM
            byte_begin = inner_row * inner_state_dim * 4
            byte_end = byte_begin + inner_state_dim * 4
            pages[inner_page].reshape(-1)[byte_begin:byte_end].fill_(7)
            if (position + 1) % kernel.COMPRESS_RATIO == 0:
                compressed_row = (position + 1) // kernel.COMPRESS_RATIO - 1
                index_page = int(layout["index"][request, compressed_row // 128])
                intra = compressed_row % kernel.VLLM_KV_PAGE_ROWS
                pages[index_page, intra].fill_(7)
                scales[index_page, intra] = 3.0
    return pages


def build_specs(kernel, batch: int, valid_tokens: int, seq: int = 6, start_pos: int = START_POS):
    from golden import TensorSpec

    if batch not in TARGET_BATCHES:
        raise ValueError(f"batch must be one of {TARGET_BATCHES}, got {batch}")
    if kernel.TP_SIZE != 1 or kernel.S != 6 or kernel.B != 64:
        raise ValueError(f"expected TP1/S6/B64, got TP{kernel.TP_SIZE}/S{kernel.S}/B{kernel.B}")
    if not 0 <= valid_tokens <= seq <= kernel.S:
        raise ValueError("require 0 <= valid_tokens <= seq <= S")
    layout = _tables(kernel, batch, seq, start_pos)
    tokens = batch * seq
    positions = layout["positions"]
    token_valid = torch.zeros(batch, seq, dtype=torch.int32)
    token_valid[:, :valid_tokens] = 1
    token_valid = token_valid.reshape(-1)
    kv_seq_lens = torch.full(
        [batch],
        start_pos + valid_tokens,
        dtype=torch.int32,
    )
    main_state = torch.zeros(
        layout["main_pages"],
        kernel.VLLM_COMPRESS_STATE_PAGE_ROWS,
        kernel.MAIN_STATE_DIM,
        dtype=torch.float32,
    )
    raw_pages = torch.zeros(
        layout["raw_pages"],
        kernel.VLLM_KV_PAGE_ROWS,
        1,
        kernel.HEAD_DIM,
        dtype=torch.bfloat16,
    )
    compressed_pages = torch.zeros(
        layout["compressed_pages"],
        kernel.VLLM_KV_PAGE_ROWS,
        1,
        kernel.HEAD_DIM,
        dtype=torch.bfloat16,
    )
    shared_pages = _shared_pages(kernel, layout)
    for request in range(batch):
        for position in positions.reshape(batch, seq)[request].tolist():
            state_page = int(layout["main"][request, position // 8])
            raw_page = int(layout["raw"][request, position // 128])
            main_state[state_page, position % 8].fill_(-5.0)
            raw_pages[raw_page, position % 128].fill_(-5.0)
            if (position + 1) % kernel.COMPRESS_RATIO == 0:
                row = (position + 1) // kernel.COMPRESS_RATIO - 1
                cmp_page = int(layout["compressed"][request, row // 128])
                compressed_pages[cmp_page, row % 128].fill_(-5.0)

    zbf16 = lambda shape: torch.zeros(shape, dtype=torch.bfloat16)
    zfp32 = lambda shape: torch.zeros(shape, dtype=torch.float32)
    zi8 = lambda shape: torch.zeros(shape, dtype=torch.int8)

    specs = [
        TensorSpec("x_normed", [tokens, kernel.D], torch.bfloat16, init_value=lambda: zbf16([tokens, kernel.D])),
        TensorSpec(
            "wq_a", [kernel.D, kernel.Q_LORA], torch.bfloat16, init_value=lambda: zbf16([kernel.D, kernel.Q_LORA])
        ),
        TensorSpec(
            "wq_b",
            [kernel.Q_LORA, kernel.H * kernel.HEAD_DIM],
            torch.int8,
            init_value=lambda: zi8([kernel.Q_LORA, kernel.H * kernel.HEAD_DIM]),
        ),
        TensorSpec(
            "wq_b_scale",
            [kernel.H * kernel.HEAD_DIM],
            torch.float32,
            init_value=lambda: zfp32([kernel.H * kernel.HEAD_DIM]),
        ),
        TensorSpec(
            "wkv", [kernel.D, kernel.HEAD_DIM], torch.bfloat16, init_value=lambda: zbf16([kernel.D, kernel.HEAD_DIM])
        ),
        TensorSpec(
            "gamma_cq",
            [kernel.Q_LORA],
            torch.bfloat16,
            init_value=lambda: torch.ones(kernel.Q_LORA, dtype=torch.bfloat16),
        ),
        TensorSpec(
            "gamma_ckv",
            [kernel.HEAD_DIM],
            torch.bfloat16,
            init_value=lambda: torch.ones(kernel.HEAD_DIM, dtype=torch.bfloat16),
        ),
    ]
    for name, fill in (
        ("freqs_cos", 1.0),
        ("freqs_sin", 0.0),
        ("cmp_freqs_cos", 1.0),
        ("cmp_freqs_sin", 0.0),
    ):
        specs.append(
            TensorSpec(
                name,
                [start_pos + seq, kernel.ROPE_HEAD_DIM],
                torch.float32,
                init_value=lambda fill=fill: torch.full(
                    [start_pos + seq, kernel.ROPE_HEAD_DIM],
                    fill,
                    dtype=torch.float32,
                ),
            )
        )
    specs.extend(
        [
            TensorSpec(
                "cmp_wkv",
                [kernel.MAIN_OUT_DIM, kernel.D],
                torch.bfloat16,
                init_value=lambda: zbf16([kernel.MAIN_OUT_DIM, kernel.D]),
            ),
            TensorSpec(
                "cmp_wgate",
                [kernel.MAIN_OUT_DIM, kernel.D],
                torch.bfloat16,
                init_value=lambda: zbf16([kernel.MAIN_OUT_DIM, kernel.D]),
            ),
            TensorSpec(
                "cmp_ape",
                [kernel.COMPRESS_RATIO, kernel.MAIN_OUT_DIM],
                torch.float32,
                init_value=lambda: zfp32([kernel.COMPRESS_RATIO, kernel.MAIN_OUT_DIM]),
            ),
            TensorSpec(
                "cmp_norm_w",
                [kernel.HEAD_DIM],
                torch.bfloat16,
                init_value=lambda: torch.ones(kernel.HEAD_DIM, dtype=torch.bfloat16),
            ),
            TensorSpec(
                "compress_state_pages", list(main_state.shape), torch.float32, init_value=lambda: main_state.clone()
            ),
            TensorSpec("kv_cache_pages", list(raw_pages.shape), torch.bfloat16, init_value=lambda: raw_pages.clone()),
            TensorSpec(
                "cmp_kv_pages",
                list(compressed_pages.shape),
                torch.bfloat16,
                init_value=lambda: compressed_pages.clone(),
            ),
            TensorSpec(
                "compress_state_block_table",
                list(layout["main"].shape),
                torch.int32,
                init_value=lambda: layout["main"].clone(),
            ),
            TensorSpec(
                "idx_wq_b",
                [kernel.Q_LORA, kernel.IDX_N_HEADS * kernel.IDX_HEAD_DIM],
                torch.int8,
                init_value=lambda: zi8([kernel.Q_LORA, kernel.IDX_N_HEADS * kernel.IDX_HEAD_DIM]),
            ),
            TensorSpec(
                "idx_wq_b_scale",
                [kernel.IDX_N_HEADS * kernel.IDX_HEAD_DIM],
                torch.float32,
                init_value=lambda: zfp32([kernel.IDX_N_HEADS * kernel.IDX_HEAD_DIM]),
            ),
            TensorSpec(
                "weights_proj",
                [kernel.D, kernel.IDX_N_HEADS],
                torch.bfloat16,
                init_value=lambda: zbf16([kernel.D, kernel.IDX_N_HEADS]),
            ),
            TensorSpec(
                "hadamard_idx",
                [kernel.IDX_HEAD_DIM, kernel.IDX_HEAD_DIM],
                torch.bfloat16,
                init_value=lambda: torch.eye(kernel.IDX_HEAD_DIM, dtype=torch.bfloat16),
            ),
            TensorSpec(
                "inner_wkv",
                [kernel.INNER_OUT_DIM, kernel.D],
                torch.bfloat16,
                init_value=lambda: zbf16([kernel.INNER_OUT_DIM, kernel.D]),
            ),
            TensorSpec(
                "inner_wgate",
                [kernel.INNER_OUT_DIM, kernel.D],
                torch.bfloat16,
                init_value=lambda: zbf16([kernel.INNER_OUT_DIM, kernel.D]),
            ),
            TensorSpec(
                "inner_ape",
                [kernel.COMPRESS_RATIO, kernel.INNER_OUT_DIM],
                torch.float32,
                init_value=lambda: zfp32([kernel.COMPRESS_RATIO, kernel.INNER_OUT_DIM]),
            ),
            TensorSpec(
                "inner_norm_w",
                [kernel.IDX_HEAD_DIM],
                torch.bfloat16,
                init_value=lambda: torch.ones(kernel.IDX_HEAD_DIM, dtype=torch.bfloat16),
            ),
            TensorSpec(
                "inner_index_pages", list(shared_pages.shape), torch.int8, init_value=lambda: shared_pages.clone()
            ),
            TensorSpec(
                "inner_compress_state_block_table",
                list(layout["inner"].shape),
                torch.int32,
                init_value=lambda: layout["inner"].clone(),
            ),
            TensorSpec(
                "ori_block_table", list(layout["raw"].shape), torch.int32, init_value=lambda: layout["raw"].clone()
            ),
            TensorSpec(
                "cmp_block_table",
                list(layout["compressed"].shape),
                torch.int32,
                init_value=lambda: layout["compressed"].clone(),
            ),
            TensorSpec(
                "index_block_table",
                list(layout["index"].shape),
                torch.int32,
                init_value=lambda: layout["index"].clone(),
            ),
            TensorSpec("position_ids", [tokens], torch.int64, init_value=lambda: positions.clone()),
            TensorSpec("token_valid", [tokens], torch.int32, init_value=lambda: token_valid.clone()),
            TensorSpec("kv_seq_lens", [batch], torch.int32, init_value=lambda: kv_seq_lens.clone()),
            TensorSpec("attn_sink", [kernel.H], torch.float32, init_value=lambda: zfp32([kernel.H])),
            TensorSpec(
                "wo_a",
                [kernel.O_GROUPS, kernel.O_LORA, kernel.O_GROUP_IN],
                torch.bfloat16,
                init_value=lambda: zbf16([kernel.O_GROUPS, kernel.O_LORA, kernel.O_GROUP_IN]),
            ),
            TensorSpec(
                "wo_b",
                [kernel.D, kernel.O_GROUPS * kernel.O_LORA],
                torch.int8,
                init_value=lambda: zi8([kernel.D, kernel.O_GROUPS * kernel.O_LORA]),
            ),
            TensorSpec("wo_b_scale", [kernel.D], torch.float32, init_value=lambda: zfp32([kernel.D])),
            TensorSpec("attn_out", [tokens, kernel.D], torch.bfloat16),
        ]
    )
    return specs, layout


def golden(kernel, layout, tensors) -> None:
    tensors["attn_out"].zero_()
    positions = layout["positions"].reshape(-1, layout["seq"])
    shared = tensors["inner_index_pages"]
    scales = (
        shared[
            :,
            kernel.VLLM_KV_PAGE_ROWS : kernel.VLLM_INDEX_PAGE_ROWS,
        ]
        .reshape(shared.shape[0], -1)
        .view(torch.float16)
    )
    zero_scale = torch.tensor(
        1.0e-4 / 127.0,
        dtype=torch.float16,
    )
    for request in range(positions.shape[0]):
        for step, position in enumerate(positions[request].tolist()):
            token = request * layout["seq"] + step
            if not int(tensors["token_valid"][token]):
                continue
            state_page = int(layout["main"][request, position // 8])
            raw_page = int(layout["raw"][request, position // 128])
            inner_page = int(layout["inner"][request, position // 8])
            tensors["compress_state_pages"][state_page, position % 8].zero_()
            tensors["kv_cache_pages"][raw_page, position % 128].zero_()
            inner_state_dim = 2 * kernel.INNER_OUT_DIM
            byte_begin = (position % 8) * inner_state_dim * 4
            byte_end = byte_begin + inner_state_dim * 4
            shared[inner_page].reshape(-1)[byte_begin:byte_end].zero_()
            if (position + 1) % kernel.COMPRESS_RATIO == 0:
                compressed_row = (position + 1) // kernel.COMPRESS_RATIO - 1
                cmp_page = int(layout["compressed"][request, compressed_row // 128])
                index_page = int(layout["index"][request, compressed_row // 128])
                intra = compressed_row % kernel.VLLM_KV_PAGE_ROWS
                tensors["cmp_kv_pages"][cmp_page, intra].zero_()
                shared[index_page, intra].zero_()
                scales[index_page, intra] = zero_scale


def nonzero_specs(kernel, specs, compressors=False):
    """Exercise QKV RoPE and inverse RoPE with nonzero attention/projection.

    By default compressors remain zero, making Top-K ties harmless. The
    compressors option also exercises their projections and the indexer.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(62)
    replacements = {}
    for spec in specs:
        if spec.name in ("x_normed", "wq_a", "wkv", "wo_a") or (
            compressors
            and spec.name
            in (
                "cmp_wkv",
                "cmp_wgate",
                "cmp_ape",
                "inner_wkv",
                "inner_wgate",
                "inner_ape",
                "weights_proj",
            )
        ):
            replacements[spec.name] = (torch.randn(spec.shape, generator=generator) * 0.02).to(spec.dtype)
        elif spec.name in ("wq_b", "wo_b") or (compressors and spec.name == "idx_wq_b"):
            replacements[spec.name] = torch.randint(-3, 4, spec.shape, generator=generator, dtype=torch.int8)
        elif spec.name in ("wq_b_scale", "wo_b_scale") or (compressors and spec.name == "idx_wq_b_scale"):
            replacements[spec.name] = torch.full(spec.shape, 0.01)
        elif spec.name in ("freqs_cos", "freqs_sin", "cmp_freqs_cos", "cmp_freqs_sin"):
            # Position- and frequency-dependent values catch wrong row/layout selection.
            angles = torch.arange(spec.shape[0]).float()[:, None] * (
                0.003 + torch.arange(kernel.ROPE_HEAD_DIM // 2).float()[None, :] * 0.001
            )
            replacements[spec.name] = (angles.cos() if "cos" in spec.name else angles.sin()).repeat_interleave(2, -1)
    return [
        TensorSpec(spec.name, spec.shape, spec.dtype, init_value=replacements[spec.name])
        if spec.name in replacements
        else spec
        for spec in specs
    ]


def golden_compressors(kernel, layout, tensors):
    """Independent paged reference; no private ring or fixed-S fixture helpers."""
    from vllm_ascend.attention.pto_kernels.dspark.decode_compressor_ratio4 import EPS

    shared = tensors["inner_index_pages"]
    inner_state = shared.view(torch.float32).reshape(shared.shape[0], -1)
    state_dim = 2 * kernel.INNER_OUT_DIM
    inner_state = inner_state[:, : 8 * state_dim].view(-1, 8, state_dim)
    scales = shared[:, 128:130].reshape(shared.shape[0], -1).view(torch.float16)
    positions = tensors["position_ids"]
    for prefix, state, table_name, dim in (
        ("cmp", tensors["compress_state_pages"], "compress_state_block_table", kernel.HEAD_DIM),
        ("inner", inner_state, "inner_compress_state_block_table", kernel.IDX_HEAD_DIM),
    ):
        table = tensors[table_name]
        out_dim = 2 * dim
        values = tensors["x_normed"].float() @ tensors[prefix + "_wkv"].float().T
        scores = tensors["x_normed"].float() @ tensors[prefix + "_wgate"].float().T
        scores += tensors[prefix + "_ape"][positions % 4]
        old_state = state.clone()
        for token, position in enumerate(positions.tolist()):
            if not int(tensors["token_valid"][token]):
                continue
            request = token // layout["seq"]
            first = int(positions[request * layout["seq"]])
            page = int(table[request, position // 8])
            state[page, position % 8, :out_dim] = values[token]
            state[page, position % 8, out_dim:] = scores[token]
            if (position + 1) % 4:
                continue
            window_values, window_scores = [], []
            for offset, logical in enumerate(range(position - 7, position + 1)):
                half = 0 if offset < 4 else dim
                value = torch.zeros(dim)
                score = torch.full((dim,), float("-inf"))
                if 0 <= logical < first:
                    history_page = int(table[request, logical // 8])
                    if history_page > 0:
                        value = old_state[history_page, logical % 8, half : half + dim]
                        score = old_state[history_page, logical % 8, out_dim + half : out_dim + half + dim]
                elif first <= logical <= position:
                    overlay = request * layout["seq"] + logical - first
                    if int(tensors["token_valid"][overlay]):
                        value = values[overlay, half : half + dim]
                        score = scores[overlay, half : half + dim]
                window_values.append(value)
                window_scores.append(score)
            pooled = (torch.stack(window_values) * torch.stack(window_scores).softmax(0)).sum(0)
            normed = pooled * torch.rsqrt(pooled.square().mean() + EPS) * tensors[prefix + "_norm_w"].float()
            rope_pos = position + 1 - 4
            rope = normed[-kernel.ROPE_HEAD_DIM :]
            signed_sin = tensors["cmp_freqs_sin"][rope_pos].clone()
            signed_sin[::2] *= -1
            normed[-kernel.ROPE_HEAD_DIM :] = (
                rope * tensors["cmp_freqs_cos"][rope_pos] + rope.reshape(-1, 2).flip(-1).reshape(-1) * signed_sin
            )
            logical_row = (position + 1) // 4 - 1
            if prefix == "cmp":
                dst_page = int(tensors["cmp_block_table"][request, logical_row // 128])
                tensors["cmp_kv_pages"][dst_page, logical_row % 128, 0] = normed.bfloat16()
            else:
                projected = (normed.bfloat16().float() @ tensors["hadamard_idx"].float()).bfloat16().float()
                amax = projected.abs().amax().clamp_min(1.0e-4)
                quantized = torch.round(projected * (127.0 / amax)).to(torch.int8)
                dst_page = int(tensors["index_block_table"][request, logical_row // 128])
                shared[dst_page, logical_row % 128] = quantized
                scales[dst_page, logical_row % 128] = amax / 127.0


def golden_nonzero(kernel, layout, tensors, compressors=False):
    from vllm_ascend.attention.pto_kernels.dspark.decode_o_proj import golden_decode_o_proj_tp1
    from vllm_ascend.attention.pto_kernels.dspark.decode_sparse_attn_csa import golden_sparse_attn
    from vllm_ascend.attention.pto_kernels.dspark.qkv_proj_rope import golden_qkv_proj_rope

    if compressors:
        golden_compressors(kernel, layout, tensors)
    else:
        golden(kernel, layout, tensors)
    positions = tensors["position_ids"]
    tokens = positions.numel()
    cos = tensors["freqs_cos"][positions, ::2].repeat(1, 2)
    sin = tensors["freqs_sin"][positions, ::2].repeat(1, 2)
    q = torch.zeros(tokens, kernel.H, kernel.HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(tokens, kernel.HEAD_DIM, dtype=torch.bfloat16)
    inputs = dict(tensors)
    inputs.update(
        x=tensors["x_normed"],
        rope_cos=cos,
        rope_sin=sin,
        q=q,
        kv=kv,
        qr=torch.zeros(tokens, kernel.Q_LORA, dtype=torch.int8),
        qr_scale=torch.zeros(tokens, 1),
    )
    golden_qkv_proj_rope(inputs)
    windows = torch.full((tokens, kernel.M.sliding_window), -1, dtype=torch.int32)
    topk = torch.full((tokens, kernel.IDX_TOPK), -1, dtype=torch.int32)
    for token, position in enumerate(positions.tolist()):
        if not tensors["token_valid"][token]:
            continue
        request = token // layout["seq"]
        page = int(tensors["ori_block_table"][request, position // 128])
        tensors["kv_cache_pages"][page, position % 128, 0] = kv[token]
        logical = torch.arange(max(0, position + 1 - kernel.M.sliding_window), position + 1)
        pages = tensors["ori_block_table"][request, logical // 128]
        windows[token, : logical.numel()] = (pages * 128 + logical % 128).int()
        visible = min((position + 1) // 4, kernel.IDX_TOPK)
        topk[token, :visible] = torch.arange(visible, dtype=torch.int32)
    if compressors:
        from utils import int8_quant_per_row

        index_q = (inputs["qr"].int() @ tensors["idx_wq_b"].int()).float()
        index_q *= inputs["qr_scale"] * tensors["idx_wq_b_scale"][None, :]
        index_q = index_q.view(tokens, kernel.IDX_N_HEADS, kernel.IDX_HEAD_DIM)
        tail = index_q[..., -kernel.ROPE_HEAD_DIM :]
        signed_sin = tensors["freqs_sin"][positions].clone()
        signed_sin[:, ::2] *= -1
        index_q[..., -kernel.ROPE_HEAD_DIM :] = (
            tail * tensors["freqs_cos"][positions, None, :]
            + tail.reshape(tokens, kernel.IDX_N_HEADS, -1, 2).flip(-1).flatten(-2) * signed_sin[:, None, :]
        )
        index_q = index_q.bfloat16().float() @ tensors["hadamard_idx"].float()
        query, query_scale = int8_quant_per_row(index_q)
        weights = (tensors["x_normed"].float() @ tensors["weights_proj"].float()) * kernel.M.index_weights_scale
        shared = tensors["inner_index_pages"]
        scales = shared[:, 128:130].reshape(shared.shape[0], -1).view(torch.float16).float()
        for token, position in enumerate(positions.tolist()):
            if not int(tensors["token_valid"][token]):
                continue
            request = token // layout["seq"]
            count = min((position + 1) // 4, int(tensors["kv_seq_lens"][request]) // 4)
            logical = torch.arange(count)
            pages = tensors["index_block_table"][request, logical // 128].long()
            keys = shared[pages, logical % 128].int()
            scores = (query[token].int() @ keys.T).float() * query_scale[token]
            scores = (scores.relu() * weights[token, :, None]).sum(0) * scales[pages, logical % 128]
            topk[token].fill_(-1)
            topk[token, : min(kernel.IDX_TOPK, count)] = scores.topk(min(kernel.IDX_TOPK, count)).indices.int()
    cmp_table = (tensors["cmp_block_table"][:, :, None] * 4 + torch.arange(4)).flatten(1).int()
    packed = torch.zeros(kernel.O_GROUPS, kernel.T_PAD, kernel.O_GROUP_IN, dtype=torch.bfloat16)
    golden_sparse_attn(
        dict(
            q=q,
            ori_kv=tensors["kv_cache_pages"].reshape(-1, 32, 1, kernel.HEAD_DIM),
            window_swa_indices=windows,
            cmp_kv=tensors["cmp_kv_pages"].reshape(-1, 32, 1, kernel.HEAD_DIM),
            cmp_block_table=cmp_table,
            idx_topk=topk,
            position_ids=positions.int().view(-1, 1),
            attn_sink=tensors["attn_sink"],
            freqs_cos=cos,
            freqs_sin=sin,
            o_packed_heads=packed,
        )
    )
    tensors["attn_out"][:] = golden_decode_o_proj_tp1(
        packed,
        tensors["wo_a"],
        tensors["wo_b"],
        tensors["wo_b_scale"],
        tokens,
    )
    assert torch.count_nonzero(tensors["attn_out"]) > 0, "nonzero fixture must exercise the output"


def inner_page_compare(kernel, layout):
    """Compare mixed-dtype pages numerically and all untouched bytes exactly."""

    def compare(actual, expected, **kwargs):
        a, e = actual.clone(), expected.clone()
        state_shape = (a.shape[0], -1)
        state_a, state_e = a.view(torch.float32).reshape(state_shape), e.view(torch.float32).reshape(state_shape)
        scale_a = a[:, 128:130].reshape(a.shape[0], -1).view(torch.float16)
        scale_e = e[:, 128:130].reshape(e.shape[0], -1).view(torch.float16)
        for request, positions in enumerate(layout["positions"].view(-1, layout["seq"]).tolist()):
            for position in positions:
                page = int(layout["inner"][request, position // 8])
                state_dim = 2 * kernel.INNER_OUT_DIM
                begin = position % 8 * state_dim
                torch.testing.assert_close(
                    state_a[page, begin : begin + state_dim],
                    state_e[page, begin : begin + state_dim],
                    rtol=0.02,
                    atol=0.002,
                )
                state_a[page, begin : begin + state_dim].zero_()
                state_e[page, begin : begin + state_dim].zero_()
                if (position + 1) % 4 == 0:
                    row = (position + 1) // 4 - 1
                    page = int(layout["index"][request, row // 128])
                    delta = (a[page, row % 128].int() - e[page, row % 128].int()).abs().max()
                    assert delta <= 1, f"int8 key max delta {delta}"
                    torch.testing.assert_close(scale_a[page, row % 128], scale_e[page, row % 128], rtol=0.02, atol=1e-7)
                    a[page, row % 128].zero_()
                    e[page, row % 128].zero_()
                    scale_a[page, row % 128] = scale_e[page, row % 128] = 0
        assert torch.equal(a, e), "untouched bytes changed"
        return True, "mixed state/key/scale pages and untouched bytes PASS"

    return compare


def shared_main_gate(kernel, specs, layout, work_dir, config):
    """Dispatch the unmodified entry with two views of one resident allocation.

    Uses the zero/sentinel fixture to check the entire shared allocation byte
    for byte; no numerical comparator can hide writes through the wrong view.
    """
    from pypto.ir import CompiledProgram
    from pypto.runtime import ChipWorker, DeviceTensor, RunConfig

    inputs = {spec.name: spec.create_tensor() for spec in specs}
    pool = inputs["compress_state_pages"]
    cmp_view = pool.view(torch.bfloat16).reshape(inputs["cmp_kv_pages"].shape)
    used_pages = torch.unique(layout["compressed"])
    used_pages = used_pages[used_pages > 0].long()
    cmp_view[used_pages] = inputs["cmp_kv_pages"][used_pages]
    inputs["cmp_kv_pages"] = cmp_view
    expected = {name: value.clone() for name, value in inputs.items()}
    expected["cmp_kv_pages"] = expected["compress_state_pages"].view(torch.bfloat16).reshape(cmp_view.shape)
    golden(kernel, layout, expected)
    compiled = CompiledProgram.from_dir(work_dir, platform=config["platform"])
    cfg = RunConfig(**config)
    with ChipWorker(config=cfg) as worker:
        resident = worker.alloc_tensor(pool.shape, pool.dtype, init=pool)
        try:
            cmp_resident = DeviceTensor(resident.data_ptr, cmp_view.shape, cmp_view.dtype, buffer=resident.buffer)
            args = [
                resident
                if s.name == "compress_state_pages"
                else cmp_resident
                if s.name == "cmp_kv_pages"
                else inputs[s.name]
                for s in specs
            ]
            compiled(*args, config=cfg)
            worker.copy_from(pool.data_ptr(), resident.data_ptr, resident.nbytes)
        finally:
            worker.free_tensor(resident)
    for name in ("compress_state_pages", "kv_cache_pages", "inner_index_pages", "attn_out"):
        assert torch.equal(inputs[name].view(torch.uint8), expected[name].view(torch.uint8)), name
        print(f"[SHARED] {name} byte-exact PASS", flush=True)
    print(f"[SHARED] main/cmp alias PASS work_dir={work_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True, choices=TARGET_BATCHES)
    parser.add_argument("--seq", type=int, default=6, choices=(1, 6))
    parser.add_argument("--start-pos", type=int, default=START_POS)
    parser.add_argument("--nonzero", action="store_true")
    parser.add_argument("--nonzero-compressors", action="store_true")
    parser.add_argument("--shared-main", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--chip-swimlane",
        type=int,
        default=0,
        choices=range(5),
        help="Capture the PyPTO kernel swimlane after functional gates pass.",
    )
    parser.add_argument(
        "--valid-tokens",
        type=int,
        default=None,
        choices=range(7),
    )
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--device", type=int, required=True, help="Use --device {} under task-submit --device auto.")
    args = parser.parse_args()

    _add_model_helpers()
    sys.argv = [sys.argv[0], "--tp", "1"]
    from golden import run

    from vllm_ascend.attention.pto_kernels.dspark import decode_csa as kernel

    valid_tokens = args.valid_tokens if args.valid_tokens is not None else args.seq
    specs, layout = build_specs(kernel, args.batch, valid_tokens, args.seq, args.start_pos)
    nonzero = args.nonzero or args.nonzero_compressors
    if args.shared_main and (nonzero or args.chip_swimlane):
        parser.error("--shared-main uses a byte-exact sentinel fixture without tracing")
    if nonzero:
        specs = nonzero_specs(kernel, specs, args.nonzero_compressors)
    reference = (
        (lambda model, allocation, tensors: golden_nonzero(model, allocation, tensors, args.nonzero_compressors))
        if nonzero
        else golden
    )
    config = {
        "platform": args.platform,
        "device_id": args.device,
        "enable_chip_swimlane": args.chip_swimlane,
        "enable_dep_gen": bool(args.chip_swimlane),
        "enable_scope_stats": True,
        "enable_pmu": 0,
        "dump_passes": False,
        "ring_heap": 512 * 1024**2,
    }
    result = run(
        fn=kernel.decode_csa_attn_tp1_test,
        specs=specs,
        golden_fn=lambda tensors: reference(kernel, layout, tensors),
        compile_only=args.compile_only or args.shared_main,
        config=config,
        rtol=0.02 if nonzero else 0.0,
        atol=0.002 if nonzero else 0.0,
        compare_fn={"inner_index_pages": inner_page_compare(kernel, layout)} if args.nonzero_compressors else None,
    )
    if args.shared_main and not args.compile_only:
        shared_main_gate(kernel, specs, layout, result.work_dir, config)
    print(
        f"batch={args.batch} tokens={args.batch * args.seq} "
        f"valid_tokens={valid_tokens} "
        f"passed={result.passed} work_dir={result.work_dir} error={result.error}",
        flush=True,
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
