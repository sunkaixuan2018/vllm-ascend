# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=1
# ruff: noqa: E402
"""DeepSeek-V4 CSA attention over vLLM-native paged cache layouts."""


import sys

from . import config


# TP specialization before sub-kernel imports.
_TP_CHOICES = (1, 2, 4)
_TP_DEFAULT = 1


def _parse_tp_argv():
    for index, arg in enumerate(sys.argv):
        if arg == "--tp" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith("--tp="):
            return int(arg.split("=", 1)[1])
    return _TP_DEFAULT


TP_SIZE = _parse_tp_argv()
if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})")
config.TP = TP_SIZE

import pypto.language as pl

from .config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
)
from .decode_compressor_ratio4 import compressor_ratio4_vllm
from .decode_indexer import indexer_vllm
from .decode_indexer_compressor import indexer_compressor_vllm
from .qkv_proj_rope import kv_proj_rope, q_proj_rope
from .decode_o_proj import (
    LOCAL_T,
    LOCAL_T_PAD,
    decode_o_proj_tp1,
)
from .decode_sparse_attn_csa import (
    T_PAD,
    sparse_attn_csa_tp1_vllm,
)

# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")  # per-request axis
T_DYN = pl.dynamic("T_DYN")  # T = B * S
ROPE_ROWS_DYN = pl.dynamic("VLLM_ROPE_ROWS_DYN")
CMP_ROPE_ROWS_DYN = pl.dynamic("VLLM_CMP_ROPE_ROWS_DYN")
VLLM_COMPRESS_STATE_PAGE_NUM_DYN = pl.dynamic(
    "VLLM_COMPRESS_STATE_PAGE_NUM_DYN"
)
VLLM_KV_CACHE_PAGE_NUM_DYN = pl.dynamic("VLLM_KV_CACHE_PAGE_NUM_DYN")
VLLM_CMP_KV_PAGE_NUM_DYN = pl.dynamic("VLLM_CMP_KV_PAGE_NUM_DYN")
VLLM_MAIN_STATE_TABLE_WIDTH_DYN = pl.dynamic(
    "VLLM_MAIN_STATE_TABLE_WIDTH_DYN"
)
VLLM_INNER_STATE_TABLE_WIDTH_DYN = pl.dynamic(
    "VLLM_INNER_STATE_TABLE_WIDTH_DYN"
)
VLLM_ORI_TABLE_WIDTH_DYN = pl.dynamic("VLLM_ORI_TABLE_WIDTH_DYN")
VLLM_CMP_TABLE_WIDTH_DYN = pl.dynamic("VLLM_CMP_TABLE_WIDTH_DYN")
VLLM_INNER_INDEX_PAGE_NUM_DYN = pl.dynamic(
    "VLLM_INNER_INDEX_PAGE_NUM_DYN"
)
VLLM_INDEX_TABLE_WIDTH_DYN = pl.dynamic("VLLM_INDEX_TABLE_WIDTH_DYN")

# model config
B = DECODE_BATCH // TP_SIZE
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_TOPK = M.index_topk
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel constants
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_DIM = 2 * MAIN_OUT_DIM
INNER_OUT_DIM = COFF * IDX_HEAD_DIM

# tiling
TP1_CSA_WB_WORKERS = 8  # TP1 CSA cache-write workers
VLLM_KV_PAGE_ROWS = 128
VLLM_COMPRESS_STATE_PAGE_ROWS = 16
VLLM_COMPRESS_STATE_LIVE_ROWS = 8
VLLM_INDEX_PAGE_ROWS = 130

if T != LOCAL_T:
    raise ValueError(f"CSA token capacity {T} must equal TP local token capacity {LOCAL_T}")
if T_PAD != LOCAL_T_PAD:
    raise ValueError(f"CSA token capacity {T_PAD} must equal TP local token capacity {LOCAL_T_PAD}")


def _decode_csa_attn_tp1(
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    freqs_sin: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_freqs_cos: pl.Tensor[[CMP_ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_freqs_sin: pl.Tensor[[CMP_ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state_pages: pl.InOut[
        pl.Tensor[
            [
                VLLM_COMPRESS_STATE_PAGE_NUM_DYN,
                VLLM_COMPRESS_STATE_PAGE_ROWS,
                MAIN_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    kv_cache_pages: pl.InOut[
        pl.Tensor[
            [VLLM_KV_CACHE_PAGE_NUM_DYN, VLLM_KV_PAGE_ROWS, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    cmp_kv_pages: pl.InOut[
        pl.Tensor[
            [VLLM_CMP_KV_PAGE_NUM_DYN, VLLM_KV_PAGE_ROWS, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    compress_state_block_table: pl.Tensor[
        [B_DYN, VLLM_MAIN_STATE_TABLE_WIDTH_DYN], pl.INT32
    ],
    idx_wq_b: pl.Tensor[
        [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8
    ],
    idx_wq_b_scale: pl.Tensor[
        [IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32
    ],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_index_pages: pl.InOut[
        pl.Tensor[
            [
                VLLM_INNER_INDEX_PAGE_NUM_DYN,
                VLLM_INDEX_PAGE_ROWS,
                IDX_HEAD_DIM,
            ],
            pl.INT8,
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[
        [B_DYN, VLLM_INNER_STATE_TABLE_WIDTH_DYN], pl.INT32
    ],
    ori_block_table: pl.Tensor[
        [B_DYN, VLLM_ORI_TABLE_WIDTH_DYN], pl.INT32
    ],
    cmp_block_table: pl.Tensor[
        [B_DYN, VLLM_CMP_TABLE_WIDTH_DYN], pl.INT32
    ],
    index_block_table: pl.Tensor[
        [B_DYN, VLLM_INDEX_TABLE_WIDTH_DYN], pl.INT32
    ],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    token_valid: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Consume real token rows and native interleaved FP32 RoPE tables.

    RoPE inputs are persistent tables indexed by absolute position, not
    token-local or compact boundary rows. Their extents are independent of T.
    """
    x_normed.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(0, ROPE_ROWS_DYN)
    cmp_freqs_cos.bind_dynamic(0, CMP_ROPE_ROWS_DYN)
    cmp_freqs_sin.bind_dynamic(0, CMP_ROPE_ROWS_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    token_valid.bind_dynamic(0, T_DYN)
    attn_out.bind_dynamic(0, T_DYN)
    kv_seq_lens.bind_dynamic(0, B_DYN)
    compress_state_pages.bind_dynamic(0, VLLM_COMPRESS_STATE_PAGE_NUM_DYN)
    kv_cache_pages.bind_dynamic(0, VLLM_KV_CACHE_PAGE_NUM_DYN)
    cmp_kv_pages.bind_dynamic(0, VLLM_CMP_KV_PAGE_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, B_DYN)
    compress_state_block_table.bind_dynamic(
        1, VLLM_MAIN_STATE_TABLE_WIDTH_DYN,
    )
    inner_index_pages.bind_dynamic(0, VLLM_INNER_INDEX_PAGE_NUM_DYN)
    inner_compress_state_block_table.bind_dynamic(0, B_DYN)
    inner_compress_state_block_table.bind_dynamic(
        1, VLLM_INNER_STATE_TABLE_WIDTH_DYN,
    )
    ori_block_table.bind_dynamic(0, B_DYN)
    ori_block_table.bind_dynamic(1, VLLM_ORI_TABLE_WIDTH_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    cmp_block_table.bind_dynamic(1, VLLM_CMP_TABLE_WIDTH_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, VLLM_INDEX_TABLE_WIDTH_DYN)

    t_dim = pl.tensor.dim(x_normed, 0)
    b_dim = pl.tensor.dim(kv_seq_lens, 0)
    s_dim = t_dim // b_dim
    positions_i32 = pl.create_tensor([t_dim], dtype=pl.INT32)
    rope_swap_idx = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.INT32)
    idx_cos_il = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    idx_sin_signed = pl.create_tensor(
        [t_dim, ROPE_HEAD_DIM], dtype=pl.FP32,
    )
    cmp_cos_il = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_sin_signed = pl.create_tensor(
        [t_dim, ROPE_HEAD_DIM], dtype=pl.FP32,
    )
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="csa_vllm_rope_interleave",
    ) as rope_tid:
        ones = pl.full([1, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        columns = pl.col_expand_mul(
            ones,
            pl.cast(
                pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32),
                target_type=pl.FP32,
            ),
        )
        duplicate = pl.cast(
            pl.cast(
                pl.mul(columns, 0.5), target_type=pl.INT32, mode="trunc",
            ),
            target_type=pl.FP32,
        )
        lane = pl.sub(columns, pl.mul(duplicate, 2.0))
        sign = pl.sub(pl.mul(lane, 2.0), 1.0)
        swap = pl.cast(
            pl.sub(pl.add(columns, 1.0), pl.mul(lane, 2.0)), pl.INT32,
        )
        for token in pl.range(t_dim):
            position = pl.cast(pl.read(position_ids, [token]), pl.INDEX)
            active_position = -1
            if pl.read(token_valid, [token]) != 0:
                active_position = position
            pl.write(positions_i32, [token], pl.cast(active_position, pl.INT32))
            # Inactive rows only read table row zero and never publish state.
            rope_row = pl.max(active_position, 0)
            cmp_row = pl.max(active_position + 1 - COMPRESS_RATIO, 0)
            idx_cos_il[token : token + 1, :] = freqs_cos[rope_row : rope_row + 1, :]
            idx_sin_signed[token : token + 1, :] = pl.mul(
                freqs_sin[rope_row : rope_row + 1, :], sign,
            )
            cmp_cos_il[token : token + 1, :] = cmp_freqs_cos[cmp_row : cmp_row + 1, :]
            cmp_sin_signed[token : token + 1, :] = pl.mul(
                cmp_freqs_sin[cmp_row : cmp_row + 1, :], sign,
            )
            rope_swap_idx[token : token + 1, :] = swap

    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    topk_scores = pl.create_tensor([t_dim, IDX_TOPK], dtype=pl.FP32)
    topk_indices = pl.create_tensor([t_dim, IDX_TOPK], dtype=pl.INT32)
    position_ids_2d = pl.reshape(positions_i32, [t_dim, 1])
    late_dep = pl.system.task_dummy(deps=[rope_tid])
    q_proj_rope(
        x_normed,
        wq_a,
        wq_b,
        wq_b_scale,
        gamma_cq,
        idx_cos_il,
        idx_sin_signed,
        rope_swap_idx,
        q,
        qr,
        qr_scale,
        late_dep,
    )
    kv_proj_rope(
        x_normed, wkv, gamma_ckv, idx_cos_il, idx_sin_signed,
        rope_swap_idx, kv, late_dep,
    )

    with pl.spmd(
        TP1_CSA_WB_WORKERS, name_hint="csa_vllm_raw_cache_write",
    ) as raw_cache_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, t_dim, TP1_CSA_WB_WORKERS):
            if pl.read(token_valid, [token]) != 0:
                request = token // s_dim
                position = pl.cast(pl.read(positions_i32, [token]), pl.INDEX)
                logical_page = position // VLLM_KV_PAGE_ROWS
                physical_page_i32 = pl.read(
                    ori_block_table, [request, logical_page],
                )
                if physical_page_i32 > 0:
                    physical_page = pl.cast(physical_page_i32, pl.INDEX)
                    intra = position % VLLM_KV_PAGE_ROWS
                    kv_cache_pages[
                        physical_page : physical_page + 1,
                        intra : intra + 1,
                        0:1,
                        0:HEAD_DIM,
                    ] = pl.reshape(
                        kv[token : token + 1, 0:HEAD_DIM],
                        [1, 1, 1, HEAD_DIM],
                    )

    cmp_out = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
    cmp_out, cmp_cache_tid, cmp_projection_tid = compressor_ratio4_vllm(
        x_normed,
        cmp_out,
        compress_state_pages,
        cmp_kv_pages,
        compress_state_block_table,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        cmp_cos_il,
        cmp_sin_signed,
        cmp_block_table,
        positions_i32,
        token_valid,
        late_dep,
        raw_cache_tid,
    )
    idx_out = pl.create_tensor([t_dim, IDX_HEAD_DIM], dtype=pl.FP32)
    idx_cache_tid = indexer_compressor_vllm(
        x_normed,
        idx_out,
        inner_index_pages,
        inner_compress_state_block_table,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        cmp_cos_il,
        cmp_sin_signed,
        hadamard_idx,
        index_block_table,
        positions_i32,
        token_valid,
        late_dep,
        cmp_projection_tid,
    )
    topk_scores, topk_indices, topk_tid = indexer_vllm(
        x_normed,
        qr,
        qr_scale,
        idx_wq_b,
        idx_wq_b_scale,
        weights_proj,
        idx_cos_il,
        idx_sin_signed,
        hadamard_idx,
        inner_index_pages,
        index_block_table,
        topk_scores,
        topk_indices,
        positions_i32,
        kv_seq_lens,
        idx_cache_tid,
    )

    attention_ready = pl.system.task_dummy(
        deps=[raw_cache_tid, cmp_cache_tid, topk_tid],
    )
    o_packed_heads = pl.create_tensor(
        [O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.BF16,
    )
    o_packed_heads, heads_dep = sparse_attn_csa_tp1_vllm(
        q,
        kv_cache_pages,
        ori_block_table,
        cmp_kv_pages,
        cmp_block_table,
        topk_indices,
        position_ids_2d,
        token_valid,
        attn_sink,
        idx_cos_il,
        idx_sin_signed,
        o_packed_heads,
        attention_ready,
    )
    decode_o_proj_tp1(
        o_packed_heads,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
        heads_dep,
    )
    return attn_out


decode_csa_attn_tp1 = pl.jit.inline(_decode_csa_attn_tp1)
decode_csa_attn_tp1_test = pl.jit(_decode_csa_attn_tp1)
