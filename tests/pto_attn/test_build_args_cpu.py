"""Dry-run build_args on CPU with vLLM-shaped stand-ins.

build_args is the piece with the most ways to be quietly wrong -- a misspelled
attribute, a transposed weight, a table whose width does not match the kernel's
compile-time column count. None of that needs a device to surface.
"""
import os
import sys
import types

import torch

sys.argv = [sys.argv[0], "--tp", "1"]

import vllm_ascend.attention.pto_attn as pa  # noqa: E402

# torch_npu's format cast is a device op; on CPU the identity is the right stand-in.
pa._to_nd = lambda w: w

from vllm_ascend.attention.pto_kernels.dspark import config as C  # noqa: E402
from vllm_ascend.attention.pto_kernels.dspark import decode_csa as K  # noqa: E402

M = C.FLASH
D = M.hidden_size
TARGET_BATCHES = (4, 8, 16, 24, 32, 40)
NREQ = int(os.environ.get("TEST_BATCH", "4"))
HOST_SEQ = int(os.environ.get("TEST_SEQ", "1"))
ROPE_ROWS = 1024
rope_cos = torch.randn(ROPE_ROWS, 64)
rope_sin = torch.randn(ROPE_ROWS, 64)
pa._native_rope_tables = lambda layer: (rope_cos, rope_sin)
if NREQ not in TARGET_BATCHES:
    raise ValueError(f"TEST_BATCH must be one of {TARGET_BATCHES}, got {NREQ}")
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


class Lin:
    def __init__(self, *shape, dtype=torch.bfloat16, scale=None):
        self.weight = torch.randn(*shape).to(dtype) if dtype != torch.int8 \
            else torch.randint(-127, 127, shape, dtype=torch.int8)
        if scale is not None:
            self.weight_scale_fp32 = torch.rand(scale) + 0.5


class Norm:
    def __init__(self, n):
        self.weight = torch.randn(n).to(torch.bfloat16)


class Cmp:
    def __init__(self, out_dim, head_dim):
        self.wkv, self.wgate = Lin(out_dim, D), Lin(out_dim, D)
        self.ape = torch.randn(C.COMPRESS_RATIO if hasattr(C, "COMPRESS_RATIO") else 4, out_dim)
        self.norm = Norm(head_dim)


class Indexer:
    def __init__(self, quantized=True):
        self.head_dim = K.IDX_HEAD_DIM
        self.wq_b = (Lin(M.q_lora_rank, K.IDX_N_HEADS * K.IDX_HEAD_DIM,
                         dtype=torch.int8, scale=K.IDX_N_HEADS * K.IDX_HEAD_DIM)
                     if quantized else Lin(K.IDX_N_HEADS * K.IDX_HEAD_DIM, M.q_lora_rank))
        self.weights_proj = Lin(K.IDX_N_HEADS, D)
        self.compressor = Cmp(K.INNER_OUT_DIM, K.IDX_HEAD_DIM)


class Impl:
    def __init__(self, quantized: bool):
        self.layer_name = "model.layers.2.self_attn.attn"
        if quantized:
            self.wq_a = Lin(D, M.q_lora_rank, dtype=torch.int8, scale=M.q_lora_rank)
            self.wq_b = Lin(M.q_lora_rank, K.H * K.HEAD_DIM, dtype=torch.int8,
                            scale=K.H * K.HEAD_DIM)
            self.wkv = Lin(D, K.HEAD_DIM, dtype=torch.int8, scale=K.HEAD_DIM)
        else:
            # An unquantized checkpoint keeps torch's [out, in].
            self.wq_a = Lin(M.q_lora_rank, D)
            self.wq_b = Lin(K.H * K.HEAD_DIM, M.q_lora_rank)
            self.wkv = Lin(K.HEAD_DIM, D)
        self.q_norm, self.kv_norm = Norm(M.q_lora_rank), Norm(K.HEAD_DIM)
        self.compressor_wkv = Lin(K.MAIN_OUT_DIM, D)
        self.compressor_wgate = Lin(K.MAIN_OUT_DIM, D)
        self.compressor_ape = torch.randn(4, K.MAIN_OUT_DIM)
        self.compressor_norm = Norm(K.HEAD_DIM)
        self.indexcom_wkv = Lin(K.INNER_OUT_DIM, D)
        self.indexcom_wgate = Lin(K.INNER_OUT_DIM, D)
        self.indexcom_ape = torch.randn(4, K.INNER_OUT_DIM)
        self.indexcom_norm = Norm(K.IDX_HEAD_DIM)
        self.indexer = Indexer(quantized)
        self.inderxer_wq_b = self.indexer.wq_b
        self.weights_proj = self.indexer.weights_proj
        self.attn_sink = torch.rand(K.H)
        self.wo_a = Lin(K.O_GROUPS, K.O_GROUP_IN, K.O_LORA)
        self.wo_b = Lin(D, K.O_GROUPS * K.O_LORA)


def md(page, ncols, nrows, cos_key):
    starts = torch.arange(NREQ, dtype=torch.int64) * 7 + 300
    pos = (starts[:, None] + torch.arange(HOST_SEQ)).reshape(-1)
    o = types.SimpleNamespace()
    o.input_positions = pos
    o.block_table = torch.arange(NREQ * ncols, dtype=torch.int32).view(NREQ, ncols) + 1
    o.seq_lens = (starts + HOST_SEQ).to(torch.int32)
    sm = torch.stack([torch.arange(NREQ) % nrows, torch.arange(NREQ) % page], 1).to(torch.int32)
    o.slot_mapping = sm
    t = torch.randn(NREQ, 1, 1, 64)
    o.cos = {cos_key: t}
    o.sin = {cos_key: t.clone()}
    o.compress_cos = {cos_key: torch.randn(max(NREQ // 4, 1), 1, 1, 64)}
    o.compress_sin = {cos_key: torch.randn(max(NREQ // 4, 1), 1, 1, 64)}
    return types.SimpleNamespace(decode=o)


def strided(nblk, rows, dim, pad):
    raw = torch.zeros(nblk * pad)
    return torch.as_strided(raw, (nblk, rows, dim), (pad, dim, 1))


QUANT = os.environ.get("TEST_QUANT", "0") == "1"
print(f"== checkpoint: {'W8A8' if QUANT else 'BF16'} ==")
impl = Impl(QUANT)
L = impl.layer_name
metas = [md(128, 64, 4096, L) for _ in range(5)]
def strided_t(nblk, rows, dim, pad, dtype):
    raw = torch.zeros(nblk * pad, dtype=dtype)
    return torch.as_strided(raw, (nblk, rows, 1, dim), (pad, dim, dim, 1))


# Shapes, strides and dtypes as the live probe/code contract records them.  Main
# state and compressed KV share a 131072-byte page pool; raw KV is separate.
# Inner state/index key/index scale share one 16640-byte allocation.
main_state_dim = 2 * K.MAIN_OUT_DIM
inner_state_dim = 2 * K.INNER_OUT_DIM
main_state_parent = torch.zeros(64, 16, main_state_dim, dtype=torch.float32)
main_state = torch.as_strided(
    main_state_parent,
    (64, 8, 1, main_state_dim),
    (32768, main_state_dim, main_state_dim, 1),
    0,
)
raw_parent = torch.zeros(64, 128, 1, K.HEAD_DIM, dtype=torch.bfloat16)
cmp_parent = main_state_parent.view(torch.bfloat16).view(64, 128, 1, K.HEAD_DIM)
index_parent = torch.zeros(64 * 16640, dtype=torch.int8)
inner_state = torch.as_strided(
    index_parent.view(torch.float32),
    (64, 8, 1, inner_state_dim),
    (4160, inner_state_dim, inner_state_dim, 1),
    0,
)
index_key = torch.as_strided(
    index_parent, (64, 128, 1, K.IDX_HEAD_DIM),
    (16640, K.IDX_HEAD_DIM, K.IDX_HEAD_DIM, 1), 0,
)
index_scale = torch.as_strided(
    index_parent.view(torch.float16), (64, 128, 1, 1),
    (8320, 1, 1, 1), 8192,
)
kvc = (
    cmp_parent,
    raw_parent,
    main_state,
    inner_state,
    index_key,
    index_scale,
)
hs = torch.randn(NREQ * HOST_SEQ, D).to(torch.bfloat16)
output = torch.empty_like(hs)

print("== build_args ==")
check("TP1 specialization", K.TP_SIZE == 1, str(K.TP_SIZE))
check("S=6 specialization", K.S == 6, str(K.S))
check("B=64 capacity", K.B == 64, str(K.B))
check("T=384 capacity", K.T == 384 and K.T_PAD == 384, f"{K.T}/{K.T_PAD}")
check(
    "target runtime batches fit",
    all(
        batch <= K.B
        and batch * K.S <= K.T
        and (batch * K.S) % 4 == 0
        for batch in TARGET_BATCHES
    ),
    str([(batch, batch * K.S) for batch in TARGET_BATCHES]),
)
try:
    args, plan = pa.build_args(impl, hs, kvc, metas, HOST_SEQ, L, output=output)
    check("returned 40", len(args) == 40, str(len(args)))
except Exception:
    import traceback
    traceback.print_exc()
    check("build_args ran", False)
    print("\nFAILED")
    sys.exit(1)

print("== shapes vs the kernel signature ==")
# Runtime rows follow the payload, not the kernel's maximum S capacity.
RT = NREQ * HOST_SEQ
want = {
    "x_normed": [RT, D], "attn_out": [RT, D],
    "freqs_cos": [ROPE_ROWS, 64], "cmp_freqs_cos": [ROPE_ROWS, 64],
    "position_ids": [RT], "token_valid": [RT],
    "compress_state_pages": [64, 16, main_state_dim],
    "kv_cache_pages": [64, 128, 1, K.HEAD_DIM],
    "cmp_kv_pages": [64, 128, 1, K.HEAD_DIM],
    "inner_index_pages": [64, 130, K.IDX_HEAD_DIM],
    "compress_state_block_table": [NREQ, 64],
    "inner_compress_state_block_table": [NREQ, 64],
    "ori_block_table": [NREQ, 64],
    "cmp_block_table": [NREQ, 64],
    "index_block_table": [NREQ, 64],
    "wo_a": [K.O_GROUPS, K.O_LORA, K.O_GROUP_IN],
    "weights_proj": [D, K.IDX_N_HEADS],
    "hadamard_idx": [K.IDX_HEAD_DIM, K.IDX_HEAD_DIM],
    # The three INT8 slots do not share a scale axis.
    "wq_b": [M.q_lora_rank, K.H * K.HEAD_DIM], "wq_b_scale": [K.H * K.HEAD_DIM],
    "idx_wq_b": [M.q_lora_rank, K.IDX_N_HEADS * K.IDX_HEAD_DIM],
    "idx_wq_b_scale": [K.IDX_N_HEADS * K.IDX_HEAD_DIM],
    "wo_b": [D, K.O_GROUPS * K.O_LORA], "wo_b_scale": [D],
    "wq_a": [D, M.q_lora_rank], "wkv": [D, K.HEAD_DIM],
    "cmp_ape": [4, K.MAIN_OUT_DIM], "inner_ape": [4, K.INNER_OUT_DIM],
    "attn_sink": [K.H],
}
by = dict(zip(pa.ARG_ORDER, args))
for n, w in want.items():
    got = list(by[n].shape)
    check(n, got == w, f"{got} want {w}")

print("== dtypes the signature fixes ==")
check(
    "inner_index_pages is INT8",
    by["inner_index_pages"].dtype == torch.int8,
    str(by["inner_index_pages"].dtype),
)
check(
    "shared page aliases allocator parent",
    by["inner_index_pages"].data_ptr() == inner_state.data_ptr(),
)
check("main state aliases its parent",
      by["compress_state_pages"].data_ptr() == main_state.data_ptr())
check("raw KV aliases its parent",
      by["kv_cache_pages"].data_ptr() == raw_parent.data_ptr())
check("compressed KV aliases its parent",
      by["cmp_kv_pages"].data_ptr() == cmp_parent.data_ptr())
check("main state and compressed KV share pages",
      by["compress_state_pages"].data_ptr() == by["cmp_kv_pages"].data_ptr())
check("input rows alias vLLM", by["x_normed"].data_ptr() == hs.data_ptr())
check("output rows alias vLLM", by["attn_out"].data_ptr() == output.data_ptr())
check("positions alias vLLM INT64", by["position_ids"].data_ptr() == metas[0].decode.input_positions.data_ptr()
      and by["position_ids"].dtype == torch.int64)
check("RoPE aliases persistent FP32 table", by["freqs_cos"].data_ptr() == rope_cos.data_ptr()
      and by["freqs_cos"].dtype == torch.float32)
check("compressed RoPE reuses persistent table", by["cmp_freqs_cos"].data_ptr() == rope_cos.data_ptr())

print("== contiguity (the binding rejects anything else) ==")
bad = [n for n, a in by.items() if not a.is_contiguous()]
check("all contiguous", not bad, str(bad))

print("== only actual token rows ==")
valid = by["token_valid"].view(NREQ, HOST_SEQ)
check("all real lanes valid", bool((valid == 1).all()))
check("no rectangular repetition", by["x_normed"].shape[0] == hs.shape[0])

print("== graph-padded request is inert ==")
# vLLM pads a size-3 replay to the size-4 descriptor by filling the last raw
# block-table row with null block 0.  That request must not write any shared page.
metas[-1].decode.block_table[-1].zero_()
padded_args, _ = pa.build_args(impl, hs, kvc, metas, HOST_SEQ, L)
padded = dict(zip(pa.ARG_ORDER, padded_args))["token_valid"].view(NREQ, HOST_SEQ)
check("live requests stay active", bool((padded[:-1, 0] == 1).all()))
check("padded request is inactive", bool((padded[-1] == 0).all()))

metas[0].decode.seq_lens[0] -= 1
tail_args, _ = pa.build_args(impl, hs, kvc, metas, HOST_SEQ, L)
tail_valid = dict(zip(pa.ARG_ORDER, tail_args))["token_valid"].view(NREQ, HOST_SEQ)
check("invalid tail is masked", tail_valid[0, -1].item() == 0)
check("earlier valid lanes preserved", bool((tail_valid[0, :-1] == 1).all()))

print()
print("FAILED:" if fails else "ALL PASS", fails or "")
sys.exit(1 if fails else 0)
