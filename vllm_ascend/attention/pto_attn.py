"""Feed vLLM's decode tensors to the PyPTO attention-only CSA kernel.

The kernel is ``pto_kernels.dspark.decode_csa.decode_csa_attn_tp1_test``: it takes
``x_normed [T, D] BF16`` and fills ``attn_out [T, D] BF16``, leaving ``npu_hc_pre``,
the input RMSNorm and ``npu_hc_post`` to the native path. Everything here is the
binding between vLLM's decode state and the kernel's native-layout arguments.

Two properties the per-step path must keep, because it has to survive ACLGraph
capture: no device-to-host read (no ``.item()``, no boolean-mask indexing, no
``torch.unique``), and no allocation whose shape depends on a device value.

``PTO_ATTN_COMPARE=<dir>`` runs one decode step through both the native path and
this kernel and writes the comparison; it is a diagnostic, never a serving path.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

# --- kernel import -----------------------------------------------------------
# decode_csa fixes its TP specialization at import time from sys.argv, which a
# vLLM process never carries. Inject it so B = DECODE_BATCH // TP and T = B * S
# land on the intended shape instead of the module's own default of 1.
def _env_int(name: str, default: int) -> int:
    """The launcher forwards unset switches as empty strings, so a present-but-empty
    variable has to fall back the same way an absent one does."""
    return int(os.environ.get(name, "") or default)


_TP = _env_int("PTO_ATTN_TP", 1)


def _import_kernel():
    argv = sys.argv
    if not any(a == "--tp" or a.startswith("--tp=") for a in argv):
        sys.argv = [*argv, "--tp", str(_TP)]
    try:
        from .pto_kernels.dspark import config as kcfg
        from .pto_kernels.dspark import decode_csa as kcsa
    finally:
        sys.argv = argv
    return kcsa, kcfg


_KCSA = None
_KCFG = None


def kernel():
    global _KCSA, _KCFG
    if _KCSA is None:
        _KCSA, _KCFG = _import_kernel()
    return _KCSA, _KCFG


# --- constants ---------------------------------------------------------------

VLLM_PAGE = 128          # swa / compressed / indexer KV page, in slots
VLLM_STATE_PAGE = 8
COMPRESS_RATIO = 4


class NativeLayoutError(ValueError):
    """The live vLLM allocation does not satisfy the native CSA ABI."""


# --- weights: one-time, cached on the impl -----------------------------------


def _to_nd(w: torch.Tensor) -> torch.Tensor:
    """Undo the FRACTAL_NZ layout vLLM gives quantized weights.

    ``weight_nz_mode`` defaults to 1, and an NZ-laid-out weight read as ND is
    silently wrong rather than an error, so every INT8 weight goes through this.
    Dense weights are never converted, so they are returned untouched.
    """
    if w.dtype not in (torch.int8, torch.uint8):
        return w
    import torch_npu

    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND

    return torch_npu.npu_format_cast(w, ACL_FORMAT_FRACTAL_ND)


def _quant_int8_per_channel(w: torch.Tensor, kcfg):
    """BF16 -> INT8 plus per-channel scale, matching the kernel's contract."""
    amax = w.float().abs().amax(dim=-1).clamp_min(kcfg.INT8_AMAX_EPS)
    sq = kcfg.INT8_SCALE_MAX / amax
    q = torch.round(w.float() * sq.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q, (1.0 / sq).float()


def _oriented(w: torch.Tensor, want: tuple) -> torch.Tensor:
    """Give the kernel its orientation, whichever one vLLM stored.

    A W8A8 linear comes out of process_weights_after_loading already transposed to
    [in, out], while an unquantized one keeps torch's [out, in]. Deciding by shape
    rather than by quantization keeps both checkpoints working.
    """
    shape = tuple(w.shape)
    if shape == want:
        return w
    if shape == want[::-1]:
        return w.t().contiguous()
    raise ValueError(f"weight is {shape}, expected {want} or its transpose")


def _dense(linear, want: tuple) -> torch.Tensor:
    """A BF16 weight in the kernel's orientation, dequantizing if vLLM quantized it."""
    w = _to_nd(linear.weight.detach())
    scale = getattr(linear, "weight_scale_fp32", None)
    if scale is None:
        scale = getattr(linear, "weight_scale", None)
    if scale is not None and w.dtype in (torch.int8, torch.uint8):
        w = w.float() * scale.detach().float().view(1, -1)
    return _oriented(w.to(torch.bfloat16), want)


def _int8(linear, want: tuple, scale_len: int, kcfg):
    """An INT8 weight plus its per-channel scale, quantizing if vLLM kept it dense.

    The scale's axis is not the same for every slot: wq_b and idx_wq_b carry one
    scale per output column, wo_b one per output row. ``scale_len`` picks which,
    so a mismatch is a shape error here rather than at kernel launch.
    """
    w = _to_nd(linear.weight.detach())
    scale = getattr(linear, "weight_scale_fp32", None)
    if scale is None:
        scale = getattr(linear, "weight_scale", None)
    if w.dtype in (torch.int8, torch.uint8) and scale is not None:
        return _oriented(w, want), scale.detach().float().reshape(-1)

    dense = _oriented(w, want)
    if scale_len == want[1]:
        q, sc = _quant_int8_per_channel(dense.t().contiguous(), kcfg)
        return q.t().contiguous(), sc
    if scale_len == want[0]:
        return _quant_int8_per_channel(dense, kcfg)
    raise ValueError(f"scale length {scale_len} matches neither axis of {want}")


def _hadamard(dim: int, device, dtype=torch.bfloat16) -> torch.Tensor:
    """Sylvester Hadamard, normalized.

    The kernel treats this as a bare right-hand matmul operand with no scaling of
    its own, so the 1/sqrt(dim) that vLLM's rotate_activation applies separately
    has to be folded in here. Mirrors decode_csa.py::init_hadamard_idx.
    """
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < dim:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return (h / (dim**0.5)).to(dtype).to(device)


def prepare_weights(impl):
    """Build and cache the kernel's weight arguments on the impl."""
    cached = getattr(impl, "_pto_attn_weights", None)
    if cached is not None:
        return cached

    kcsa, kcfg = kernel()
    D, QL, HD = kcsa.D, kcsa.Q_LORA, kcsa.HEAD_DIM
    IH, ID = kcsa.IDX_N_HEADS, kcsa.IDX_HEAD_DIM
    wq_b, wq_b_scale = _int8(impl.wq_b, (QL, kcsa.H * HD), kcsa.H * HD, kcfg)
    idx_wq_b, idx_wq_b_scale = _int8(impl.inderxer_wq_b, (QL, IH * ID), IH * ID, kcfg)
    wo_b, wo_b_scale = _int8(impl.wo_b, (D, kcsa.O_GROUPS * kcsa.O_LORA), D, kcfg)

    w = {
        "wq_a": _dense(impl.wq_a, (D, QL)),
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale,
        "wkv": _dense(impl.wkv, (D, HD)),
        "gamma_cq": impl.q_norm.weight.detach().to(torch.bfloat16),
        "gamma_ckv": impl.kv_norm.weight.detach().to(torch.bfloat16),
        "cmp_wkv": _dense(impl.compressor_wkv, (kcsa.MAIN_OUT_DIM, D)),
        "cmp_wgate": _dense(impl.compressor_wgate, (kcsa.MAIN_OUT_DIM, D)),
        "cmp_ape": impl.compressor_ape.detach().float(),
        "cmp_norm_w": impl.compressor_norm.weight.detach().to(torch.bfloat16),
        "idx_wq_b": idx_wq_b,
        "idx_wq_b_scale": idx_wq_b_scale,
        "weights_proj": _dense(impl.weights_proj, (D, IH)),
        "hadamard_idx": _hadamard(ID, impl.wo_b.weight.device),
        "inner_wkv": _dense(impl.indexcom_wkv, (kcsa.INNER_OUT_DIM, D)),
        "inner_wgate": _dense(impl.indexcom_wgate, (kcsa.INNER_OUT_DIM, D)),
        "inner_ape": impl.indexcom_ape.detach().float(),
        "inner_norm_w": impl.indexcom_norm.weight.detach().to(torch.bfloat16),
        "attn_sink": impl.attn_sink.detach().float(),
        # vLLM keeps [G, O_GROUP_IN, O_LORA]; the kernel wants the transpose.
        "wo_a": impl.wo_a.weight.detach().transpose(1, 2).contiguous().to(torch.bfloat16),
        "wo_b": wo_b,
        "wo_b_scale": wo_b_scale,
    }
    impl._pto_attn_weights = w
    return w


# --- per-step derivation: everything below must stay device-only ------------


def _native_rope_tables(layer: str):
    """Alias the layer's persistent, interleaved FP32 vLLM RoPE tables.

    Do not use the per-step proxy: compressed rows there are compacted by
    boundary. The kernel can address the same static table by absolute position.
    """
    from vllm_ascend.ops.rope_dsv4 import _ROPE_STATE

    try:
        config_key, _ = _ROPE_STATE.layer_info[layer]
        cos, sin = _ROPE_STATE.static_cache[config_key]
    except KeyError as error:
        raise NativeLayoutError(f"no persistent RoPE table registered for {layer}") from error
    for name, table in (("cos", cos), ("sin", sin)):
        if table.dtype != torch.float32 or not table.is_contiguous() or table.shape[-1] != 64:
            raise NativeLayoutError(f"native RoPE {name} must be contiguous FP32 with 64 columns")
    if cos.shape != sin.shape:
        raise NativeLayoutError("native RoPE cos/sin shapes differ")
    return cos.view(-1, 64), sin.view(-1, 64)


_DEBUG_REFUSED = set()


def capture_active() -> bool:
    """Whether an ACLGraph capture is recording right now.

    vllm-ascend clears ``forward_context.capturing`` at the start of every forward
    and sets it immediately before entering the graph context, so it is true for
    the recorded pass and false for the warm-up that precedes it.
    """
    try:
        from vllm.forward_context import get_forward_context

        return bool(getattr(get_forward_context(), "capturing", False))
    except Exception:
        return False


# --- structure probe ---------------------------------------------------------


def describe(obj, depth: int = 0, limit: int = 3):
    """Shape/dtype sketch of a metadata object, for the first-call dump."""
    if isinstance(obj, torch.Tensor):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype),
                "contig": bool(obj.is_contiguous()), "stride": list(obj.stride())}
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [describe(x, depth + 1, limit) for x in obj[:8]]
    if isinstance(obj, dict):
        return {str(k): describe(v, depth + 1, limit) for k, v in list(obj.items())[:24]}
    if depth < limit:
        out = {"__class__": type(obj).__name__}
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                v = getattr(obj, name)
            except Exception:
                continue
            if callable(v):
                continue
            out[name] = describe(v, depth + 1, limit)
        return out
    return {"__class__": type(obj).__name__}


def dump_structure(path, hidden_states, kv_cache, attn_metadata, impl) -> None:
    payload = {
        "hidden_states": describe(hidden_states),
        "kv_cache": [describe(c, 2) for c in kv_cache],
        "attn_metadata": describe(attn_metadata, 0, 4),
        "impl_shapes": {
            k: describe(getattr(impl, k, None), 2)
            for k in ("wq_a", "wq_b", "wkv", "q_norm", "kv_norm", "wo_a", "wo_b",
                      "attn_sink", "weights_proj", "inderxer_wq_b",
                      "compressor_wkv", "compressor_wgate", "compressor_ape", "compressor_norm",
                      "indexcom_wkv", "indexcom_wgate", "indexcom_ape", "indexcom_norm")
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# --- native-layout arguments -------------------------------------------------

ARG_ORDER = (
    "x_normed",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "cmp_freqs_cos", "cmp_freqs_sin",
    "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
    "compress_state_pages", "kv_cache_pages", "cmp_kv_pages",
    "compress_state_block_table",
    "idx_wq_b", "idx_wq_b_scale", "weights_proj", "hadamard_idx",
    "inner_wkv", "inner_wgate", "inner_ape", "inner_norm_w",
    "inner_index_pages", "inner_compress_state_block_table",
    "ori_block_table", "cmp_block_table",
    "index_block_table",
    "position_ids", "token_valid", "kv_seq_lens", "attn_sink",
    "wo_a", "wo_b", "wo_b_scale",
    "attn_out",
)


def _full_page_view(cache: torch.Tensor, rows: int, row_shape: tuple[int, ...]):
    """Expose a padded physical page as a canonical contiguous tensor view."""
    row_elems = 1
    for extent in row_shape:
        row_elems *= extent
    page_elems = rows * row_elems
    if cache.stride(0) != page_elems:
        raise NativeLayoutError(
            f"physical page stride is {cache.stride(0)}, expected {page_elems}"
        )
    trailing = 1
    trailing_strides = []
    for extent in reversed(row_shape):
        trailing_strides.append(trailing)
        trailing *= extent
    shape = (cache.shape[0], rows, *row_shape)
    strides = (page_elems, row_elems, *reversed(trailing_strides))
    view = torch.as_strided(
        cache,
        size=shape,
        stride=strides,
        storage_offset=cache.storage_offset(),
    )
    if not view.is_contiguous():
        raise NativeLayoutError(
            f"native page view {shape} is not contiguous: {view.stride()}"
        )
    return view


def build_args(impl, hidden_states, kv_cache, metadata_list, seq: int, layer: str, output=None):
    """Bind one decode step to the kernel's native-layout arguments.

    ``metadata_list`` is what ``filter_metadata`` returns for a ratio-4 layer:
    five per-cache metadata objects sorted by key -- attn, compressor state,
    indexer-compressor state, indexer k, sliding window.
    """
    if not isinstance(layer, str):
        # RopeDataProxy takes a non-string key as a slice and hands back another
        # proxy, so a wrong name surfaces two frames later as a missing reshape.
        raise NativeLayoutError(
            f"layer must be the layer's name, got {type(layer).__name__}"
        )
    if len(metadata_list) != 5 or len(kv_cache) != 6:
        raise NativeLayoutError(
            f"ratio-4 CSA requires 5 metadata groups and 6 cache views, got "
            f"{len(metadata_list)} and {len(kv_cache)}"
        )
    if any(m.decode is None for m in metadata_list):
        raise NativeLayoutError("native CSA only accepts decode metadata")
    kcsa, _ = kernel()
    if not 1 <= seq <= kcsa.S:
        raise NativeLayoutError(f"native CSA requires 1..{kcsa.S} tokens per request, got seq={seq}")
    cmp_md, cst_md, ist_md, idx_md, swa_md = (m.decode for m in metadata_list)
    cmp_kv_c, swa_kv_c, state_c, ist_c, idx_k_c, idx_s_c = kv_cache

    host_pos = metadata_list[0].decode.input_positions
    if host_pos.dtype != torch.int64 or host_pos.ndim != 1 or not host_pos.is_contiguous():
        raise NativeLayoutError("input_positions must be contiguous INT64 token rows")
    if host_pos.shape[0] % seq:
        raise NativeLayoutError(
            f"position rows {host_pos.shape[0]} are not divisible by seq={seq}"
        )
    n_real = host_pos.shape[0] // seq            # graph descriptor request rows
    if not 1 <= n_real <= kcsa.B:
        raise NativeLayoutError(
            f"{n_real} requests exceed the kernel's B={kcsa.B}"
        )
    if hidden_states.shape[0] < host_pos.shape[0]:
        raise NativeLayoutError(
            f"hidden_states has {hidden_states.shape[0]} rows, expected at least "
            f"{host_pos.shape[0]}"
        )

    b, t = n_real, host_pos.shape[0]
    pos = host_pos
    host_positions = pos.view(b, seq)
    raw_logical_page = torch.div(
        host_positions, VLLM_PAGE, rounding_mode="floor",
    )
    raw_page_in_range = (raw_logical_page >= 0) & (
        raw_logical_page < swa_md.block_table.shape[1]
    )
    raw_logical_page = raw_logical_page.clamp(
        min=0, max=swa_md.block_table.shape[1] - 1,
    )
    raw_pages = swa_md.block_table[:b].gather(
        1, raw_logical_page,
    )
    rope_cos, rope_sin = _native_rope_tables(layer)
    if cmp_md.seq_lens.dtype != torch.int32 or not cmp_md.seq_lens.is_contiguous():
        raise NativeLayoutError("seq_lens must be contiguous INT32 request rows")
    seq_lens = cmp_md.seq_lens[:b]
    token_valid = (
        raw_page_in_range & (raw_pages > 0)
        & (host_positions < rope_cos.shape[0])
        & (host_positions < seq_lens.view(b, 1))
    ).reshape(t)

    def native_table(name: str, table: torch.Tensor) -> torch.Tensor:
        if table.dtype != torch.int32:
            raise NativeLayoutError(f"{name} must be INT32, got {table.dtype}")
        if table.shape[0] < b:
            raise NativeLayoutError(
                f"{name} has {table.shape[0]} requests, expected at least {b}"
            )
        if not table.is_contiguous():
            raise NativeLayoutError(f"{name} must be contiguous; no per-step copy is made")
        return table[:b]

    expected_dtypes = {
        "main state": (state_c, torch.float32),
        "inner state": (ist_c, torch.float32),
        "raw KV": (swa_kv_c, torch.bfloat16),
        "compressed KV": (cmp_kv_c, torch.bfloat16),
        "index key page": (idx_k_c, torch.int8),
        "index scale view": (idx_s_c, torch.float16),
    }
    for name, (tensor, dtype) in expected_dtypes.items():
        if tensor.dtype != dtype:
            raise NativeLayoutError(f"{name} must be {dtype}, got {tensor.dtype}")

    a = dict(prepare_weights(impl))
    a["x_normed"] = hidden_states[:t]
    a["attn_out"] = output[:t] if output is not None else torch.empty_like(a["x_normed"])
    for name in ("x_normed", "attn_out"):
        tensor = a[name]
        if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous() or tensor.shape != (t, kcsa.D):
            raise NativeLayoutError(f"{name} must be contiguous BF16 [{t}, {kcsa.D}]")

    a["freqs_cos"], a["freqs_sin"] = rope_cos, rope_sin
    a["cmp_freqs_cos"], a["cmp_freqs_sin"] = rope_cos, rope_sin

    # Main state and compressed KV are different views of one physical page
    # pool; raw sliding KV uses a separate allocation.
    if state_c.stride(0) != 32768:
        raise NativeLayoutError(
            f"main state page stride is {state_c.stride(0)}, expected 32768 FP32"
        )
    if swa_kv_c.stride(0) != 65536 or cmp_kv_c.stride(0) != 65536:
        raise NativeLayoutError(
            "raw/compressed KV page stride does not match 131072 bytes"
        )
    a["compress_state_pages"] = _full_page_view(
        state_c, kcsa.VLLM_COMPRESS_STATE_PAGE_ROWS, (kcsa.MAIN_STATE_DIM,),
    )
    a["kv_cache_pages"] = _full_page_view(
        swa_kv_c, kcsa.VLLM_KV_PAGE_ROWS, (1, kcsa.HEAD_DIM),
    )
    a["cmp_kv_pages"] = _full_page_view(
        cmp_kv_c, kcsa.VLLM_KV_PAGE_ROWS, (1, kcsa.HEAD_DIM),
    )
    if (a["compress_state_pages"].data_ptr() != a["cmp_kv_pages"].data_ptr()
            or a["compress_state_pages"].numel()
            * a["compress_state_pages"].element_size()
            != a["cmp_kv_pages"].numel()
            * a["cmp_kv_pages"].element_size()):
        raise NativeLayoutError(
            "main state and compressed KV must cover the same page pool"
        )
    a["compress_state_block_table"] = native_table(
        "main state block table", cst_md.block_table,
    )
    a["inner_compress_state_block_table"] = native_table(
        "inner state block table", ist_md.block_table,
    )
    a["ori_block_table"] = native_table(
        "raw KV block table", swa_md.block_table,
    )
    a["cmp_block_table"] = native_table(
        "compressed KV block table", cmp_md.block_table,
    )
    shared_storage = idx_k_c.untyped_storage().data_ptr()
    if ist_c.untyped_storage().data_ptr() != shared_storage:
        raise NativeLayoutError(
            "inner state and index key do not share one vLLM allocation"
        )
    if ist_c.data_ptr() != idx_k_c.data_ptr():
        raise NativeLayoutError(
            "inner state and index key do not start at the same physical page"
        )
    if ist_c.stride(0) != 4160:
        raise NativeLayoutError(
            f"inner state page stride is {ist_c.stride(0)}, expected 4160 FP32"
        )
    if idx_s_c.untyped_storage().data_ptr() != shared_storage:
        raise NativeLayoutError("index key and scale do not share one vLLM page")
    if idx_k_c.stride(0) != 16640 or idx_s_c.stride(0) != 8320:
        raise NativeLayoutError(
            "index key/scale physical strides do not match the 16640-byte page"
        )
    if idx_s_c.data_ptr() - idx_k_c.data_ptr() != 16384:
        raise NativeLayoutError(
            "index scale does not start at byte 16384 of the packed page"
        )
    a["inner_index_pages"] = _full_page_view(
        idx_k_c, kcsa.VLLM_INDEX_PAGE_ROWS, (kcsa.IDX_HEAD_DIM,),
    )
    a["index_block_table"] = native_table(
        "index block table", idx_md.block_table,
    )
    a["position_ids"] = pos
    a["token_valid"] = token_valid.to(torch.int32)
    a["kv_seq_lens"] = seq_lens

    return [a[name] for name in ARG_ORDER], (pos, seq, n_real)


# --- one-shot comparison -----------------------------------------------------

_OP = None
_DONE: set = set()


def _registered():
    global _OP
    if _OP is None:
        from pypto.torch import init, register

        kcsa, _ = kernel()
        init()
        _OP = register(kcsa.decode_csa_attn_tp1_test, "pypto_csa::attention_csa")
    return _OP


_SEEN_ADDRS = {}
_AUDITS = [0]
_OWNERSHIP_AUDITS = [0]


def audit_shared_pool_ownership(
    metadata_list, kv_cache, n_real: int, seq: int,
) -> None:
    """Verify groups sharing each physical allocation own disjoint blocks.

    This intentionally reads a scalar back to the host and therefore runs only
    on eager/warm-up calls, never while ACLGraph capture is active.
    """
    cmp_md, cst_md, ist_md, idx_md, swa_md = (
        m.decode for m in metadata_list
    )
    host_rows = torch.arange(
        n_real, device=ist_md.input_positions.device,
    ) * seq
    positions = ist_md.input_positions.index_select(0, host_rows).long()
    raw_current_column = torch.div(
        positions, VLLM_PAGE, rounding_mode="floor",
    )
    raw_in_range = (raw_current_column >= 0) & (
        raw_current_column < swa_md.block_table.shape[1]
    )
    raw_current_column = raw_current_column.clamp(
        min=0, max=swa_md.block_table.shape[1] - 1,
    )
    raw_current_page = swa_md.block_table[:n_real].gather(
        1, raw_current_column.reshape(-1, 1),
    ).reshape(-1)
    active = raw_in_range & (raw_current_page > 0)

    # Include both the old seven-row history and every submitted token. An S=6
    # step can cross an 8-row state page or a 128-row cache page.
    history = positions.reshape(-1, 1) + torch.arange(
        -7, seq, device=positions.device,
    ).reshape(1, -1)
    state_valid = (history >= 0) & active.reshape(-1, 1)
    state_valid &= history < cmp_md.seq_lens[:n_real].reshape(-1, 1)
    state_columns = torch.div(
        history.clamp_min(0), VLLM_STATE_PAGE, rounding_mode="floor",
    ).clamp(max=ist_md.block_table.shape[1] - 1)
    inner_ids = ist_md.block_table[:n_real].gather(1, state_columns)
    inner_ids = inner_ids.masked_select(state_valid & (inner_ids > 0))

    last_positions = torch.minimum(positions + seq - 1, cmp_md.seq_lens[:n_real] - 1)
    compressed_rows = torch.div(
        last_positions + 1, COMPRESS_RATIO, rounding_mode="floor",
    ).clamp_min(0)
    index_page_count = torch.div(
        compressed_rows + VLLM_PAGE - 1,
        VLLM_PAGE,
        rounding_mode="floor",
    )
    columns = torch.arange(
        idx_md.block_table.shape[1], device=positions.device,
    ).reshape(1, -1)
    index_ids = idx_md.block_table[:n_real]
    index_ids = index_ids.masked_select(
        active.reshape(-1, 1)
        & (columns < index_page_count.reshape(-1, 1))
        & (index_ids > 0)
    )

    inner_set = set(inner_ids.detach().cpu().tolist())
    index_set = set(index_ids.detach().cpu().tolist())
    inner_capacity = kv_cache[4].shape[0]
    if any(block >= inner_capacity for block in inner_set | index_set):
        raise NativeLayoutError(
            "inner/index block table contains a physical page outside the pool"
        )
    if inner_set & index_set:
        raise NativeLayoutError(
            "inner-state and index block tables overlap in the shared page pool"
        )

    state_ids = cst_md.block_table[:n_real].gather(1, state_columns)
    state_ids = state_ids.masked_select(state_valid & (state_ids > 0))

    raw_first = (positions - 127).clamp_min(0)
    first_raw_page = torch.div(raw_first, VLLM_PAGE, rounding_mode="floor")
    raw_columns = first_raw_page[:, None] + torch.arange(
        (128 + seq + VLLM_PAGE - 2) // VLLM_PAGE + 1, device=positions.device,
    )[None, :]
    raw_columns = torch.minimum(raw_columns, torch.div(last_positions, VLLM_PAGE, rounding_mode="floor")[:, None])
    raw_columns = raw_columns.clamp(min=0, max=swa_md.block_table.shape[1] - 1)
    raw_ids = swa_md.block_table[:n_real].gather(1, raw_columns)
    raw_ids = raw_ids.masked_select(active.reshape(-1, 1) & (raw_ids > 0))

    cmp_columns = torch.arange(
        cmp_md.block_table.shape[1], device=positions.device,
    ).reshape(1, -1)
    cmp_ids = cmp_md.block_table[:n_real]
    cmp_ids = cmp_ids.masked_select(
        active.reshape(-1, 1)
        & (cmp_columns < index_page_count.reshape(-1, 1))
        & (cmp_ids > 0)
    )
    # Main state and compressed KV share one physical pool, while raw KV has
    # its own allocation. Check both bounds and active ownership.
    main_sets = (
        ("main-state", set(state_ids.detach().cpu().tolist()), kv_cache[2].shape[0]),
        ("raw-kv", set(raw_ids.detach().cpu().tolist()), kv_cache[1].shape[0]),
        ("compressed-kv", set(cmp_ids.detach().cpu().tolist()), kv_cache[0].shape[0]),
    )
    for name, blocks, capacity in main_sets:
        if any(block >= capacity for block in blocks):
            raise NativeLayoutError(
                f"{name} block table contains a physical page outside its pool"
            )
    if main_sets[0][1] & main_sets[2][1]:
        raise NativeLayoutError(
            "main-state and compressed-KV block tables overlap in the shared page pool"
        )
    _OWNERSHIP_AUDITS[0] += 1


def audit_inputs(metadata_list, n_real: int) -> None:
    """Check that vLLM hands us the same buffers each step, on the first two.

    Capture bakes the address of every tensor read here into the recorded pass, so
    a buffer vLLM reallocates per step makes each replay read whatever now sits at
    the old address -- with no error anywhere. Block 0 matters for the same
    reason the recorded addresses matter at all.

    This reads tensor values, so the caller must keep it off the captured pass.
    """
    _AUDITS[0] += 1
    names = []
    for i, m in enumerate(metadata_list):
        d = m.decode
        for attr in ("block_table", "slot_mapping", "seq_lens", "input_positions"):
            t = getattr(d, attr, None)
            if isinstance(t, torch.Tensor):
                names.append((f"md{i}.{attr}", t))

    moved = [n for n, t in names
             if n in _SEEN_ADDRS and _SEEN_ADDRS[n] != t.data_ptr()]
    for n, t in names:
        _SEEN_ADDRS[n] = t.data_ptr()

    if _AUDITS[0] == 1:
        print("[pto-attn-audit] tensors=%d requests=%d" % (len(names), n_real),
              flush=True)
        return

    print("[pto-attn-audit] moved_between_steps=%s"
          % (",".join(moved) or "none"), flush=True)
    if moved:
        print("[pto-attn-audit] WARNING: those buffers are reallocated per step; "
              "an ACLGraph replay would read stale addresses", flush=True)


def compare_once(self, hidden_states, kv_cache, metadata_list, native_out, out_dir: str) -> bool:
    """Run the kernel on this step's real tensors and record how it compares.

    The kernel writes six caches, so this runs after the native path and reads
    caches the native path has already advanced: the output is not numerically
    comparable, and is not meant to be. What it answers is whether the native
    arguments assemble, bind and execute on live vLLM state at all.
    """
    impl = self.dsa_attn.impl
    layer = self.dsa_attn.layer_name
    if getattr(impl, "compress_ratio", 0) != COMPRESS_RATIO:
        # Only the ratio-4 layers carry the five cache groups this kernel needs.
        return False
    if metadata_list[0].decode is None:
        # A prefill step: this kernel is the decode path only. Returning False
        # leaves the caller's once-per-layer bookkeeping untouched, so the first
        # decode step still gets its turn.
        return False
    rec = {"layer": layer, "stage": "start"}
    path = Path(out_dir) / f"compare__{layer.replace('.', '_')}.json"

    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")

    try:
        seq = _env_int("PTO_ATTN_SEQ", 1)
        rec["seq"] = seq
        rec["stage"] = "build_args"
        save()
        args, plan = build_args(impl, hidden_states, kv_cache, metadata_list, seq, layer)
        rec["arg_shapes"] = {
            n: [list(a.shape), str(a.dtype), bool(a.is_contiguous())]
            for n, a in zip(ARG_ORDER, args)
        }
        rec["stage"] = "register"
        save()
        op = _registered()

        rec["stage"] = "launch"
        save()
        op(*args)
        torch.npu.synchronize()

        pto = args[-1].float()
        nat = native_out[: pto.shape[0]].float()
        rec["pto"] = {"finite": bool(torch.isfinite(pto).all().item()),
                      "absmax": pto.abs().max().item(),
                      "absmean": pto.abs().mean().item()}
        rec["native"] = {"absmax": nat.abs().max().item(),
                         "absmean": nat.abs().mean().item()}
        rec["max_abs_diff"] = (pto - nat).abs().max().item()
        denom = (pto.norm() * nat.norm()).clamp_min(1e-12)
        rec["cosine"] = ((pto * nat).sum() / denom).item()
        rec["ok"] = True
        rec["stage"] = "complete"
    except Exception:
        import traceback

        rec["ok"] = False
        rec["error"] = traceback.format_exc()
    finally:
        save()
        print(f"[pto-attn-compare] {layer} stage={rec['stage']} ok={rec.get('ok')}", flush=True)
    return True


_TALLY: dict = {}
_RAN = [0]


def _tally(layer: str, ratio, has_decode: bool) -> None:
    """Count what the substitution was offered, so a silent decline is visible."""
    k = (layer, int(ratio or 0), bool(has_decode))
    _TALLY[k] = _TALLY.get(k, 0) + 1
    if _TALLY[k] <= 3 or _TALLY[k] % 25 == 0:
        print("[pto-attn-offer] %s ratio=%s decode=%s n=%d"
              % (layer, ratio, has_decode, _TALLY[k]), flush=True)


# --- replacement -------------------------------------------------------------


def substitute(self, hidden_states, kv_cache, metadata_list, output) -> bool:
    """Run the kernel in place of the native attention and publish its result.

    Unlike :func:`compare_once` this owns the step: the kernel directly updates
    vLLM's live state and cache pages, and writes directly to its output buffer.
    """
    impl = self.dsa_attn.impl
    ratio = getattr(impl, "compress_ratio", 0)
    decode = metadata_list[0].decode
    _tally(self.dsa_attn.layer_name, ratio, decode is not None)
    if ratio != COMPRESS_RATIO or decode is None:
        return False
    seq = _env_int("PTO_ATTN_SEQ", 1)
    kcsa, _ = kernel()
    if not 1 <= seq <= kcsa.S:
        if "seq" not in _DEBUG_REFUSED:
            _DEBUG_REFUSED.add("seq")
            print(
                f"[pto-attn] declined host seq={seq}: native CSA currently "
                f"supports 1..{kcsa.S} tokens per request",
                flush=True,
            )
        return False
    if decode.input_positions.shape[0] % seq:
        if "position_rows" not in _DEBUG_REFUSED:
            _DEBUG_REFUSED.add("position_rows")
            print(
                "[pto-attn] declined: position rows are not divisible by host seq",
                flush=True,
            )
        return False
    n_offered = decode.input_positions.shape[0] // seq
    if n_offered > kcsa.B:
        # Under capture this is the padded graph batch, not the live request
        # count, and raising here would abort capture_model with a half-recorded
        # graph. Declining leaves that one descriptor on the native path.
        if "batch" not in _DEBUG_REFUSED:
            _DEBUG_REFUSED.add("batch")
            print(f"[pto-attn] declined a batch of {n_offered}: the kernel takes "
                  f"B={kcsa.B}", flush=True)
        return False
    try:
        args, plan = build_args(
            impl,
            hidden_states,
            kv_cache,
            metadata_list,
            seq,
            self.dsa_attn.layer_name,
            output=output,
        )
        _pos, _, n_real = plan
        if not capture_active():
            if _OWNERSHIP_AUDITS[0] < 2:
                audit_shared_pool_ownership(
                    metadata_list, kv_cache, n_real, seq,
                )
            if _AUDITS[0] < 2:
                audit_inputs(metadata_list, n_real)
    except NativeLayoutError as error:
        key = f"layout:{error}"
        if key not in _DEBUG_REFUSED:
            _DEBUG_REFUSED.add(key)
            print(f"[pto-attn] declined native layout: {error}", flush=True)
        return False

    _registered()(*args)

    _RAN[0] += 1
    # Capture happens once per descriptor and Python never runs on replay, so an
    # ungated print there costs one line per captured shape and is the only way
    # to tell a recorded pass from the warm-up that precedes it.
    cap = capture_active()
    if cap or _RAN[0] <= 5 or _RAN[0] % 10 == 0:
        print("[pto-attn-ran] n=%d tokens=%d capturing=%s"
              % (_RAN[0], n_real * seq, cap), flush=True)
    return True
