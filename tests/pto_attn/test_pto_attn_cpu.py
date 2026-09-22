"""CPU checks for native vLLM page views, no device and no PyPTO."""

import sys
import types
from unittest.mock import patch

import torch

import vllm_ascend.attention.pto_attn as pa

B, S, DIM = 4, 1, 32
T = B * S
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


print("== rope table ==")
t = torch.arange(T * 64, dtype=torch.float32).view(T, 1, 1, 64)
rope_module = types.ModuleType("vllm_ascend.ops.rope_dsv4")
rope_module._ROPE_STATE = types.SimpleNamespace(
    layer_info={"layer": ("config", ["default"])},
    static_cache={"config": (t, t)},
)
with patch.dict(sys.modules, {"vllm_ascend.ops.rope_dsv4": rope_module}):
    cos, sin = pa._native_rope_tables("layer")
    check("shape", list(cos.shape) == [T, 64])
    check("no copy", cos.data_ptr() == t.data_ptr() == sin.data_ptr())
    check("no conversion", cos.dtype == torch.float32 and torch.equal(cos, t.view(T, 64)))
    rope_module._ROPE_STATE.static_cache["config"] = (t.bfloat16(), t.bfloat16())
    try:
        pa._native_rope_tables("layer")
        check("reject incompatible dtype", False)
    except pa.NativeLayoutError:
        check("reject incompatible dtype", True)

print("== native padded-page views ==")
main_state_parent = torch.arange(
    3 * 16 * 2048, dtype=torch.float32,
).view(3, 16, 2048)
main_state = torch.as_strided(
    main_state_parent,
    (3, 8, 2048),
    (32768, 2048, 1),
)
main_full = pa._full_page_view(main_state, 16, (2048,))
check("main state page shape", list(main_full.shape) == [3, 16, 2048])
check("main state aliases parent", main_state.data_ptr() == main_full.data_ptr())
check("main state page is contiguous", main_full.is_contiguous())

raw_parent = torch.arange(
    3 * 128 * 512, dtype=torch.float32,
).to(torch.bfloat16).view(3, 128, 1, 512)
raw_full = pa._full_page_view(raw_parent, 128, (1, 512))
check("raw KV page shape", list(raw_full.shape) == [3, 128, 1, 512])
check("raw KV aliases parent", raw_full.data_ptr() == raw_parent.data_ptr())
check("raw KV page is contiguous", raw_full.is_contiguous())

index_parent = torch.arange(3 * 130 * 128, dtype=torch.int32).to(torch.int8).view(3, 130, 128)
inner_live = torch.as_strided(
    index_parent.view(torch.float32),
    (3, 8, 512),
    (4160, 512, 1),
)
index_key = index_parent[:, :128]
index_full = pa._full_page_view(index_key, 130, (128,))
check("index padded view shape", list(index_full.shape) == [3, 130, 128])
check("index tail is visible", torch.equal(index_full, index_parent))
check("inner and index share allocation", inner_live.data_ptr() == index_full.data_ptr())

print()
print("FAILED:" if fails else "ALL PASS", fails or "")
sys.exit(1 if fails else 0)
