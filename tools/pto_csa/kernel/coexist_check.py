"""验证 vendor 算子与 PyPTO 能否在同一个 cp310 进程里共存。"""
import json, os, sys, traceback

r = {"python": sys.version.split()[0], "stage": "start"}
try:
    import torch
    r["torch"] = torch.__version__
    r["stage"] = "torch_npu"
    import torch_npu
    r["torch_npu"] = torch_npu.__version__

    r["stage"] = "vendor"
    from vllm_ascend.utils import bootstrap_custom_op_env
    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    r["vendor_op"] = hasattr(torch.ops._C_ascend, "npu_sparse_attn_sharedkv")
    r["vendor_so"] = vllm_ascend.vllm_ascend_C.__file__

    r["stage"] = "pypto"
    import pypto, pypto.language as pl  # noqa: F401
    r["pypto"] = pypto.__file__
    from pypto.runtime import RunConfig  # noqa: F401
    import simpler  # noqa: F401
    r["simpler"] = "ok"

    r["stage"] = "pypto_lib"
    import decode_sparse_attn_csa as csa
    r["csa_kernel"] = {"compress_ratio": csa.COMPRESS_RATIO, "topk": csa.TOPK}

    r["stage"] = "complete"
    r["ok"] = True
except Exception:
    r["ok"] = False
    r["error"] = traceback.format_exc().splitlines()[-3:]
print(json.dumps(r, ensure_ascii=False, indent=2))
sys.exit(0 if r.get("ok") else 1)
