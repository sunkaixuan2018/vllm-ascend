"""从 pypto-lib 内联进来的 PTO decode kernel 源码。

vllm-ascend 不依赖 pypto-lib 仓：这些 .py 就是全部所需。仍然依赖 pypto 工具链
（`pypto.language` / `pypto.ir` / `pypto.torch`）和 simpler 运行时 —— 那是编译器，
不是可内联的东西。

上游 commit：15d9ae75aaba452594bff0fe16500fadc250564d

与上游的唯一差异是 import 形式：上游用裸顶层 import（靠 PYTHONPATH 解析），
这里改成包内相对 import，于是不需要把 pypto-lib 的模型目录挂到 sys.path 上，
也不会把 config / utils 这些通用名字暴露成顶层模块。

重新同步上游：
    python -m vllm_ascend.attention.pto_kernels.resync /path/to/pypto-lib
"""

PYPTO_LIB_COMMIT = "a791b1d7e1ba3f1ddae68e343709a8b8a418f35b"
VARIANTS = {
    "dspark": ("models/deepseek_v4_flash_dspark", ("decode_csa",)),
    "mtp": ("models/deepseek_v4_flash_mtp", ("decode_sparse_attn_csa",)),
}
