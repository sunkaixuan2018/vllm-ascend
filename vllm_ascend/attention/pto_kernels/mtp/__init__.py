"""只有 sparse attention 的小 kernel（静态形状），当前在跑的那条。

上游：hw-native-sys/pypto-lib  models/deepseek_v4_flash_mtp/
入口模块：decode_sparse_attn_csa —— sparse_attn_test
"""

UPSTREAM_SUBDIR = "models/deepseek_v4_flash_mtp"
ENTRY_MODULES = ("decode_sparse_attn_csa",)
