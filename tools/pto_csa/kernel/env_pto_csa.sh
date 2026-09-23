# PTO 侧环境：只读借用 linyifan 已构建的 PyPTO + 本机 PTOAS 0.61。
# source 本文件后可直接跑 pypto-lib 的 kernel harness。
#
# PyPTO 的选取踩过两轮：
#   - all_pyptos/* 全停在 ptoas v0.54 且 simpler 未构建，用不了；
#   - linyifan 的 /data/linyifan/pypto（v0.61、构建齐全）pypto_core.so 权限不可读；
#   - yj_subdir/nalinaly-...(v0.57) 可读但太旧：今天的 pypto-lib 要 RunConfig 的
#     enable_chip_swimlane 字段，9/9 那版 PyPTO 没有 —— 传了报 TypeError，
#     不传报 AttributeError（golden/runner.py 无条件读 cfg.enable_chip_swimlane），两头堵。
# 最终用 yj_subdir/kernel-csa-feat-pypto：pin v0.61、simpler a2a3 产物齐、
# pypto_core 与 _task_interface 都可读，且 runtime 里有 enable_chip_swimlane。
# RunConfig 来自 pypto.runtime，所以 PyPTO 与 pypto-lib 的新旧必须配套。
#
# 解释器必须是 cp310：_task_interface 是 cpython-310 ABI。

_PTO_NOUNSET=0
case $- in *u*) _PTO_NOUNSET=1; set +u ;; esac
source /usr/local/Ascend/cann-9.0.0/set_env.sh
if [ "$_PTO_NOUNSET" = 1 ]; then set -u; fi
unset _PTO_NOUNSET

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export PYPTO_ROOT=/data/sunkaixuan/yj_subdir/kernel-csa-feat-pypto
export PYPTO_LIB_ROOT="${PYPTO_LIB_ROOT:-/data/sunkaixuan/sunkaixuan_subdir/all_libs/pypto-lib-csa-20260917}"
export PTOAS_ROOT=/usr/local/ptoas/0.61
export PTO_ISA_ROOT="${PTO_ISA_ROOT:-$PYPTO_ROOT/runtime/build/pto-isa}"

# 借用解释器的包（torch 2.10 / torch_npu 2.10.0.post4，cp310）
BORROWED_SITE=/data/linyifan/.conda/envs/vllm-pypto/lib/python3.10/site-packages
export PTO_PY="${PTO_PY:-/data/linyifan/.conda/envs/vllm-pypto/bin/python}"

# pypto-lib 的 harness 要求仓库根在 PYTHONPATH（golden/ 是顶层包），
# 而模型脚本用**平铺 import**（`from config import FLASH`、`from utils import ...`），
# 所以模型目录本身也必须在路径上，否则报 ImportError: cannot import name 'FLASH'。
export PYPTO_MODEL_DIR="${PYPTO_MODEL_DIR:-$PYPTO_LIB_ROOT/models/deepseek_v4_flash_mtp}"
export PYTHONPATH="$PYPTO_LIB_ROOT:$PYPTO_MODEL_DIR:$PYPTO_ROOT/python:$PYPTO_ROOT/runtime:$PYPTO_ROOT/runtime/python:$BORROWED_SITE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

# 缓存/HOME 全部落到本任务目录，不污染借用环境
WORK="${PTO_CSA_WORK:-/data/sunkaixuan/skx_log_output/csa_b_tier}"
mkdir -p "$WORK"/{tmp,home,ascend,work,triton,hf} 2>/dev/null
export HOME="$WORK/home" TMPDIR="$WORK/tmp"
export ASCEND_CACHE_PATH="$WORK/ascend" ASCEND_WORK_PATH="$WORK/work"
export TRITON_CACHE_DIR="$WORK/triton" HF_HOME="$WORK/hf" HF_HUB_OFFLINE=1

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

echo "[pto-csa] python=$PTO_PY"
echo "[pto-csa] pypto=$PYPTO_ROOT  ptoas=$PTOAS_ROOT"
echo "[pto-csa] pypto-lib=$PYPTO_LIB_ROOT"
echo "[pto-csa] model dir=$PYPTO_MODEL_DIR"
