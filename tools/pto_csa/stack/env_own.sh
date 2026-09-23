# 自建独立栈的运行环境（myserver）。用法: source env_own.sh
#
# 和 codex_sh/dsv4_vllm_20260917/env_dsv4_vllm.sh 的差别只有一条：**不借任何私人目录**。
#   解释器  自带 CPython 3.11.16（python-build-standalone，含 ssl）  ← 原来借 linyifan 的 conda
#   ATB     own_stack/atb-9.0.0（从 linyifan 那份拷来，可重定位）    ← 原来直接 source 他的
#   PyPTO   own_stack/pypto，从上游 clone 后自建                     ← 原来拷的源码快照
#   Torch   venv 内的 torch 2.10.0                                   ← 参考环境构建期链的是别人 conda 里的
# CANN 9.0.0 与 PTOAS 0.61 在 /usr/local 下，是 root 的厂商工具包，不算私人目录。

OWN=/data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918
export OWN_ROOT="$OWN"
export OWN_PY_HOME="$OWN/python311"
export OWN_VENV="${OWN_VENV:-$OWN/.venv}"
export OWN_WORK="${OWN_WORK:-/data/sunkaixuan/skx_log_output/own_stack}"

export PYPTO_ROOT="$OWN/pypto"
# pypto#2822 turned persistent JIT caching off by default; without this every
# call recompiles (~3.5 s) and throughput drops by ~8x with no error surfaced.
export PYPTO_CACHE=1
export PYPTO_LIB_ROOT="${PYPTO_LIB_ROOT:-/data/sunkaixuan/sunkaixuan_subdir/all_libs/pypto-lib-csa-20260917}"
# 登录环境自带 PTOAS_ROOT=/usr/local/bin/ptoas-bin（机器级默认，指的是可执行文件不是根目录），
# 所以这里必须无条件覆盖；要换版本用 PTOAS_VER。
export PTOAS_ROOT="/usr/local/ptoas/${PTOAS_VER:-0.61}"
export PTO_ISA_ROOT="${PTO_ISA_ROOT:-$PYPTO_ROOT/runtime/build/pto-isa}"
export OWN_VLLM_SRC="$OWN/vllm-v0.20.2"
export OWN_VLLM_ASCEND_SRC="$OWN/vllm-ascend-v0.20.2rc1"

# 用户 site 里有别人的 editable .pth 元路径查找器，会劫持 import，必须关掉。
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

# ── 编译器 ────────────────────────────────────────────────────────────
# 自带的 CPython 来自 python-build-standalone，是在 Clang 容器里编的，它的 sysconfig
# 把 CC/CXX/LDSHARED 记成 clang、AR 记成 /tools/llvm/bin/llvm-ar。setuptools 与
# scikit-build-core 建扩展时会照抄这些值导出成环境变量，而本机只有 gcc，于是 cmake 报
#   Could not find the compiler specified in the environment variable CC: clang
# 看着像没装编译器，其实是解释器自带的元数据在指路。显式盖掉。
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export AR=/usr/bin/ar
export LDSHARED="/usr/bin/gcc -pthread -shared"

# ── CANN / ATB ────────────────────────────────────────────────────────
# CANN/ATB 的 set_env.sh 直接读 LD_LIBRARY_PATH / PYTHONPATH 等可能未定义的变量，
# 在 set -u 的调用方里 source 会直接退出，所以先摘掉 nounset。
_OWN_NOUNSET=0
case $- in *u*) _OWN_NOUNSET=1; set +u ;; esac

source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
[ -f /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh ] &&
    source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh

# ATB 必须与 CANN 9.0.0 配套，且用 cxx_abi_1 那一份（torch_npu 是 C++11 ABI 编的）。
# 宿主机 /usr/local/Ascend/nnal/atb 下只有 8.3/8.5，版本对不上 —— 用它起 vllm 能起来，
# 但一发请求所有 ATB 算子就失败（AtbCommon.cpp:203 / LinearOperation），故障只在推理时暴露。
# 这份 9.0.0 的 set_env.sh 用 BASH_SOURCE 定位自己，拷到哪都能用。
ATB_SET_ENV="${ATB_SET_ENV:-$OWN/atb-9.0.0/atb/set_env.sh}"
if [ -r "$ATB_SET_ENV" ]; then
    source "$ATB_SET_ENV" --cxx_abi=1
else
    echo "[own] 错误: 找不到 $ATB_SET_ENV" >&2
fi

if [ "$_OWN_NOUNSET" = 1 ]; then set -u; fi
unset _OWN_NOUNSET

# ── venv ──────────────────────────────────────────────────────────────
[ -d "$OWN_VENV" ] && source "$OWN_VENV/bin/activate"

# ── 缓存重定向（多人共用机器，别写共享 HOME）────────────────────────────
export TMPDIR="$OWN_WORK/cache/tmp"
export XDG_CACHE_HOME="$OWN_WORK/cache/xdg"
export HF_HOME="$OWN_WORK/cache/hf" HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="$OWN_WORK/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$OWN_WORK/cache/inductor"
export TRITON_CACHE_DIR="$OWN_WORK/cache/triton"
export ASCEND_CACHE_PATH="$OWN_WORK/cache/ascend"
export ASCEND_WORK_PATH="$OWN_WORK/cache/work"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
         "$ASCEND_CACHE_PATH" "$ASCEND_WORK_PATH" 2>/dev/null

# ── 芯片型号 ───────────────────────────────────────────────────────────
# vllm-ascend 的 setup.py 与运行时都要 SOC_VERSION。不设的话 setup.py 会去 shell 调
# npu-smi 探测，而普通用户拿不到 DCMI（exit 187），构建断在 "Get chip info failed"。
# 本机是 Ascend910 + NPU Name 9392（CI 机那台是 9391，不能照抄）。
export SOC_VERSION="${SOC_VERSION:-ascend910_9392}"

# ── 运行时开关（与 CI 机容器 env 一致）─────────────────────────────────
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=600
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=1
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1
export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1
export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5

# ── 网络 ──────────────────────────────────────────────────────────────
# 到 127.0.0.1 的 vLLM 走代理会 502；PyPI 直连不通但华为云镜像可达。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
export PIP_TRUSTED_HOST=mirrors.huaweicloud.com

echo "[own] python      = $(command -v python) $(python -V 2>&1)"
echo "[own] pypto       = $PYPTO_ROOT"
echo "[own] pypto-lib   = $PYPTO_LIB_ROOT"
echo "[own] ptoas       = $PTOAS_ROOT"
echo "[own] atb         = $ATB_SET_ENV"
