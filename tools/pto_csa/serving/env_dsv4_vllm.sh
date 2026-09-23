# DSV4-Flash MTP vLLM 运行环境（myserver / liteserver-hps-148e-00001）
# 用法: source env_dsv4_vllm.sh
#
# 对标 CI 机镜像 quay.io/ascend/vllm-ascend:v0.20.2rc1-a3-openeuler 的运行环境：
#   vllm        v0.20.2        (github.com/vllm-project/vllm, tag v0.20.2)
#   vllm-ascend v0.20.2rc1     (releases/v0.20.2rc, commit 367b8e62)
#   torch 2.10.0 / torch_npu 2.10.0 / CANN 9.0.0
# 本机没有 docker 权限（sunkaixuan 不在 docker 组），所以走源码 + venv 原生路径。

WS=/data/sunkaixuan/sunkaixuan_subdir/all_servings
export DSV4_VLLM_SRC="$WS/vllm-v0.20.2"
export DSV4_VLLM_ASCEND_SRC="$WS/vllm-ascend-v0.20.2rc1"
export DSV4_VENV="${DSV4_VENV:-$WS/.venv-dsv4-vllm}"
export DSV4_WORK="${DSV4_WORK:-/data/sunkaixuan/skx_log_output/dsv4_vllm}"

# 用户 site 里有别人的 editable .pth 元路径查找器，会劫持 import，必须关掉。
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

# ── CANN / ATB ────────────────────────────────────────────────────────
# CANN/ATB 的 set_env.sh 直接读 LD_LIBRARY_PATH / PYTHONPATH / ZSH_VERSION 等
# 可能未定义的变量，在 set -u 的调用方里 source 会直接退出，所以先摘掉 nounset。
_DSV4_NOUNSET=0
case $- in *u*) _DSV4_NOUNSET=1; set +u ;; esac

source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
[ -f /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh ] &&
    source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
# ATB 必须用与 CANN 9.0.0 配套的 9.0.0，且是 cxx_abi_1 那一份（torch_npu 用 C++11 ABI 编译）。
# 宿主机 /usr/local/Ascend/nnal/atb 下只有 8.3.RC1.alpha003 和 8.5.0.alpha001（latest 指向后者），
# 版本对不上 —— 用它起 vllm 能起来，但一发请求所有 ATB 算子就失败：
#   Exception raised from OperationSetup at op_plugin/utils/custom_functions/atb/AtbCommon.cpp:203
#   RuntimeError: ... the current working operator name is LinearOperation
# 走 aclnn 的算子不受影响，所以服务能正常 startup，故障只在推理时暴露。
# 9.0.0 是 linyifan 从官方镜像 quay.io/ascend/vllm-ascend:v0.22.1rc1-a3 里抠出来的：
#   docker run --rm -v <out>:/out <IMG> bash -lc 'cp -a /usr/local/Ascend/nnal/atb/9.0.0 /out/'
ATB_SET_ENV="${ATB_SET_ENV:-/data/linyifan/atb-from-image/9.0.0/atb/set_env.sh}"
if [ -r "$ATB_SET_ENV" ]; then
    source "$ATB_SET_ENV" --cxx_abi=1
else
    echo "[dsv4-vllm] 警告: 找不到配套的 ATB 9.0.0（$ATB_SET_ENV），回退到宿主机版本，ATB 算子很可能失败" >&2
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
fi

if [ "$_DSV4_NOUNSET" = 1 ]; then set -u; fi
unset _DSV4_NOUNSET

# ── venv ──────────────────────────────────────────────────────────────
[ -d "$DSV4_VENV" ] && source "$DSV4_VENV/bin/activate"

# ── 缓存重定向 ─────────────────────────────────────────────────────────
# 这台机器多人共用，编译/算子缓存写到共享 HOME 会互相踩。
export TMPDIR="$DSV4_WORK/cache/tmp"
export XDG_CACHE_HOME="$DSV4_WORK/cache/xdg"
export HF_HOME="$DSV4_WORK/cache/hf" HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="$DSV4_WORK/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$DSV4_WORK/cache/inductor"
export TRITON_CACHE_DIR="$DSV4_WORK/cache/triton"
export ASCEND_CACHE_PATH="$DSV4_WORK/cache/ascend"
export ASCEND_WORK_PATH="$DSV4_WORK/cache/work"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
         "$ASCEND_CACHE_PATH" "$ASCEND_WORK_PATH" 2>/dev/null

# ── 芯片型号 ───────────────────────────────────────────────────────────
# vllm-ascend 的 setup.py 与运行时都要 SOC_VERSION。不设的话 setup.py 会去 shell
# 调 npu-smi 探测，而本机普通用户直接跑 npu-smi 拿不到 DCMI（exit 187），构建就断在
# "Get chip info failed"。本机取值按 setup.py 的 A3 规则推出：
#   npu-smi info -t board -i 0 -c 0  →  Chip Name=Ascend910, NPU Name=9392, 无 Chip Type
#   ⇒ (chip_name + "_" + npu_name).lower() = ascend910_9392
# 注意 CI 机那台是 ascend910_9391，不是同一个 A3 SKU，不能照抄容器 env。
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
# 本机到 127.0.0.1 的 vLLM 走代理会 502；PyPI 直连不通但华为云镜像可达。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
export PIP_TRUSTED_HOST=mirrors.huaweicloud.com

echo "[dsv4-vllm] python=$(command -v python3) $(python3 -V 2>&1)"
echo "[dsv4-vllm] vllm src=$DSV4_VLLM_SRC"
echo "[dsv4-vllm] vllm-ascend src=$DSV4_VLLM_ASCEND_SRC"
