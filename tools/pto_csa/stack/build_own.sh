#!/usr/bin/env bash
# 从零建一套**完全本地**的栈：自带 Python、自建 PyPTO/simpler、自带 ATB。
# 可重复执行，已完成的阶段会跳过（靠产物存在性判断）。
#
# 和 codex_sh/dsv4_vllm_20260917/build_env.sh 同源，坑的处理逐条照搬，只是：
#   · 基础解释器换成 own_stack/python311（自带 ssl，不再借 conda）
#   · 多了 simpler 与 PyPTO 两步自建
#   · 所有源码树都在 own_stack 下，不碰行动者正在用的那套
#
# 单阶段重跑: STAGES=pypto ./build_own.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env_own.sh"
OWN="$OWN_ROOT"
LOG="$OWN/logs"
mkdir -p "$LOG"
STAGES="${STAGES:-venv torch deps simpler pypto vllm vllm_ascend triton check}"
want() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
step() { echo; echo "######## $* ($(date +%H:%M:%S))"; }

# ── venv ──────────────────────────────────────────────────────────────
if want venv; then
    step "venv <- $OWN_PY_HOME"
    [ -x "$OWN_VENV/bin/python" ] || "$OWN_PY_HOME/bin/python3" -m venv "$OWN_VENV" || exit 1
    source "$OWN_VENV/bin/activate"
    python -VV
    python -m pip install --upgrade pip setuptools wheel || exit 1
fi
source "$OWN_VENV/bin/activate"

# ── torch ─────────────────────────────────────────────────────────────
if want torch; then
    step "torch 2.10.0 / torch-npu 2.10.0"
    python -c "import torch" 2>/dev/null ||
        python -m pip install "torch==2.10.0" "torch-npu==2.10.0" "torchvision==0.25.0" || exit 1
fi

# ── 构建依赖 ───────────────────────────────────────────────────────────
if want deps; then
    step "构建依赖（供 --no-build-isolation 使用）"
    # vllm 0.20.2 的 build-system 要 torch==2.11.0，隔离构建会在 overlay 里再下一份 2.11
    # （几 GB、十几分钟），而运行时要的是 2.10.0。关掉隔离，构建依赖在这里显式装。
    python -m pip install "cmake>=3.26.1" ninja "packaging>=24.2" \
        "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" wheel jinja2 \
        pybind11 decorator attrs numpy pyyaml scipy psutil einops regex \
        googleapis-common-protos cloudpickle || exit 1
fi

# 编译期必须关掉 torch 的设备后端自动加载：CMakeLists 跑 python 去取 TorchConfig.cmake
# 的位置，而裸 import torch 会连带 import torch_npu，后者在没有设备锁的普通进程里拿不到
# 驱动/DCMI，直接抛 "Failed to load the backend extension"。探测失败后 CMAKE_PREFIX_PATH
# 里就没有 torch，find_package(Torch REQUIRED) 报 "Could not find ... Torch" —— 看着像
# 路径问题，其实是 import 失败。**只在编译期设**：运行时 vllm 要真的用 torch_npu。
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# ── simpler（PyPTO 的 runtime 子模块）──────────────────────────────────
if want simpler; then
    step "simpler @ $(git -C "$PYPTO_ROOT/runtime" rev-parse --short HEAD)"
    # scikit-build-core 的 editable 安装把扩展放进 venv 的 site-packages，
    # 不是落回 runtime/python/，所以按能不能 import 判断，别按路径判断。
    if python -c "import _task_interface" 2>/dev/null; then
        echo "  已有 _task_interface，跳过"
    else
        # build-constraints.txt 是 PyPTO/simpler 实际构建against 的输入钉子（cmake/nanobind/
        # ninja/scikit-build-core）。只在这两步加，别全局加 —— vllm 要的 cmake 版本不同。
        PIP_CONSTRAINT="$PYPTO_ROOT/build-constraints.txt" \
            python -m pip install "scikit-build-core>=0.10.0" "nanobind>=2.0.0,<3" "cmake>=3.15" ninja || exit 1
        PIP_CONSTRAINT="$PYPTO_ROOT/build-constraints.txt" \
            python -m pip install --no-build-isolation -e "$PYPTO_ROOT/runtime" \
            2>&1 | tee "$LOG/simpler_build.log" | tail -25
        python -c "import _task_interface" 2>/dev/null || {
            echo "  !! simpler 没产出 _task_interface，见 $LOG/simpler_build.log"; exit 1; }
    fi
    python -c "import _task_interface as t; print('  _task_interface:', t.__file__)"
fi

# ── PyPTO ─────────────────────────────────────────────────────────────
if want pypto; then
    step "PyPTO @ $(git -C "$PYPTO_ROOT" rev-parse --short HEAD)"
    # 两个模块必须同时在：只看 pypto_core 的话，一份只编了 program 模式（torch 扩展 OFF）
    # 的构建会被误判成已完成。文档也要求换 torch 版本后两个模块一起重编，不能只重编一个。
    if [ -n "$(ls "$PYPTO_ROOT/python/pypto/"pypto_core.cpython-*.so 2>/dev/null)" ] &&
       [ -n "$(ls "$PYPTO_ROOT/python/pypto/"_torch_npu.cpython-*.so 2>/dev/null)" ]; then
        echo "  已有 pypto_core + _torch_npu，跳过"
    else
        CMAKE_PREFIX_PATH="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
        export CMAKE_PREFIX_PATH
        # 裸 cmake 调用时 find_package(nanobind) 找不到它 —— nanobind 的 cmake 配置装在
        # site-packages 里，路径要自己问它要（pip install 那条路是 scikit-build-core 代劳的）。
        NB_DIR="$(python -m nanobind --cmake_dir)"
        cmake -B "$PYPTO_ROOT/build" -S "$PYPTO_ROOT" \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DPYPTO_BUILD_TORCH_NPU=ON \
            -DPython_EXECUTABLE="$OWN_VENV/bin/python" \
            -Dnanobind_DIR="$NB_DIR" \
            2>&1 | tee "$LOG/pypto_cmake.log" | tail -15 || exit 1
        cmake --build "$PYPTO_ROOT/build" --parallel "$(nproc)" \
            2>&1 | tee "$LOG/pypto_build.log" | tail -15
        for m in pypto_core _torch_npu; do
            [ -n "$(ls "$PYPTO_ROOT/python/pypto/$m".cpython-*.so 2>/dev/null)" ] || {
                echo "  !! PyPTO 没产出 $m，见 $LOG/pypto_build.log"; exit 1; }
        done
    fi
    ls -1 "$PYPTO_ROOT/python/pypto/"*.cpython-*.so
fi

# ── vllm ──────────────────────────────────────────────────────────────
if want vllm; then
    step "vllm editable"
    python -c "import vllm" 2>/dev/null ||
        VLLM_TARGET_DEVICE=empty python -m pip install -e "$OWN_VLLM_SRC" --no-build-isolation || exit 1
fi

# ── vllm-ascend ───────────────────────────────────────────────────────
if want vllm_ascend; then
    step "vllm-ascend editable"
    # vllm-ascend 钉 triton-ascend==3.2.1，但华为云与清华源都只发到 3.2.0（3.2.1 只在官方
    # 镜像里）。装 3.2.0，再用 --no-deps 装 vllm-ascend 绕开该钉子。
    # arctic-inference==0.1.1 也跳过：只有 sdist，构建依赖钉 torch==2.7.0，隔离构建会再下
    # 一份 2.7 并拿它编 C++；本用例用 DSV4 自带的 MTP 投机解码，不走 Arctic 那条路。
    sed -e 's/^triton-ascend==.*/triton-ascend==3.2.0/' -e '/^arctic-inference/d' \
        "$OWN_VLLM_ASCEND_SRC/requirements.txt" > "$TMPDIR/va_requirements.txt"
    python -m pip install -r "$TMPDIR/va_requirements.txt" || exit 1
    CMAKE_PREFIX_PATH="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export CMAKE_PREFIX_PATH
    python -m pip install -e "$OWN_VLLM_ASCEND_SRC" --no-build-isolation --no-deps \
        2>&1 | tee "$LOG/vllm_ascend_build.log" | tail -20 || exit 1
fi

# ── triton 补丁 ────────────────────────────────────────────────────────
if want triton; then
    step "triton npu_utils.cpp 枚举改名"
    # triton-ascend 的 NPU 驱动首次使用时把 backends/ascend/npu_utils.cpp 现场编译成 .so。
    # 3.2.0 那份写的是 RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE，而本机 CANN 9.0.0 里这个枚举项
    # 叫 RT_LIMIT_TYPE_SIMT_STACK_SIZE（同一个 slot，只改了名）。名字对不上，vllm 起服务时
    # 就在 import vllm_ascend.ops.triton.* 处炸掉。CANN 换成有新名字的版本后这段自动跳过。
    NPU_UTILS=$(ls "$OWN_VENV"/lib/python3.*/site-packages/triton/backends/ascend/npu_utils.cpp 2>/dev/null | head -1)
    RT_HDR=$(grep -rl "RT_LIMIT_TYPE_STACK_SIZE" "$ASCEND_HOME_PATH" --include=*.h 2>/dev/null | head -1)
    if [ -n "$NPU_UTILS" ] && [ -n "$RT_HDR" ] &&
       grep -q "RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE" "$NPU_UTILS" &&
       ! grep -q "RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE" "$RT_HDR" &&
       grep -q "RT_LIMIT_TYPE_SIMT_STACK_SIZE" "$RT_HDR"; then
        cp -n "$NPU_UTILS" "$NPU_UTILS.orig"
        sed -i 's/RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE/RT_LIMIT_TYPE_SIMT_STACK_SIZE/g' "$NPU_UTILS"
        echo "  已改名（按本机 CANN 拼写）"
    else
        echo "  不需要改（或找不到文件）: ${NPU_UTILS:-未找到}"
    fi
fi

# ── 自检 ──────────────────────────────────────────────────────────────
if want check; then
    step "不占卡自检"
    PYTHONPATH="$PYPTO_ROOT/python:$PYPTO_ROOT/runtime:$PYPTO_ROOT/runtime/python:$PYPTO_LIB_ROOT:$PYPTO_LIB_ROOT/models/deepseek_v4_flash_mtp" \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python - <<'PY'
import sys, torch, vllm, vllm_ascend
from vllm_ascend import _version
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__)
print("vllm        ", vllm.__version__)
print("vllm_ascend ", _version.version)
import pypto, pypto.language as pl
print("pypto       ", pypto.__file__)
from pypto.runtime import RunConfig
print("RunConfig   ok")
import simpler
print("simpler     ", simpler.__file__)
PY
    echo "  退出码=$?"
fi
echo; echo "BUILD_OWN_DONE $(date +%H:%M:%S)"
