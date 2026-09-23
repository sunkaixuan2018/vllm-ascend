#!/usr/bin/env bash
# 建 DSV4 vLLM 原生运行环境（本机无 docker，服务端要自己装）。可重复执行，已装的步骤会跳过。
#
# 基础解释器用 /data/miniconda3/envs/vllm/bin/python（3.11，带可用 ssl）：
# 系统 /usr/local 的 python3.10 缺 ssl 模块，pip 连不上任何 HTTPS 源。
#
# 两个 editable 安装都必须 --no-build-isolation：vllm 0.20.2 的 build-system 要
# torch==2.11.0，隔离构建会在 overlay 里再下一份 2.11（几 GB、十几分钟），而运行时
# 要的是 torch 2.10.0。关掉隔离直接复用已装的 2.10.0，构建依赖下面显式装。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env_dsv4_vllm.sh"
BASE_PY="${BASE_PY:-/data/miniconda3/envs/vllm/bin/python}"

set -x
[ -x "$DSV4_VENV/bin/python" ] || "$BASE_PY" -m venv "$DSV4_VENV" || exit 1
source "$DSV4_VENV/bin/activate"
python -V

python -m pip install --upgrade pip setuptools wheel || exit 1
python -m pip install "torch==2.10.0" "torch-npu==2.10.0" "torchvision==0.25.0" || exit 1

# vllm / vllm-ascend 的 build-system 依赖，显式装进主环境供 --no-build-isolation 使用
python -m pip install "cmake>=3.26.1" ninja "packaging>=24.2" \
    "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" wheel jinja2 \
    pybind11 decorator attrs numpy pyyaml scipy psutil einops regex \
    googleapis-common-protos || exit 1

VLLM_TARGET_DEVICE=empty python -m pip install -e "$DSV4_VLLM_SRC" --no-build-isolation || exit 1

# vllm-ascend 钉 triton-ascend==3.2.1，但 PyPI（华为云与清华源都一样）只发到 3.2.0：
# 3.2.1 只存在于官方镜像里。这里装 3.2.0，再用 --no-deps 装 vllm-ascend 绕开该钉子。
# 若 DSV4 的 triton kernel 真的要 3.2.1，从镜像里拷（845MB，装在 triton/ 包里）：
#   docker run --rm -v <out>:/out <IMG> bash -lc \
#     'cp -a /usr/local/python3.11.15/lib/python3.11/site-packages/{triton,triton*-*.dist-info} /out/'
#
# arctic-inference==0.1.1 也跳过：它只有 sdist，构建依赖钉 torch==2.7.0，隔离构建会再下
# 一份 2.7 并拿它编 C++（nanobind/grpcio-tools），而本用例用的是 DSV4 自带的 MTP 投机解码，
# 不走 Arctic 那条路。若运行时真的 import 到它，再单独处理。
sed -e 's/^triton-ascend==.*/triton-ascend==3.2.0/' \
    -e '/^arctic-inference/d' "$DSV4_VLLM_ASCEND_SRC/requirements.txt" \
    > "$TMPDIR/vllm_ascend_requirements.txt"
python -m pip install -r "$TMPDIR/vllm_ascend_requirements.txt" || exit 1

# 编译期必须关掉 torch 的设备后端自动加载。CMakeLists 用
# append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path") 跑 python 去取
# TorchConfig.cmake 的位置，而裸 import torch 会连带 import torch_npu，后者在没有设备锁
# 的普通进程里拿不到驱动/DCMI，直接抛 "Failed to load the backend extension: torch_npu"。
# 探测失败后 CMAKE_PREFIX_PATH 里就没有 torch，find_package(Torch REQUIRED) 报
# "Could not find a package configuration file provided by Torch" —— 看着像路径问题，
# 其实是 import 失败。**只在编译期设这个变量**：运行时 vllm 要真的用 torch_npu。
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
CMAKE_PREFIX_PATH="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CMAKE_PREFIX_PATH
python -m pip install -e "$DSV4_VLLM_ASCEND_SRC" --no-build-isolation --no-deps || exit 1
set +x

# triton-ascend 的 NPU 驱动在首次使用时把 backends/ascend/npu_utils.cpp 现场编译成 .so。
# 3.2.0 那份写的是 rtLimitType_t::RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE，而本机 CANN 9.0.0
# （innerversion V100R001C10SPC001B250）里这个枚举项叫 RT_LIMIT_TYPE_SIMT_STACK_SIZE
# （同一个 slot=1，只是改了名）。名字对不上，vllm 起服务时就在
# import vllm_ascend.ops.triton.* 处炸掉：
#   npu_utils.cpp:321: error: 'RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE' is not a member of rtLimitType_t
# 镜像里那份是 3.2.1（PyPI 没发），本机另一个能跑的环境用 3.2.2（只有 cp310，我们是 cp311），
# 两者都拿不来，所以按本机头文件的拼写改名。CANN 换成有新名字的版本后这段会自动跳过。
NPU_UTILS="$DSV4_VENV/lib/python3.11/site-packages/triton/backends/ascend/npu_utils.cpp"
RT_HDR=$(grep -rl "RT_LIMIT_TYPE_STACK_SIZE" "$ASCEND_HOME_PATH" --include=*.h 2>/dev/null | head -1)
if [ -f "$NPU_UTILS" ] && [ -n "$RT_HDR" ] &&
   grep -q "RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE" "$NPU_UTILS" &&
   ! grep -q "RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE" "$RT_HDR" &&
   grep -q "RT_LIMIT_TYPE_SIMT_STACK_SIZE" "$RT_HDR"; then
    cp -n "$NPU_UTILS" "$NPU_UTILS.orig"
    sed -i 's/RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE/RT_LIMIT_TYPE_SIMT_STACK_SIZE/g' "$NPU_UTILS"
    echo "[patch] npu_utils.cpp: RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE -> RT_LIMIT_TYPE_SIMT_STACK_SIZE（按本机 CANN 拼写）"
fi

# 不占卡的自检：torch_npu 的 import 要驱动，放到下面的 task-submit 里做。
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "
import torch, vllm, vllm_ascend
from vllm_ascend import _version
print('torch       ', torch.__version__)
print('vllm        ', vllm.__version__)
print('vllm_ascend ', _version.version, '(', vllm_ascend.__file__, ')')
" || exit 1

# 占一张卡确认 torch_npu 能真正起来（普通进程没有 DCMI 权限，必须走 task-submit）
task-submit --device auto --device-num 1 --timeout 600 --max-time 300 \
    --run "$DSV4_VENV/bin/python -c 'import torch, torch_npu; print(\"torch_npu\", torch_npu.__version__, \"npu_count\", torch.npu.device_count())'" \
    || echo "[warn] torch_npu 设备自检未通过，跑用例前先查这里"
echo BUILD_ENV_DONE
