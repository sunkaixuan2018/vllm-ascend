# DSV4-Flash MTP vLLM 用例（从 CI 机移植到 myserver）

CI 机（`ci-runner@1.95.79.227`，工作目录 `/data/l00955553`）上的 vLLM 用例迁到本机。
迁的是代码与执行骨架，参数不照搬：卡号、DP、batch、prompt 长度都是入口脚本的变量。

## 服务端跑的是哪个版本

CI 机用现成镜像 `m.daocloud.io/quay.io/ascend/vllm-ascend:v0.20.2rc1-a3-openeuler`，
不是私有分支。镜像里 `/vllm-workspace` 下两棵源码树：

| 组件 | 仓库 | 分支 / tag | commit |
| ---- | ---- | ---------- | ------ |
| vllm | github.com/vllm-project/vllm | tag `v0.20.2` | `bc150f50` |
| vllm-ascend | github.com/vllm-project/vllm-ascend | 分支 `releases/v0.20.2rc`，tag `v0.20.2rc1` | `367b8e62` |

`367b8e62` 的提交题目就是 *"Reduce sampling is reconstructed to eliminate all patch
behaviors and support DFlash and MTP (#9946)"* —— DSV4-Flash + MTP 的支持就在这个
release 分支上。镜像内其余版本：python 3.11.15、torch 2.10.0、torch_npu 2.10.0、
transformers 5.5.3、CANN 9.0.0、SOC `ascend910_9391`。

本机已按同一 commit clone：

- `/data/sunkaixuan/sunkaixuan_subdir/all_servings/vllm-v0.20.2`
- `/data/sunkaixuan/sunkaixuan_subdir/all_servings/vllm-ascend-v0.20.2rc1`

## 本机与 CI 机的差异

| 事项 | CI 机 | myserver |
| ---- | ----- | -------- |
| 服务承载 | docker 容器（现成镜像） | 无 docker 权限（`sunkaixuan` 不在 docker 组，`task-submit` 也是本用户身份），只能源码 + venv 原生跑 |
| 设备锁 | `npu-lock` | `task-submit`（整轮在一个 task 内持锁） |
| 卡数 | 16 张全占（DP16） | 共 16 张，但本用户默认白名单只有奇数 8 张；DP16 需 `--ignore-whitelist` 且整机空闲 |
| 模型 | `/data/l00955553/model/dsv4-flash-w8a8` | `/data/models/` 下四个同架构变体，按卡数选（见下） |
| profiler 产物 | 容器内 root 权限，`docker exec` 读 | 本用户文件，直读 |
| PyPI | 直连 | 直连不通，华为云/清华镜像可达（`env_dsv4_vllm.sh` 已设 `PIP_INDEX_URL`） |

### 显存账：按 index 算，不要用 `du -sh`

`/data/models/dsv4-flash-w8a8/` 目录 598G，但里面有一个 **不在 `model.safetensors.index.json`
里的文件** `pypto-deepseek-v4-stacked-r8.safetensors`（322.8 GiB，PyPTO 用的 stacked 版）。
**vLLM 只读 index 里的 46 个分片，合计 274.4 GiB**。拿目录大小算显存会把卡数需求估高一倍。

单层 6.18–6.21 GiB，其中路由专家 6.008 GiB（96.8%），attn 全部只有 148–204 MiB：

| 层类型 | 层号 | 单层 | 非专家部分 | CSA(compressor+indexer) |
| ------ | ---- | ---- | ---------- | ----------------------- |
| 无 CSA（compress_ratio=0） | 0、1 | 6.179 GiB | 175 MiB | — |
| 含 CSA，ratio=4 | 2,4,…,42 | 6.207 GiB | 204 MiB | 28.5 MiB（含 indexer） |
| 含 CSA，ratio=128 | 3,5,…,41 | 6.181 GiB | 177 MiB | 8.25 MiB（无 indexer） |

EP=DP 时每 rank ≈ 专家 258.3/DP + 每 rank 都要复制的 16.1 GiB（层内非专家 7.9 + embed/head 8.2）：

| DP | 每 rank | 单卡 61.28 GiB |
| -- | ------- | -------------- |
| 2 | ~153 GiB | ✗ OOM（实测如此） |
| 8 | ~48.4 GiB | ✓ 装得下，KV cache 只剩 ~7 GiB |
| 16 | ~32 GiB | ✓ 宽裕 |

**8 卡装得下整模 43 层**，不需要截层，也不需要整机 16 卡。官方 nightly 配置用 TP=8/DP=2
（TP 会把 attention 和 expert 都切开）。

### 挡路的是权重命名，不是层数/卡数

这份 checkpoint 的量化 scale 张量后缀是 **`.scale`**（34,035 个），`weight_scale` 一个都没有；
而 v0.20.2rc1 的量化 linear 注册的参数名是 `weight_scale`/`weight_offset`
（`quantization/methods/w8a8_dynamic.py:70-71`），整个 vllm_ascend 里**没有任何
`.scale → weight_scale` 的映射**（`patch/worker/patch_weight_utils.py` 只处理
`fa_q/k/v.scale` 和 `indexer.*`）。

MoE 专家的 scale 走 `expert_params_mapping` 那条独立通道，**attention 的 linear 没有映射**，
在 `models/deepseek_v4.py:1348` 直接 `params_dict[name]` 字面查表，于是必然 KeyError：

```
KeyError: 'model.layers.0.self_attn.wo_b.scale'
```

它只是加载顺序里第一个撞上的张量，后面还有三万多个同类。**与截层无关**（写进 config.json 或
用 `--hf-overrides` 都一样）、**与卡数无关**（DP2/DP8 一样）、**与 config 版本无关**
（和 CI 哈希一致的原版 config 照样报）。

根因是这份 checkpoint 的 `config.json` 声明 `quant_method: compressed-tensors`，但张量命名用的是
昇腾 modelslim 那套 `.scale`，而 compressed-tensors(llmcompressor) 的标准命名是 `weight_scale`
—— **声明的量化格式和权重文件的命名约定对不上**。

仓库自带的官方配置
`tests/e2e/nightly/single_node/models/configs/DeepSeek-V4-Flash-W8A8-A3.yaml` 指向的是另一个
产物 `Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp`，而且跑法也和 CI 脚本不同：

```yaml
--tensor-parallel-size 8      # CI 脚本是 TP=1
--data-parallel-size 2        # CI 脚本是 DP=16
--safetensors-load-strategy prefetch
--tokenizer-mode deepseek_v4
VLLM_ASCEND_APPLY_DSV4_PATCH: "1"   # 我们没设（它 gate 的是 KV cache + 投机解码的 patch）
USE_MULTI_BLOCK_POOL / USE_MULTI_GROUPS_KV_CACHE / HCCL_BUFFSIZE=1024
transformers: "5.9.0"               # 我们装的是 5.5.3
```

要继续推 DSV4，下一步是换成官方那份权重，而不是调参数或截层。

### DSV4 用例目前的状态

**这个用例在 CI 机上也没跑通过，而且死得更早** —— CI 最近一轮
（`tmp/dsv4_mtp_vllm_20260915_211038_dp16_bs4_locked`）的 `status.txt` 是
`server failed to become healthy for bs=4`，栈是
`worker_base.py:317 init_device → vllm_ascend/worker/worker.py:325 → model_runner_v1.py:265`，
报 `ACL stream synchronize failed, error code:507034` —— **在 `load_model()` 之前**，
它的日志里一行 `Loading safetensors` 都没有。更早三轮连 status 文件都没有。
所以没有可对齐的基准，而且本机反而走得更远（进到了权重加载）。

本机额外的限制：

- **卡**：`task-submit` 的可分配范围被全局白名单钉在 8 张奇数卡
  （`/home/pypto-tools/pto-task/state/available_devices`，root 所有），`--ignore-whitelist`
  与 `TASKQUEUE_IGNORE_WHITELIST=1` 都无效。不过按上面的显存账，**8 卡够用**，
  只有想复刻 CI 的 DP16 才需要让管理员放开。
- **另外两个小体积变体都不可用**：`DeepSeek-V4-Flash[-0731]`（fp8）被 vllm-ascend 拒绝
  （`deepseek_v4_fp8 quantization is currently not supported in npu`）；
  `dsv4-flash-0731-dspark-w8a8` 多了 `dspark_*` 系列配置和 `li_cache_scheme`，
  v0.20.2rc1 不认识，加载同样报 `wo_b.scale` 的 KeyError。

### 2026-09-17 实测矩阵（产物都在 `/data/sunkaixuan/skx_log_output/dsv4_vllm/`）

| 模型 | 层数 | 卡 | MTP | 结果 |
| ---- | ---- | -- | --- | ---- |
| `DeepSeek-V4-Flash`（fp8） | 43 | 8 | 开 | 配置阶段被拒：`deepseek_v4_fp8 quantization is currently not supported in npu` |
| `dsv4-flash-0731-dspark-w8a8` | 43 | 8 | 开 | `KeyError: model.layers.0.self_attn.wo_b.scale` |
| `dsv4-flash-w8a8`（本机 yyd config） | 3（`--hf-overrides`） | 8 / 2 | 开 | 同上 KeyError |
| `dsv4-flash-w8a8-ci`（CI 原版 config） | 3（`--hf-overrides`） | 2 | 开 | 同上 KeyError |
| `dsv4-flash-w8a8-ci-l3`（层数写进 config） | 3 | 2 | 关 | 同上 KeyError |
| `dsv4-flash-w8a8-ci` | 43 | 2 | 开 | **无 KeyError**，加载到 2% 时 `NPU out of memory`（2 卡装不下，预期） |

两条都证实了：量化路径两种情况下都正常选中（日志都有 `llmcompressor Quantization now`、
都没有 `Falling back to UnquantizedLinearMethod`），所以不是 scheme 选择的问题。

**还没定论的一件事**：43 层那次是 2 卡跑到 2% 就 OOM 了，不能确定它真的越过了出错的那个
张量。要分辨"截层才触发"还是"本机根本加载不了"，需要 43 层 + 8 卡（8 rank 每个 ~75G >
61G，会 OOM，但会推进到很深才 OOM）：

```bash
MODEL=/data/sunkaixuan/skx_log_output/dsv4_vllm/models/dsv4-flash-w8a8-ci   DEVICE_NUM=8 BATCH_SIZE=4 HEALTH_TRIES=200 MAX_TIME=2400 WAIT_TIMEOUT=2400   ./run_dsv4_mtp_vllm.sh
# 看 server.log：先出现 wo_b.scale 的 KeyError ⇒ 本机加载不了；
#                只出现 out of memory 且加载百分比很高 ⇒ 截层是触发点。
```

这台机器同时只跑一个 8 卡用例，排队可能很久（实测前面压了两个等 26–38 分钟的 8 卡任务）。

## 文件

| 文件 | 来源 | 说明 |
| ---- | ---- | ---- |
| `run_dsv4_mtp_vllm.sh` | 对应 CI 机 `tmp/run_dsv4_mtp_vllm_sweep.sh` | 入口：算产物目录，把整轮交给 `task-submit` 持锁执行 |
| `dsv4_case_inner.sh` | 同上（用例执行体部分） | 锁内执行：起服务 → 等健康 → 跑客户端 → 收产物 → 回收进程组 |
| `dsv4_mtp_vllm_client.py` | CI 机 `tmp/dsv4_mtp_vllm_client.py`，原样 | 流式压测客户端（TTFT/TPOT/吞吐），纯标准库 + argparse，无需改动 |
| `ascendc_profile.py` | CI 机 `profile-util/profilling/ascendc_profile.py` | 压测 + torch profiler + trace 解析；`docker exec` 改成容器可选 |
| `prompt/` | CI 机 `profile-util/prompt/` | `ascendc_profile.py` 的 prompt 语料，默认 `long_prompt_3.5k.txt` |
| `env_dsv4_vllm.sh` | 新写 | CANN/ATB/venv/缓存重定向/HCCL 开关，对齐镜像内 env |
| `make_ci_model_dir.sh` | 新写 | 用符号链接 + 原版 config 拼一个与 CI 一致的模型目录；`LAYERS=N` 截层 |
| `build_env.sh` | 新写 | 一次性建原生 venv（torch/torch_npu/vllm/vllm-ascend） |
| `run_qwen3_vllm.sh` + `qwen3_case_inner.sh` | 新写，参数抠自 CI 的 `vllm-qwen3` 容器 | Qwen3-14B 压测 + profiling（CI 上真正跑出过数据的那条），单卡 TP=1 |
| `fetch_official_shards.sh` / `make_official_l3_dir.sh` / `auto_after_fetch.sh` | 新写 | 从 ModelScope 拉官方 DSV4 权重的指定分片、拼成截层模型目录、下完自动接力跑 |

`docker` 相关调用在移植版里的对应关系写在 `dsv4_case_inner.sh` 头部注释里。

## 用法

```bash
# 服务端环境（首次先按下面「原生服务端环境」建好 venv）
source /data/sunkaixuan/codex_sh/dsv4_vllm_20260917/env_dsv4_vllm.sh

# Qwen3-14B（CI 上跑通过的那条，单卡即可）
cd /data/sunkaixuan/codex_sh/dsv4_vllm_20260917
./run_qwen3_vllm.sh                    # VERBOSE=1 看逐算子明细

# 跑一轮 DSV4 MTP（默认 bs=4；卡数/模型/batch 都是变量）
cd /data/sunkaixuan/codex_sh/dsv4_vllm_20260917
DEVICE_NUM=8 BATCH_SIZE=4 ./run_dsv4_mtp_vllm.sh

# 产物
cat /data/sunkaixuan/skx_log_output/dsv4_vllm/latest_out.txt
```

`ascendc_profile.py` 对任何已起好的 vLLM 服务都能用（原生模式不传 `--container`）：

```bash
python3 ascendc_profile.py --base-url http://127.0.0.1:8113 --model dsv4 \
    --profile-dir "$OUT/profile" --server-log "$OUT/server.log" -v
```

注意它的 decode 切步口径是按 Qwen3 稠密模型标的（`ArgMaxV2` 切步、`MatMulV2`/`MatMulV3`
判 decode/prefill）。DSV4-Flash 是 MoE + MTP，算子构成不同，这套切步规则要先核对再采信。

## 原生服务端环境

本机没有 docker，服务端自己装，`build_env.sh` 一把建好（按镜像的版本组合）：

```bash
cd /data/sunkaixuan/codex_sh/dsv4_vllm_20260917
nohup ./build_env.sh > /data/sunkaixuan/skx_log_output/dsv4_vllm/build_env.log 2>&1 &
```

它做四件事：建 venv → `torch==2.10.0 / torch-npu==2.10.0 / torchvision==0.25.0`
→ `VLLM_TARGET_DEVICE=empty pip install -e vllm-v0.20.2` → `pip install -e vllm-ascend-v0.20.2rc1`
（这一步要编 C++，依赖 CANN 9.0.0 + ATB cxx_abi_1，`env_dsv4_vllm.sh` 里已 source）。

装的过程踩到八个坑，`build_env.sh` 里都已处理，换机器/重装时留意：

- **基础解释器不能用系统 `python3.10`**：`/usr/local/lib/python3.10` 那份缺 `ssl` 模块，
  pip 连不上任何 HTTPS 源（报 `Can't connect to HTTPS URL because the SSL module is
  not available`）。改用 `/data/miniconda3/envs/vllm/bin/python`（3.11.11，OpenSSL 3.5.7），
  版本也更贴近镜像里的 3.11.15。换解释器用 `BASE_PY=` 覆盖。
- **两个 editable 安装都要 `--no-build-isolation`**：vllm 0.20.2 的 build-system 钉
  `torch==2.11.0`，隔离构建会在 overlay 里再下一份 2.11（几 GB、十几分钟），而运行时要的是
  2.10.0。关掉隔离直接复用已装的 2.10.0，构建依赖脚本里显式装。
- **必须先 `export SOC_VERSION`**：不设的话 `vllm-ascend/setup.py` 会 shell 调 `npu-smi`
  探芯片，而本机普通用户拿不到 DCMI（exit 187），构建断在 `Get chip info failed`。
  本机是 **`ascend910_9392`**（`npu-smi info -t board -i 0 -c 0` 给 Chip Name=Ascend910、
  NPU Name=9392、无 Chip Type，按 setup.py 的 A3 规则拼），**和 CI 机的 `ascend910_9391`
  不是同一个 A3 SKU**，不能照抄容器 env。
- **`triton-ascend==3.2.1` 装不到**：PyPI（华为云与清华源一样）只发到 3.2.0，3.2.1 只存在于
  官方镜像里。脚本装 3.2.0 并用 `--no-deps` 装 vllm-ascend 绕开这个钉子。若 DSV4 的 triton
  kernel 真要 3.2.1，从镜像里拷（装在 `triton/` 包里，845MB）。

- **`arctic-inference==0.1.1` 跳过**：它只有 sdist，构建依赖钉 `torch==2.7.0`，隔离构建会
  再下一份 2.7 并拿它编 C++（nanobind/grpcio-tools）。本用例走的是 DSV4 自带的 MTP 投机
  解码，不是 Arctic 那条路；真被 import 到再单独处理。

- **编译期要 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`**，这是最容易误判的一个。vllm-ascend 的
  `CMakeLists.txt` 用 `append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")`
  跑一个 python 去问 TorchConfig.cmake 在哪，而裸 `import torch` 会连带 import
  `torch_npu`，后者在没有设备锁的普通进程里拿不到驱动/DCMI，抛
  `Failed to load the backend extension: torch_npu`。探测失败 ⇒ prefix path 里没有 torch
  ⇒ `find_package(Torch REQUIRED)` 报 **"Could not find a package configuration file
  provided by Torch"**——看着像 CMake 路径问题，实际是 import 挂了，往 CMAKE_PREFIX_PATH
  里塞路径没用。**只在编译期设**：运行时 vllm 要真的用 torch_npu，所以用例必须在
  `task-submit` 的设备锁里跑。

- **triton-ascend 的 `npu_utils.cpp` 要按本机 CANN 拼写改名**。triton 的 NPU 驱动在首次
  使用时现场编译 `triton/backends/ascend/npu_utils.cpp`，3.2.0 那份写
  `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE`，而本机 CANN（`V100R001C10SPC001B250`，头文件
  `cann-9.0.0/aarch64-linux/pkg_inc/runtime/runtime/base.h`）里这项叫
  `RT_LIMIT_TYPE_SIMT_STACK_SIZE`（同一个 slot=1，只是改了名）。名字对不上，服务启动时
  在 `import vllm_ascend.ops.triton.*` 处编译失败，报 `status=server_unhealthy`。
  `build_env.sh` 探到头文件用旧拼写就自动 sed 改名（原文件备份为 `.orig`），CANN 升级后
  这段自动跳过。

- **ATB 要用 9.0.0，不能用宿主机那份**（最隐蔽的一个）。`/usr/local/Ascend/nnal/atb/` 下只有
  `8.3.RC1.alpha003` 和 `8.5.0.alpha001`（`latest` 指向后者），与 CANN 9.0.0 不配套。用它
  **服务能正常 startup**（走 aclnn 的算子不受影响），但一发请求所有 ATB 算子就失败：
  `OperationSetup at .../atb/AtbCommon.cpp:203` + `current working operator name is
  LinearOperation` + `ERR00100 PTA call acl api failed`，客户端看到 HTTP 500 与
  `EngineDeadError`。配套的 9.0.0 在 `/data/linyifan/atb-from-image/9.0.0/atb/set_env.sh`
  （从官方镜像 `v0.22.1rc1-a3` 里抠出来的），`env_dsv4_vllm.sh` 已改为优先用它、缺失时告警回退。

**PyPI 直连不通**，华为云镜像可达（`https` 与 `http` 都通），`env_dsv4_vllm.sh` 已设
`PIP_INDEX_URL`；GitHub 反而直连可达，不要开代理。

另一条路是让管理员把 `sunkaixuan` 加进 `docker` 组（组里已有 9 个用户），
那样可以直接拉同一个镜像，和 CI 机保持二进制级一致，`dsv4_case_inner.sh` 只需
换回 `docker run` 那一段。
