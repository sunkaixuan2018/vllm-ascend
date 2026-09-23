# DSV4-Flash 三层 vLLM ／ 把 CSA 换成 PTO 算子 —— 复现手册

两条路线：**运行 A** 原生 vllm-ascend 跑 DeepSeek-V4-Flash 前三层；**运行 B** 同一套环境，
把 ratio-4 层 decode 路径的 CSA 换成 pypto-lib 的 PTO 算子。除 CSA 外参数完全一致，可直接对比。

已验证（2026-09-18，三层 + 单卡 + `--enforce-eager`）：两条路线都 `status=passed`；
PTO 相对 vendor `max_abs 0.047 / mean_abs 0.0072`（vendor `|mean|` 0.92，即 0.8% 相对误差）。

---

## 1. 环境

机器 `myserver` = `liteserver-hps-148e-00001`，账号 `sunkaixuan`，Ascend A3，
`SOC_VERSION=ascend910_9392`。占卡的活儿一律走 `task-submit`（脚本已包好）。

| 组件 | 版本 / 路径 |
| --- | --- |
| CANN | `/usr/local/Ascend/cann-9.0.0` |
| ATB | `/data/linyifan/atb-from-image/9.0.0/atb/set_env.sh --cxx_abi=1`（必须 9.0.0，宿主机 `/usr/local/Ascend/nnal/atb` 下那几个版本不配套） |
| PTOAS | `/usr/local/ptoas/0.61`（机器上另有 0.45–0.60，必须用 0.61） |
| venv（两条路线共用） | `/data/sunkaixuan/sunkaixuan_subdir/all_servings/.venv-dsv4-vllm310`，Python **3.10.20** |
| 关键包 | `torch 2.10.0` · `torch_npu 2.10.0` · `transformers 5.5.3` · `triton-ascend 3.2.0` · `numpy 2.2.6` |

**解释器必须是 cp310**：PyPTO 的 `_task_interface` 是 cpython-310 ABI 的 `.so`。
该 venv 里 vLLM 与 vllm-ascend 都已 editable 装齐，直接用。

### ATB：必须 9.0.0，且宿主机自带的那几个是错的

ATB = Ascend Transformer Boost，随 CANN 的 NNAL 包发布的 Transformer 加速算子库
（`libatb.so` / `libasdops.so` / `libatb_mixops.so` / `liblcal.so`）。vLLM 不直接调它，
链路是 `vllm-ascend → torch_npu 的 op_plugin（libop_plugin_atb.so）→ libatb.so`，
所以在 `vllm_ascend/` 里 grep 不到 "atb"。

本机可用的一份（1.1 GB，太大不入库）：

```
/data/linyifan/atb-from-image/9.0.0/atb/set_env.sh --cxx_abi=1
```

`cxx_abi_0` / `cxx_abi_1` 对应 GCC 的 `_GLIBCXX_USE_CXX11_ABI`，必须和 torch 编译时用的
那个一致，本环境取 `--cxx_abi=1`。这份是从官方镜像
`quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` 里抠出来的；换机器时照样从该镜像取。

**宿主机 `/usr/local/Ascend/nnal/atb/` 下只有 `8.3.RC1.alpha003` 和 `8.5.0.alpha001`
（`latest` 指向后者），与 CANN 9.0.0 不配套。** 用错版本的失败形态极具迷惑性：
**起服务时一切正常** —— 模型加载、graph capture、健康检查全过（走 aclnn 的算子不受影响），
**一发请求才炸**，而且所有 ATB 算子一起炸：

```
Exception raised from OperationSetup at op_plugin/.../atb/AtbCommon.cpp:203
RuntimeError: ... current working operator name is LinearOperation
ERR00100 PTA call acl api failed
```

客户端那头看到的是 HTTP 500 + `EngineDeadError`。不知道这回事会一路去查模型/权重/参数，
而根因在一个日志里从没出现过的库版本上。

---

## 2. 代码版本

### vllm

```
路径    /data/sunkaixuan/sunkaixuan_subdir/all_servings/vllm-v0.20.2
remote  https://github.com/vllm-project/vllm.git
tag     v0.20.2
commit  bc150f50299199599673614f80d12a196f377655   (2026-05-05)
HEAD    detached（停在 tag 上）        改动  无
```

### vllm-ascend　← 我们的改动都在这个仓

```
路径    /data/sunkaixuan/sunkaixuan_subdir/all_servings/vllm-ascend-v0.20.2rc1
remote  https://github.com/vllm-project/vllm-ascend
分支线  origin/releases/v0.20.2rc
tag     v0.20.2rc1
commit  367b8e62da799870a7476ce34f5f7658589a8aad   (2026-06-03)
HEAD    detached（停在 tag 上）
```

### pypto-lib　← CSA 算子本体

```
路径    /data/sunkaixuan/sunkaixuan_subdir/all_libs/pypto-lib-csa-20260917
remote  https://github.com/hw-native-sys/pypto-lib.git
分支    main
commit  675f027f1bf9ce8f79740f0b2ed042b0c4b2c8b3   (2026-09-17)
        "Add: fuse DSpark decode into one device-state L2 (#1263)"
改动    无
用到的  models/deepseek_v4_flash_mtp/decode_sparse_attn_csa.py
```

### PyPTO　← 编译并下发 kernel 的运行时

```
路径    /data/sunkaixuan/yj_subdir/kernel-csa-feat-pypto
remote  git@github.com:hw-native-sys/pypto.git
分支    feat/kernel-mode-integration-test
commit  d46c609846e806485b6b4049c39ee00ed6479a95   (2026-09-16 16:56)
        "test(torch): validate registered kernel graph execution (08B) (#2795)"
源      /data/linyifan/pypto（该仓现已前进，我们用的是 d46c6098 那一刻的源码）
```

这份快照**盘上没有 `.git`**，上面的坐标是从源仓反查确认的。只拷了源码，`.so` 产物
（`pypto_core.cpython-310-*.so`、`runtime/python/_task_interface.cpython-310-*.so`、
`build_output`，共 3.9 GB）是在 myserver 上自建的。

**快照相对上游有 2 处本地改动，复现时必须一并带上：**

1. `python/pypto/_kernel_abi.py:23` →
   `SIMPLER_KERNEL_REVISION = "0d55fac71df44151059d67fcfe6c7dc6e0b33e66"`，
   对齐实际 checkout 的 `runtime/` 子模块。这个常量是**手维护**的，没有任何脚本会写它。
2. `python/pypto/torch/shutdown.py:93` → torch_npu 白名单放宽为 `("2.6.0.post2", "2.8.0")`。

### simpler　← PyPTO 的 `runtime/` 子模块

```
路径    /data/sunkaixuan/yj_subdir/kernel-csa-feat-pypto/runtime
remote  https://github.com/hw-native-sys/simpler
commit  0d55fac71df44151059d67fcfe6c7dc6e0b33e66   (2026-09-16 16:00)
        "Add: integrate TMR kernel execution coordination (K7) (#2250)"
分支    只在 feat/kernel-mode-integration-test 与 feat-kmi 上，不在 main
HEAD    detached                       改动  无
```

PyPTO `d46c6098` 自带的子模块指针是 `b5a0ea0c`，这里换成了 `0d55fac` —— 所以上面要手改
`SIMPLER_KERNEL_REVISION`。

### vllm-ascend 的本地改动清单

每个改过的文件都留了备份，`git status` 可核对。

| 文件 | 改动 | 作用 |
| --- | --- | --- |
| `vllm_ascend/attention/pto_csa.py` | 新增 694 行 | PTO 替换的全部逻辑；开关全不设时对 vLLM 无任何影响 |
| `vllm_ascend/attention/dsa_v1.py` | +67（备份 `.pre_pto`） | 三处钩子：暂存 PTO 需要的张量、记录 vendor stage-1、用 PTO 结果覆盖 decode 行 |
| `vllm_ascend/models/deepseek_v4.py` | +10（备份 `.orig`） | 加载器按层号过滤，截层后不再触碰超出 `num_hidden_layers` 的权重 |
| `vllm_ascend/profiler/torch_npu_profiler.py` | +13/−4（备份 `.pre_pto`） | 让 `with_stack` 跟随 `--profiler-config.torch_profiler_with_stack`；加 `VLLM_ASCEND_PROFILER_LEVEL`。**不打这个补丁，trace 里没有 Python 调用栈** |
| `csrc/utils/inc/kernel/moe_distribute_base.h` | −47/+8 | cp310 那份 `vllm_ascend_C` 扩展的编译前置 |

只跑**运行 A** 时，前三项不生效，后两项仍然需要。

### 脚手架脚本

脚本已随本仓提交到 `tools/pto_csa/`（原处 `/data/sunkaixuan/codex_sh/` 仍在，两边同源）：

```
tools/pto_csa/serving/            vLLM 服务侧
  env_dsv4_vllm.sh          CANN/ATB/venv/缓存/芯片型号/网络
  build_env.sh              从零建 venv（可重复执行）
  run_dsv4_mtp_vllm.sh      入口：挑端口 → task-submit → 整轮交给 inner
  dsv4_case_inner.sh        锁内执行体：起服务 + 健康探测 + 客户端 + 回收
  dsv4_mtp_vllm_client.py   压测客户端（TPOT/TTFT/吞吐）
  profile_decode_client.py  只录 decode 段的 profiling 客户端
  run_trace_ab.sh           两条路线各采一份 trace 的串行驱动
  ascendc_profile.py        单次 profiling（从 CI 机迁来）
  run_qwen3_vllm.sh         Qwen3-14B 对照用例
  model/                    权重准备
    fetch_official_shards.sh  从 ModelScope 拉官方权重分片
    make_official_l3_dir.sh   用已下分片拼「前 N 层」可加载目录
  prompt/                   定长 prompt 语料

tools/pto_csa/kernel/             PTO 侧验证工具
  run_adapter_check.sh      适配器离线自查（不要权重、不起服务，约 1 分钟）
  run_coexist_gate.sh       PyPTO 与 torch_npu 同进程共存门禁
  run_a_tier.sh             A 档：同一份 fixture，两条 CSA 路径各对 golden
  run_csa_b_tier.sh         B 档：PTO 版 CSA 出 trace
  run_eager_path.sh         eager 路径检查
  run_kmode_gdb.sh          kernel 模式段错误定位到栈帧
  analyze_dump.py           把 live 一步切成 kernel/注意力/o_proj 三段归因
  env_pto_csa.sh            PyPTO / PTOAS / pypto-lib 的环境

产物根目录  /data/sunkaixuan/skx_log_output/{dsv4_vllm,csa_b_tier}/
```

脚本里的绝对路径都是 myserver 上的实际位置；换机器时改各脚本开头的根变量
（`WS` / `RUN_ROOT` / `PYPTO_ROOT` / `PYPTO_LIB_ROOT` / `DSV4_VENV` / `MODEL`）即可。

---

## 3. 模型权重

用 ModelScope 上的 `Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp`（**不在 HuggingFace**），
即 vllm-ascend v0.20.2rc1 官方 nightly 配置指向的那份。整仓 279 GiB / 70 个分片，
跑前三层只要 6 片。

```bash
cd /data/sunkaixuan/codex_sh/dsv4_vllm_20260917
./fetch_official_shards.sh 1 2 3 4 5 70      # MTP=0 的三层只要这 6 片，约 24 GiB
LAYERS=3 MTP=0 ./make_official_l3_dir.sh     # 拼出 official-l3
./fetch_status.sh                            # 看下载进度
```

下载约 1.5 MiB/s，支持断点续传。盘上已是这个状态：源目录
`…/models/official/DeepSeek-V4-Flash-w8a8-mtp/`（27 GB，分片 `1 2 3 4 5 70`）；
三层目录 `…/models/official-l3/`（符号链接 + 改写过的 `config.json`：
`num_hidden_layers=3`、`num_nextn_predict_layers=0`，索引 7029 个张量）。

**层与 CSA 的对应**：`compress_ratios = (0, 0, 4, 128, 4, …)`，三层里**只有 layer 2 是
ratio-4**，也就是 PTO 替换唯一生效的那一层。

---

## 4. 建环境

现成的 venv 已经建好，一般不需要重跑。要重建：

```bash
cd /data/sunkaixuan/codex_sh/dsv4_vllm_20260917
DSV4_VENV=/data/sunkaixuan/sunkaixuan_subdir/all_servings/.venv-dsv4-vllm310 \
BASE_PY=/data/linyifan/.conda/envs/vllm-pypto/bin/python \
  ./build_env.sh
```

**建完必须手动打一次 triton 补丁**（`build_env.sh` 里那一步的路径写死了 `python3.11`，
cp310 venv 走不到）：

```bash
F=/data/sunkaixuan/sunkaixuan_subdir/all_servings/.venv-dsv4-vllm310/lib/python3.10/site-packages/triton/backends/ascend/npu_utils.cpp
cp -n $F $F.orig
sed -i 's/RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE/RT_LIMIT_TYPE_SIMT_STACK_SIZE/' $F
```

---

## 5. 运行 A：原生 vLLM 三层

```bash
cd /data/sunkaixuan/codex_sh/dsv4_vllm_20260917 && \
 MODEL=/data/sunkaixuan/skx_log_output/dsv4_vllm/models/official-l3 \
 DEVICE_NUM=1 BATCH_SIZE=4 MTP=0 PROMPT_TOKENS=4096 MAX_TOKENS=32 \
 SERVE_EXTRA="--enforce-eager" \
 DSV4_VENV=/data/sunkaixuan/sunkaixuan_subdir/all_servings/.venv-dsv4-vllm310 \
 bash run_dsv4_mtp_vllm.sh
```

脚本自己排队拿卡、起服务、等健康、跑客户端、回收，全程在一个 `task-submit` 任务里。

| 参数 | 说明 |
| --- | --- |
| `MODEL=…/official-l3` | **必须给**。默认值指向的是不能用的那份权重 |
| `MTP=0` | **必须给**。`official-l3` 的 `num_nextn_predict_layers=0` |
| `DEVICE_NUM=1` · `--enforce-eager` · cp310 venv | 为了和运行 B 对齐才这么写。只跑基线可放宽为 `DEVICE_NUM=2`、不加 eager、cp311 venv |

### 验收

```bash
O=$(cat /data/sunkaixuan/skx_log_output/dsv4_vllm/latest_out.txt)
cat $O/status.txt                                  # 期望 status=passed
cat $O/run_meta.txt                                # 本轮实际参数与 vllm-ascend commit
python3 -m json.tool $O/client/summary.json | head -40
```

`summary.json` 里 `all_measured_valid: true`、`requests_succeeded: 4`、`errors: []` 即为正常。

---

## 6. 运行 B：vLLM 三层 + PTO CSA

与运行 A 的唯一差别是多了 `PTO_CSA=1` 和 PyPTO 那几个路径：

```bash
cd /data/sunkaixuan/codex_sh/dsv4_vllm_20260917 && \
 MODEL=/data/sunkaixuan/skx_log_output/dsv4_vllm/models/official-l3 \
 DEVICE_NUM=1 BATCH_SIZE=4 MTP=0 PROMPT_TOKENS=4096 MAX_TOKENS=32 \
 SERVE_EXTRA="--enforce-eager" \
 DSV4_VENV=/data/sunkaixuan/sunkaixuan_subdir/all_servings/.venv-dsv4-vllm310 \
 PTO_CSA=1 \
 PTO_CSA_REPORT=/data/sunkaixuan/skx_log_output/csa_b_tier/btier_report.json \
 PYPTO_ROOT=/data/sunkaixuan/yj_subdir/kernel-csa-feat-pypto \
 PYPTO_LIB_ROOT=/data/sunkaixuan/sunkaixuan_subdir/all_libs/pypto-lib-csa-20260917 \
 PTOAS_ROOT=/usr/local/ptoas/0.61 \
 bash run_dsv4_mtp_vllm.sh
```

这一轮以下四项**不能放宽**：

| 参数 | 原因 |
| --- | --- |
| `DSV4_VENV` 指 cp310 | `_task_interface` 是 cpython-310 ABI |
| `DEVICE_NUM=1`（TP=1） | kernel 的 `H=64`、`D=4096` 是 pypto-lib `config.py` 的编译期常量，TP 一切分就对不上 |
| `--enforce-eager` | ACLGraph 捕获期不能做 PyPTO 那种 host 侧 D2H/H2D 同步 |
| `MTP=0` | 适配器目前在 S>1 下会回退到 vendor |

### 验收

`status=passed` 只说明服务没崩，**不代表 PTO 真的跑了**，要看日志：

```bash
O=$(cat /data/sunkaixuan/skx_log_output/dsv4_vllm/latest_out.txt)
cat $O/status.txt
grep "pto-csa" $O/server.log
```

期望输出：

```
[pto-csa] wo_b 由 torch.bfloat16 量化成 INT8 per-channel (kernel 签名不接受 BF16)
[pto-csa] compiled T=8 B=4 S=2 H=64 HEAD_DIM=512 device_id=0
[pto-csa] model.layers.2.self_attn.attn step=1  max_abs=0.0390625 mean_abs=0.0074 (vendor |mean|=0.91)
[pto-csa] model.layers.2.self_attn.attn step=10 max_abs=0.0390625 mean_abs=0.0072 (vendor |mean|=0.91)
```

判据：`max_abs ≈ 0.04`、`mean_abs ≈ 0.007`（vendor `|mean|` 约 0.9，即 0.8% 相对）。
只有 `model.layers.2` 会出现 —— 另两层不是 ratio-4，本来就不该被替换。

> **看性能数字时注意**：替换代码是**先让 vendor 算完整一轮，再用 PTO 的结果覆盖 decode
> 那几行**（这样 padding 行不会留下未初始化内存，并顺手得到上面那组逐步对拍值）。
> 所以运行 B 里 vendor 的 CSA 仍然在跑，其 device 总耗时**不是「PTO 方案的耗时」**。

若日志里一条 `[pto-csa]` 都没有：确认 `PTO_CSA=1` 已传进去（`run_meta.txt` 里有）、
venv 是 cp310、`PYPTO_ROOT` / `PYPTO_LIB_ROOT` 都给了。若有 `compiled T=8 …` 但没有逐步
对拍行，说明每步都被回退，`PTO_CSA_REPORT` 的 json 里 `runner.fallbacks` 会写明原因。

---

## 7. 采 trace（可选）

```bash
cd /data/sunkaixuan/codex_sh/dsv4_vllm_20260917 && bash run_trace_ab.sh
```

两轮串行、除 CSA 外参数完全一致，**只录 decode 段**：客户端走流式，等第一个 token
吐出来（prefill 已结束）再 `POST /start_profile`，录满 8 个 decode token 再
`POST /stop_profile`。

产物 `<OUT>/profile/**/ASCEND_PROFILER_OUTPUT/trace_view.json`（约 93 MB / 141 MB），
含 `Python` / `Ascend Hardware` / `CANN` / `AI Core Freq` / `Overlap Analysis` 五条泳道，
事件的 `args["Call stack"]` 是完整的 `文件(行号): 函数` 帧链。PTO 那份里有 3,688 个事件
的栈落在 `pto_csa.py:run_decode` ← `dsa_v1.py:forward`。

`--profiler-config` 既接受整块 JSON，也接受点号形式 `--profiler-config.profiler=torch`；
`VLLM_TORCH_PROFILER_DIR` 这个环境变量在 0.20.2 里已经没人读了。

---

## 8. 开关一览

| 变量 | 作用 |
| --- | --- |
| `PTO_CSA=1` | 打开替换（只作用于 ratio-4 层的 decode 路径） |
| `PTO_CSA_REPORT=<file>` | 逐步 PTO vs vendor 的统计落盘 |
| `PTO_CSA_PROBE=<dir>` | 只读探针，把两处调用点的张量长相落盘 |
| `PTO_CSA_VERIFY=1` | 逐入参按**内容**对拍：用两边各自的约定把 KV 取一遍比结果 |
| `PTO_CSA_DUMP=<dir>` | 第一次替换那一步的全部入参 + vendor 输出落盘，供 `analyze_dump.py` 归因 |
| `PTO_CSA_MODE` | `program`（默认）或 `kernel`。见 §12 |
| `PTO_CSA_CMP_COLS` | kernel 模式下压缩块表的固定列数，默认 128。见 §12 |
| `PTO_CSA_PLATFORM` | PyPTO 平台，默认 `a2a3` |
| `VLLM_ASCEND_PROFILER_LEVEL` | `Level0/1/2`，默认 `Level1`；Level2 额外带 AICPU 与通信算子 |
| `PROFILE=1` + `PROFILE_TOKENS` / `PROFILE_WARMUP` | 走 profiling 客户端，录几个 decode token |

全不设时 `pto_csa.py` 对 vLLM 没有任何影响 —— 这也是运行 A 的做法。

---

## 9. 数值不对时怎么定位

先带 `PTO_CSA_DUMP=<dir>` 跑一轮运行 B，然后（**不占卡**）：

```bash
PL=/data/sunkaixuan/sunkaixuan_subdir/all_libs/pypto-lib-csa-20260917
PP=/data/sunkaixuan/yj_subdir/kernel-csa-feat-pypto
V=/data/sunkaixuan/sunkaixuan_subdir/all_servings/.venv-dsv4-vllm310
cd /data/sunkaixuan/codex_sh/csa_b_tier_20260917
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
PYTHONPATH=$PL:$PL/models/deepseek_v4_flash_mtp:$PP/python:$PP/runtime:$PP/runtime/python \
  $V/bin/python analyze_dump.py --dump <PTO_CSA_DUMP 目录>
```

输出把一步切成四段：

| 对比项 | 说明 |
| --- | --- |
| `golden(翻译后入参)` vs `PTO 输出` | ≈0 说明 kernel 照着算了，错在翻译层 |
| `golden stage-1` vs `vendor stage-1` | 只有注意力，看窗口 / 压缩槽的翻译对不对 |
| `golden后半段(喂 vendor stage-1)` vs `vendor 输出` | 看逆 RoPE + o_proj 的建模对不对 |
| `golden(翻译后入参)` vs `vendor 输出` | 端到端还差多少 |

改过 `pto_csa.py` 想先自查（不需要模型权重，约 1 分钟）：

```bash
cd /data/sunkaixuan/codex_sh/csa_b_tier_20260917 && bash run_adapter_check.sh
```

判据：`within_tol: true`、`max_abs_error ≈ 1.5e-5`、`runner_stats.fallbacks` 为空。

---

## 10. 已知限制

- **TP 只能是 1**，decode 批次要能摊进 kernel 编译期固定的 `(B=4, S=2)`；凑不齐会回退 vendor。
- **S>1（MTP / 投机解码）暂不支持**：两张块表是按**序列**给的，适配器按 **token** 索引，
  S=1 时两者相等，S>1 会错位。目前靠 `obt.shape[0] < n` 那条回退挡住。
- **性能不是这一版的目标**（138 ms/step）。PyPTO 的 dispatch 只接受 host 张量或它自己
  Worker 分配的 `DeviceTensor`，吃不了 vLLM 的 KV cache 指针，所以每步要把用到的页收拢成
  小 cache 再走一轮 D2H+H2D。trace 里 `simpler_aicpu_register_callable_*`
  **每次 dispatch 都重新注册**，6.9 ms/步，是最大的一块；真正的 CSA 计算只有 5.3 ms / 8 步。
- **那 0.8% 残差是格式差异，不是缺陷**：官方 checkpoint 的 `wo_b` 是 BF16，而 pypto-lib 下
  三个 CSA 变体的 kernel 签名全都写死 `INT8 + per-channel scale`，只能在 setup 时量化一次
  （相对误差 3.9e-3）；RoPE 表 vLLM 用 FP32 而 kernel 签名要 BF16（差 1.8e-3，正好一个
  BF16 ULP）。分段归因证实残差几乎全在 o_proj 段，注意力段只有 mean 3.0e-4。
- **aclgraph 下替换不生效**，两种模式都是。开 aclgraph 后 vLLM 每步重放捕获好的图，
  `AscendDSAImpl.forward` 整轮只被进入个位数次，且实测 `is_current_stream_capturing()`
  全为 False —— 替换一次都没在捕获期执行，所以录进图里的是 vendor 路径。判据是输出
  token 指纹：`aclgraph 原生` 与 `aclgraph + PTO` 完全相同。见 §12。

---

## 11. 三个反复踩到的坑（给后来者）

**一 · 块表的有效列数不能当页容量的依据。** vLLM 给压缩缓存分配的块数和滑窗缓存一样多
（4098 token 各 33 列），据此推「一页 32 槽」是错的。实测只有前 8 页被写过
（8×128 = 1024 = 4098/4），一页确实是 **128 个压缩槽**。**判断页容量要看写没写，
不是看分了几块** —— 扫每页非零行数即可。

**二 · RoPE 表两边排布不同。** vLLM 走 `rotary_mode="interleave"`，表是 `[c0,c0,c1,c1,…]`，
64 个位置上只有 32 个不同频率；pypto-lib 的是「前半 32 个真值 + 后半复制」，golden 只读
`[:, :HALF_ROPE]`。**直接截前 32 个只会拿到前 16 个频率各两份**，要按步长 2 去重。

**三 · fixture 的 `init_*` 返回 fp32，而 spec 声明的是 bf16。** 自己造张量时要调
`sp.create_tensor()` 套上 spec 的 dtype，否则 vendor 算子直接拒：
`Io input dtype or format is not supported`。

---

## 12. kernel 模式（`PTO_CSA_MODE=kernel`）

### 两种模式的区别

| | `program`（默认） | `kernel` |
| --- | --- | --- |
| 调用方式 | `op.compile(...)(...)`，走 `CompiledProgram` | 直接调 `@pl.jit` 对象 |
| 张量 | 只吃 host 张量，PyPTO 自己做 H2D/D2H | 直接吃 vLLM 的 NPU 张量，kernel 借用 |
| KV cache | 每步把用到的页收拢成小 cache 再上传 | 整份 cache 原样交过去，用原始物理索引 |
| 前置 | 无 | 每进程一次 `pypto.torch.init()`，且在 capture 之外 |

kernel 模式要求 PyPTO 构建时开了 `-DPYPTO_BUILD_TORCH_NPU=ON`（**默认 OFF**），
且 `_torch_npu` 与 `pypto_core` 必须按**同一套 torch** 一起重编 —— 只重编一个会留下
ABI 不一致，症状是首次 invoke 段错误，栈停在构造 `at::Tensor` 时给 `TensorImpl` 加引用计数。

### 数值

kernel 模式与 program 模式**逐 token 一致**（输出指纹相同），整网相对误差同为 0.77%。

### kernel 路径上不能有 host 读回

捕获期 CPU 不等 NPU 算完，读回的值没有意义，而且会被当成常量固定进图里，之后每步重放
都用这个过期值 —— 不报错，结果悄悄错。更糟的是这类值往往还决定张量形状。

所以 `run_decode` 里两条路径是**完全分开**的：`bool(ok.any())`、`int(pblk[ok].max())`、
`torch.unique(...)`、布尔掩码索引、`int(idx_t.max().item())` 这些全部只出现在 `else`
（program）分支。kernel 分支用固定宽度块表（`PTO_CSA_CMP_COLS`）绕开 `max_slot` 那次读回，
越界改成在设备上 `clamp` / 置 -1。**改这段代码时不要把校验挪回共用位置。**

### 已验证 / 未通过

`tools/pto_csa/kernel/run_kernel_mode.sh` 按 handoff 的三项判据验真实 CSA kernel：

```
eager.max_abs                                  1.53e-5   < 1e-2  ✅
replay.max_abs                                 1.53e-5   < 1e-2  ✅
replay_follows_input.max_abs_vs_new_golden     1.53e-5   < 1e-2  ✅
replay_follows_input.differs_from_first_replay true              ✅
```

**独立脚本里 kernel 能被 NPUGraph 捕获、重放正确、改输入后输出跟着变。**

但**接进 vLLM 之后，替换进不了 vLLM 捕获的图**（见 §10）。断点不在 kernel 的捕获能力，
而在接入点：钩子在 `AscendDSAImpl.forward` 这段 Python 里，而 vLLM 捕获时不走到这里。
要打通得把调用改造成"固定缓冲区 + 原地更新"的形态（参照 PyPTO 仓内
`examples/runtime/torch_kernel_capture.py` 的 `step()`），再设法让它落在 vLLM 捕获的执行路径上。
