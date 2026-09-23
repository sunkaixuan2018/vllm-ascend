# PTO CSA 接入的脚手架

`vllm_ascend/attention/pto_csa.py` 是**产品代码**——跑在 vLLM 进程里，把 DeepSeek-V4
ratio-4 层 decode 路径的 CSA 换成 pypto-lib 的 PTO 算子。这个目录是**让它可被验证**的外围：
建环境、占卡起服务、采 trace、以及不依赖模型权重的离线自查。

完整复现步骤、版本坐标、已知限制见 [REPRODUCE.md](REPRODUCE.md)。

## 先跑这个

```bash
tools/pto_csa/kernel/run_adapter_check.sh
```

**不要模型权重、不起服务、约 1 分钟。** 它用 pypto-lib 自己的 fixture 反造出 vLLM 侧的
数据约定，过一遍适配器，再对 pypto-lib 的 golden。适配器里那套索引换算（压缩页重映射、
窗口物理槽反推、wo_a 转置、wo_b 再量化、decode 批次摊到编译期固定的 `(B, S)`）全靠它守着。
改完 `pto_csa.py` 先跑它，不要直接去起整网。

## 三组脚本

| | 用途 |
| --- | --- |
| `serving/` | 建 venv、拉权重、占卡起 vLLM、压测、采 trace |
| `kernel/` | 只验 CSA 本身：离线对拍、共存门禁、kernel 模式捕获验证、失败归因 |
| `stack/` | 从零建一套**不依赖他人目录**的栈：自带 CPython、从上游自建 PyPTO/simpler |

`stack/build_own.sh` 是 `serving/build_env.sh` 的替代路线。（目录不叫 `env/` 是因为仓库的 `.gitignore` 用 `env/` 挡 virtualenv。）后者的基础解释器和 ATB 都借自
别人的目录，前者自带 CPython（python-build-standalone，含 ssl）、把 ATB 拷进自己目录、
从上游 clone 并自建 PyPTO 与它的 simpler 子模块。kernel 模式必须走这条 —— 它要求
`-DPYPTO_BUILD_TORCH_NPU=ON`，而那个扩展必须与实际运行的 torch 版本同套重编。

`serving/` 里的入口是 `run_dsv4_mtp_vllm.sh`（挑端口 → `task-submit` 拿设备锁 → 把整轮交给
`dsv4_case_inner.sh`）。卡是共用的，**不要绕过 `task-submit` 直接起服务**。

`kernel/run_kernel_mode.sh` 验 kernel 模式：真实 CSA kernel 能否被 `NPUGraph` 捕获、
重放是否正确、改输入后输出是否跟着变。第三条最容易漏 —— 捕获若把值烘死在图里，重放会
"成功"但结果不更新，接到服务里表现为输出静止不动，很难定位。

## 整网 DeepSeek 推理时采 CSA 泳道图

支持 eager CSA 调用和 ACLGraph replay 两种采集入口：
`pto_csa.py` 的实际下发处，以及标准 ACLGraph wrapper / v2 full-graph manager 的 replay 外围。
使用已构建好 torch_npu adapter 的 PyPTO `feat/kernel-mode-integration-test`
分支及其配套 Simpler runtime，在现有整网启动配置上加：

```bash
export PTO_CSA=1
export PTO_CSA_MODE=graph
export VLLM_ASCEND_PTO_CSA_SWIMLANE_LEVEL=4
export VLLM_ASCEND_PTO_CSA_SWIMLANE_DIR="$PWD/csa_swimlane"
export VLLM_ASCEND_PTO_CSA_SWIMLANE_MAX_CAPTURES=8
tools/pto_csa/serving/run_dsv4_mtp_vllm.sh
```

保留原有 `MODEL`、`PYPTO_ROOT`（指向上述分支）、`PYPTO_LIB_ROOT`、`DSV4_VENV`
及设备配置。直接使用 `vllm serve` 时同样设置这些开关。采集 replay 时不要传
`--enforce-eager`，并确保已有 `SERVE_EXTRA` 中也没有它。`PTO_CSA_MODE=kernel`
的直接 JIT 路径也支持采集；`graph` 使用常驻缓冲区和注册算子。
要采集 eager 调用，则在启动参数中加入 `--enforce-eager`。
启动器会将开关传入 task-submit，具体是否使用图仍由 vLLM 的图配置和 batch 匹配决定。
原有 CSA 适配器的模型、batch/head 形状限制仍适用。

| 开关 | 默认值 | 含义 |
| --- | --- | --- |
| `VLLM_ASCEND_PTO_CSA_SWIMLANE_LEVEL` | `0` | `0` 关闭；`1..4` 指定采集级别，并采集依赖边 |
| `VLLM_ASCEND_PTO_CSA_SWIMLANE_DIR` | 空 | 开启时必填的产物根目录 |
| `VLLM_ASCEND_PTO_CSA_SWIMLANE_MAX_CAPTURES` | `1` | 每个 worker 最多打开多少个 eager/replay 采集窗口，必须为正整数 |

强制 eager 时，每个层和入参形状组合的第一次真实调用用于编译/热身，后续真实调用才
包在 `begin_dfx()` / `end_dfx()` 窗口中。允许图模式时，eager warmup 和 capture 不消耗
采集预算，窗口放在实际 `replay()` 外围，一次 replay 可以包含多个 CSA 调用。
不会额外重跑请求；达到上限后恢复普通下发。图模式下回退的 eager 调用也不消耗
预算，因此始终没有 replay 的运行不会生成图泳道图。
若引擎在正式请求前也执行 replay，该次 replay 同样计入预算；产物不自动代表稳定态性能。

图中必须真正包含 PTO CSA 节点。此开关只采集，不会把没被 vLLM 捕获的替换逻辑自动
加入图中。若重放没有记录到 CSA 任务，保留原始诊断与 `capture.json`，其 `status`
为 `no_csa_tasks`，日志明确提示，不生成空的 `merged_swimlane.json`。空窗口也占用
一次预算。只有已有 CSA Worker 的进程才会采集，replay 钩子不会临时初始化 PyPTO。

每个 worker 使用独立的 `worker_<pid>_device<id>_<unique>/` 目录。
首次采集直接写在其中，后续使用 `window_1/`、`window_2/` 等子目录，内容为：

- `merged_swimlane.json`：用 Perfetto 打开，查看 CSA 内部任务与依赖边。
- `chip_swimlane_records.json`、`deps.json`：原始计时记录和依赖图。
- `name_map.json`：从实际 warmup 的 kernel 编译产物读取的函数名映射。
- `capture.json`：执行模式和进程；eager 包含模型层、设备、下发序号、入参形状，
  replay 包含对应的图描述。

依赖 ID 的转换由 runtime 的 `simpler_setup.tools.swimlane_converter` 负责，
需包含 `hw-native-sys/simpler#2393`（`17ea3002`）的整数 ID 修复；vLLM 不再改写转换结果。
泳道图应显示实际 kernel 函数名，而不是 `func_0_a` 等占位名称。

日志中的 `[pto-csa] eager swimlane -> ...` 或 `graph_replay swimlane -> ...` 给出文件路径。
这会同步当前流并收集依赖图，增加推理开销；采集期间的整网 TPOT/吞吐不能作为
正常推理性能。它展示整网执行中的 **PTO CSA 内部任务**，其他 torch/vendor 算子
仍需 torch profiler。采集边界必须在图捕获之外，并在同一当前流上配对；内层图在
外层 capture 中被 replay 时不打开窗口。转换器对窗口内各次下发使用相同的依赖拓扑，
含不同 CSA 拓扑的图不能据此解读跨下发的依赖关系。

## 出了问题从哪查

`kernel/analyze_dump.py` 把 live 那一步的入参重放一遍，切成三段分别归因：

```text
golden(翻译后入参) vs PTO 输出   -> kernel 有没有照着算
golden stage-1 vs vendor stage-1 -> 窗口/压缩槽的翻译对不对（不含投影）
golden 后半段 vs vendor 输出     -> 逆 RoPE + o_proj 的建模对不对
```

服务端只会给你一句"输出不对"。这次接入踩到的两处翻译错误（压缩缓存页大小、RoPE 表排布）
就是靠这个切开的——第一段只差 0.0156 说明 kernel 没问题，第二段差 0.39 说明翻译走偏了。

## 开关

| 环境变量 | 作用 |
| --- | --- |
| `PTO_CSA=1` | 打开层级替换。不设时 `pto_csa.py` 对 vLLM 没有任何影响 |
| `PTO_CSA_VERIFY=1` | 每次替换的第一步做一遍逐入参一致性断言 |
| `PTO_CSA_PROBE=<dir>` | 只读探针，把两处调用点的张量长相落盘 |
| `PTO_CSA_DUMP=<dir>` | 把第一次替换用到的全部张量落盘，供 `analyze_dump.py` 重放 |
| `PTO_CSA_REPORT=<file>` | 逐步 PTO vs vendor 对拍与回退统计 |
| `VLLM_ASCEND_PROFILER_LEVEL` | `Level0` / `Level1`（默认）/ `Level2` |
