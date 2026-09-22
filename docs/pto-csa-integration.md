# PyPTO CSA 算子接入 vLLM 的对接说明

> **Run 062 更新（2026-09-22）**：本文第 2～8 节记录的是旧 46 参数转换层及其性能，
> 仅作为问题背景保留，不再描述当前工作树。当前实现已切换为 lib 提供的 40 参数
> vLLM-native attention-only ABI：主 state、raw KV、compressed KV 分别直接绑定
> `[N,16,2048] FP32`、`[N,128,1,512] BF16`、`[N,128,1,512] BF16`
> 三个连续 view；主 state 与 compressed KV 共用物理 allocation，raw KV 独立；
> inner state/index key/FP16 scale 直接绑定同一份
> `[N,130,128] INT8` 物理页；adapter 不再构造私有 state ring、repage cache 或五类
> slot mapping。本文将在真实 eager/ACLGraph/performance A/B 完成后整体改写；在此之前不要复制下文的
> 46 参数 `ARG_ORDER` 或旧性能数字作为新接口说明。

### 当前工作树：直接绑定真实 token 与常驻 RoPE

- 40 个参数的数量和顺序不变；原 compiled cache 必须重建。
- TP1 的容量为 B=64、S=6；实际输入按 request-major 排列，`T = runtime_batch * runtime_seq`。
  S=1 不再扩成六份；S=6 提交六个真实 token。请求之间使用相同的 `runtime_seq`，
  padding/无效尾行由 `token_valid` 表达，不支持无映射的变长 query 拼接。
- `x_normed` 和 `attn_out` 为实际 `[T,4096] BF16` 连续 view；直接写调用方 output，
  不再在返回后 index-select。
- `position_ids` 为原生连续 `[T] INT64`，入口的既有 RoPE task 内转换为内部 INT32。
- `freqs_cos/sin`、`cmp_freqs_cos/sin` 改为常驻 `[rope_rows,64] FP32` 的 interleaved 全表，
  直接 view vLLM `_ROPE_STATE.static_cache`，不新建全表。RoPE 行轴与 T 独立。
  普通 RoPE 读取 `position`；ratio4 边界读取 `position + 1 - 4`。
  adapter 不再进行 BF16 转换、半频率 concat、boundary Cumsum 或 compact-row gather。
- 原有 `csa_vllm_rope_interleave` task 一次生成消费者共用的 token-local 行；
  QKV 和 sparse inverse RoPE 不再各自启动一次转换 task。
- 仍有动态 `token_valid` 的构造。这里只移除了重复数据/形状转换，不声称所有准备开销归零。
- 验证覆盖 PyPTO 算子、host payload、kernel numerical golden，以及下文的单层
  `impl.forward()` A/B；不代表完整 vLLM serving 已验收。
- 当前 host payload、S1 回归，以及 TP1/S6/128K 的 B=4/8/16/24/32/40 单卡矩阵
  均已通过。S1 的单列 fillpad UB 越界已修复并通过设备校验。
- 已补 cold-start、全 inactive、部分有效尾行、非零 compressor/indexer，以及设备上
  main state/compressed KV 共用一份 allocation 的按字节校验。B4/S1、B4/S6、B40/S6
  的128K trace 均通过输出/cache/state检查，已下载到本地。
- 测试使用 PyPTO 机器的 dsj-pypto-dev、PTOAS0.63；测试脚本显式接收task-submit
  分配的设备号。不传卡号的旧并发提交结果不计入本次验收。
  详细任务号、产物和瓶颈数据见 Run062 §15.8–15.9。
- 单步kernel golden使用合成输入；不代表真实请求激活回放、多轮持久cache或
  旧/新版本同负载性能A/B已完成；截图旧数字不能直接作为新版本收益基线。

### 单层 native / PTO A/B 状态（2026-09-22）

使用真实checkpoint的第2层attention权重、固定合成hidden states/history cache，
只调用单层CSA，不加载整模型、不包含外部HC-pre/输入RMS/HC-post/MoE。
native为 `AscendDSAImpl.forward()`；PTO为 `build_args()` 加注册的attention kernel。

- B4/S1/128、B4/S6/128、B4/S6/128K均能执行；输出尾部哨兵和position输入不变检查通过。
- **native/PTO精度比较尚未通过**：output相对L2分别约1.5264%、1.5391%、1.8601%，
  index key也有差异。不能将kernel自身golden通过表述成真实native结果完全一致。
- TP1 output projection的最后一个不足8行的tile已用 `pl.set_validshape` 限制写回；
  修复了S1/B4时越界覆盖相邻buffer的问题，但未消除上述数值差异。
- 按性能专项口径，另测B4/S6/128K；`start_pos=131066`，最后位置131071，
  visible length为131072。独立ACLGraph、同一张卡、6轮×50次重放，报告轮均值的中位数。
  首次编译、权重转换、metadata构造、cache恢复、graph capture及profiler均不计入。

| 路径 | 单次24-query forward（μs） |
| --- | ---: |
| native `impl.forward()` | 762.869 |
| PTO完整调用（含adapter） | 1861.026 |
| PTO参数预绑定、仅kernel | 1847.819 |
| 仅adapter准备，独立图 | 39.268 |

该配置下PTO完整时延约为native的2.44倍。独立分项各有图下发开销，不能直接相加。
主要差距在kernel/runtime路径，而非输入准备；尚未通过该trace进一步隔离PTO内部各阶段。
native的 `multistream_dsa_preprocess` 和 `multistream_dsv4_dsa_overlap` 均关闭，
不宣称这是native所有调优配置中的最优结果。此性能记录不构成精度验收。

上述单卡kernel gate与单层A/B使用本地实验脚本，按提交范围要求未纳入本PR；
脚本、命令和产物保留在Run062任务记录中。仓内保留CPU接口契约测试。
性能环境使用Python3.11、Torch2.10.0 CPU、torch-npu2.10.0、vLLM0.20.2、
当前vLLM Ascend源码及新编译的native算子、CANN9.0.0、PTOAS0.63。
多步状态、真实服务请求以及完整数值闭环仍待验证。

这条分支把 DeepSeek-V4 某一层的 `attention.forward` 整段换成 PyPTO 写的 CSA 算子,
在 vLLM 的真实推理服务路径上跑。本文给 pypto-lib 侧同事看,不假设读者了解 vLLM 这边。

内容分两部分:一是**已经确定的事实**(版本、代码位置、参数、维度关系、性能归因),
二是**仍待补的信息**(几处只能从活跃张量上读的布局细节)。每条都标了来源。

## 1. 版本

| 仓 | commit | 分支 | 链接 |
| --- | --- | --- | --- |
| vLLM | `bc150f50299199599673614f80d12a196f377655` | detached | [vllm-project/vllm@bc150f50](https://github.com/vllm-project/vllm/commit/bc150f50299199599673614f80d12a196f377655) |
| vLLM Ascend 上游基线 | `367b8e62da799870a7476ce34f5f7658589a8aad` | v0.20.2rc1 | [vllm-project/vllm-ascend@367b8e62](https://github.com/vllm-project/vllm-ascend/commit/367b8e62da799870a7476ce34f5f7658589a8aad) |
| **本分支(CSA 适配)** | `213933829aededf9e75359fa1de6e14426b81c5f` | `feat/csa-attn-cut-20260920` | 本 PR |
| pypto | `6b49cfd58de65f8a325339a30d6fa45e1973c237` | `feat/kernel-mode-integration-test` | [hw-native-sys/pypto@6b49cfd5](https://github.com/hw-native-sys/pypto/commit/6b49cfd58de65f8a325339a30d6fa45e1973c237) |
| simpler | `17ea300256e2a6db5af397ce619a7d480b595d80` | `feat/kernel-mode-integration-test`(tip) | [hw-native-sys/simpler@17ea3002](https://github.com/hw-native-sys/simpler/commit/17ea300256e2a6db5af397ce619a7d480b595d80) |
| pypto-lib | `675f027f1bf9ce8f79740f0b2ed042b0c4b2c8b3` | — | [hw-native-sys/pypto-lib@675f027f](https://github.com/hw-native-sys/pypto-lib/commit/675f027f1bf9ce8f79740f0b2ed042b0c4b2c8b3) |

本分支相对上游基线领先 18 个提交。pypto 与 simpler 必须**按 commit 取**:
pypto 的本地分支名是 `feat/kernel-mode-20260920`,但提交实际属于
`feat/kernel-mode-integration-test`;simpler 停在 detached HEAD。

**复现时的一个坑**:开发机上默认 `pip` 解析到的 pypto 和 simpler 是另外两个目录,
不是上表这套。我们靠启动脚本的 `PYTHONPATH` 覆盖。不设 `PYTHONPATH` 会拿到别的版本,
症状通常是 `Kernel ABI requires Simpler <hash>; native binding is <other>` —— 那是
pypto 与 simpler 原生 binding 的指纹不匹配,与 pypto-lib 无关。

## 2. 代码位置

| 文件 | 行数 | 作用 |
| --- | ---: | --- |
| `vllm_ascend/attention/pto_attn.py` | 1018 | 转换层:把 vLLM 的运行时状态翻译成算子的 46 个参数 |
| `vllm_ascend/ops/dsa.py` | 309 | 钩子:在 `dsa_forward` 里 `_build_kv_cache` 之后、`impl.forward` 之前分流 |
| `vllm_ascend/attention/pto_kernels/dspark/` | 11k+ | 从 pypto-lib 剥离的算子副本 |

剥离副本相对 pypto-lib 原版的真实差异是 **+222 / −54 行(34 处)**,绝大部分是
绝对 import 改相对。实质分叉只有一处:新增 `decode_csa_attn_tp1`(46 参数,只做 attention)
与其 `@pl.jit` 入口 `decode_csa_attn_tp1_test`,原来的 `decode_csa_tp1` 改成
`hc_pre_norm → attn → hc_post` 的薄包装以保持 51 参数契约。

> 用 `diff` 直接比会报「整个文件都变了」,那是行尾差异造成的假象,先 `tr -d '\r'`。

这两个入口**在 pypto-lib 里不存在**,是本分支 fork 出来的。

## 3. 运行配置

```text
vLLM        tensor_parallel_size=1  data_parallel_size=1  pipeline_parallel_size=1
            单个 device (ASCEND_RT_VISIBLE_DEVICES=15)
            dtype=bfloat16  quantization=ascend
            max_model_len=8704  max_num_batched_tokens=4096
            gpu-memory-utilization=0.60
            cudagraph_mode=FULL_DECODE_ONLY  cudagraph_capture_sizes=[1,2,4]
负载        模型 official-l3（DeepSeek-V4 截成 3 层，MTP 层已去除）
            batch=4  prompt=1024 token  max_tokens=32
算子        PTO_ATTN_TP=4   PTO_ATTN_SEQ=1
```

`cudagraph_capture_sizes` 每项必须 ≤ 算子的 `B`,超出的 descriptor 会被转换层
decline 并回落原生路径。

## 4. 注册入口与参数绑定

```python
# vllm_ascend/attention/pto_attn.py
def _registered():
    global _OP
    if _OP is None:
        from pypto.torch import init, register
        kcsa, _ = kernel()
        init()
        _OP = register(kcsa.decode_csa_attn_tp1_test, "pypto_csa::attention_csa")
    return _OP
```

绑定顺序(`ARG_ORDER`,共 46 项):

```python
ARG_ORDER = (
    "x_normed",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "cmp_freqs_cos", "cmp_freqs_sin",
    "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
    "compress_state", "compress_state_block_table",
    "idx_wq_b", "idx_wq_b_scale", "weights_proj", "hadamard_idx",
    "inner_wkv", "inner_wgate", "inner_ape", "inner_norm_w",
    "inner_compress_state", "inner_compress_state_block_table",
    "kv_cache", "cmp_kv", "cmp_block_table",
    "idx_kv_cache", "idx_kv_scale", "idx_block_table",
    "ori_slot_mapping", "window_swa_indices",
    "cmp_slot_mapping", "idx_slot_mapping",
    "state_slot_mapping", "inner_state_slot_mapping",
    "position_ids", "kv_seq_lens", "attn_sink",
    "wo_a", "wo_b", "wo_b_scale",
    "attn_out",
)
```

参数值在 `build_args()` 里组装,钩子在 `ops/dsa.py::dsa_forward` 调用
`pto_attn.substitute()`;返回 `True` 表示已接管,不再走 `impl.forward`。

## 5. 维度关系

看到 `B=16` 与服务 `bs=4` 并存不必困惑,两者不是一回事。

`decode_csa.py:132` 是 `B = DECODE_BATCH // TP_SIZE`,`DECODE_BATCH=64`,
所以 `PTO_ATTN_TP=4` 得到 `B=16`。这是**算子实例的批容量**,不是实际批。
4 个真实请求摊进 `T = B × S` 的矩形,padding 车道置 `-1`。

**注意 `TP_SIZE=4` 在本部署里是容量选择器,不是四路并行。** 我们在单 device 上跑,
注册的入口用全局权重常量(`O_GROUPS=8`),没有跨 rank 通信;那些
`pld.DistributedTensor` 参数属于别的入口。用 `--tp 4` 的原因是 `--tp 1` 时
`T=512` 会让 ring heap 分配失败(`FATAL: Task Allocator Deadlock - Heap Exhausted`),
而 kernel 模式没有 ring 尺寸的配置入口。代价是这个实例最多接 16 个并发请求。

本实例化下的编译期常量:

```text
B=16  S=8  T=T_PAD=128  BLOCK_SIZE=32  HEAD_DIM=512  IDX_HEAD_DIM=128
WIN=128  COMPRESS_RATIO=4
MAIN_STATE_BLOCK_SIZE=2   MAIN_STATE_MAX_BLOCKS=8   MAIN_STATE_DIM=2048
MAIN_STATE_STORAGE_LEN=16
INNER_STATE_BLOCK_SIZE=2  INNER_STATE_MAX_BLOCKS=8  INNER_STATE_DIM=512
CMP_MAX_BLOCKS=IDX_MAX_BLOCKS=8192
```

## 6. 入参格式与 vLLM 实际布局的差异

vLLM 用 `as_strided` 在一块 padded 缓冲上切出多份逻辑 cache,所以其中四份是
非连续视图;而 PyPTO 绑定层拒收非连续张量(`pypto/torch/interop.py:141` 直接
`raise ValueError("... requires a contiguous strided tensor; no copy is made")`)。

| 错配 | 算子这边 | vLLM 那边 | 逼出的代价 |
| --- | --- | --- | --- |
| 压缩器 state 的组织 | 每请求 16 行私有环 + 自带块表(页 2 行 × 8 页) | 按自己的分配器分页,页 8 行 | 每步现建环 + 写回 |
| indexer key / scale | 两个独立张量,scale 要 FP32 | 同一个 16640 字节页,key 占前 128 行、scale 在尾部 | key 非连续 + 类型转换 |
| 页大小 | 32 行 | 128 行 | **已零开销解决**(4 倍,块表列号换算) |
| 槽号空间 | 按算子自己的页编号 | 按 vLLM 的 `(block, intra)` | 每步翻译 |

页大小那条说明这类差异**可以被消除**:只要是整数倍关系,换算块表即可,不需要搬数据。
滑窗 KV 与压缩 KV 走的就是这条路,实测搬运开销为 0。

## 7. 性能现状

aclgraph `FULL_DECODE_ONLY` 下 8 个 decode 步的实测,每步设备耗时:

| 项 | us/步 | 说明 |
| --- | ---: | --- |
| 搬运(去 stride) | 20,811 | 转换层 |
| 索引换算 | 2,070 | 转换层,183 次算子启动/步 |
| **CSA 算子本体** | **976** | AICPU 派发段与 AICore 执行段的并集 |
| 图内其余模型算子 | 3,031 | 两轮都有 |
| 图外 eager 通道 | 1,507 | metadata |
| 合计 | 28,395 | 原生对照 4,737 |

算子耗时**不能把 AICPU 段与 AICore 段相加**:后者在全部 8 步里都严格嵌套于前者,
相加会把 976 us 说成 1924 us。

**转换层占了 80%,算子本体占 3.4%。**

搬运那 20.8 ms 技术上可以在转换层内修掉(改成不触发去 stride),修完约 7.2 ms/步。
但即使做完,索引换算里仍有约 1.0~1.5 ms/步**消不掉** —— 两套寻址约定之间必须有人翻译,
且必须做在图内,因为图重放时 Python 不执行,翻译结果没法预先算好塞进去。

**我们决定不做这部分优化。** 两个原因:对齐版本出来之后这些代码全都要删掉,现在投入是废功;
而且做完离原生仍有距离。现有转换层功能上够用(能跑通、能进图),性能等对齐版本解决。

希望 lib 侧提供的是一个**入参与 vLLM 实际布局对齐**的 CSA 算子版本,
让转换层只做参数传递 —— 不搬移数据、不改数据格式、不做索引换算。
那样每步可以从 28.4 ms 直接降到约 5.6 ms,对原生的 4.7 ms 约 1.19 倍,
剩下的差距只剩算子本身的计算时间。

## 8. 布局与语义的详细答复

六份 cache 的布局、五类 slot mapping 的语义、维度关系等详细问答见
[pto-csa-layout-qa.md](pto-csa-layout-qa.md)。三条最要紧的结论:

- **state block table 记的是整段历史的绝对逻辑页号**,`[B, 1088]` INT32,
  列 j 对应 position ∈ [8j, 8j+8)。ring 语义只存在于算子这一侧,vLLM 侧不 wrap。
- **inner state 与 indexer key 共享同一块 allocation**,inner state 页尾那
  256 字节是 indexer scale,**不是 padding,算子绝不能往那里写**。
- **indexer 的三份 cache 只存在于三层里的一层**(`ops/dsa.py:284` 按
  `compress_ratio == 4` 门控),所有 indexer 的实测只对 layer 2 成立。

仍待补的是几处只能从活跃张量上读的:各张量的 `data_ptr` 关系、各块表的实际示例值,
以及一份覆盖边界的 fixture(8 行逻辑页边界、16 行环回绕、128 行 KV 页边界、
indexer 页切换、inactive token、多请求不同物理页)。需要再跑一轮带数值转储的运行。

## 9. 复现

```bash
export PTO_ATTN_REPLACE=1 PTO_ATTN_SEQ=1 PTO_ATTN_TP=4 PYPTO_CACHE=1
export SERVE_EXTRA='--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
```

`SERVE_EXTRA` 的 JSON **不能有空格**:启动脚本按空格分词。

判据不要用客户端的 `output_token_fingerprints`,它是对每请求哈希取
`sorted({...})`,丢掉了请求身份,同配置多轮会给出不同结果。可用的判据是三条同时成立:
捕获进度跑满且服务健康;每个 capture descriptor 有两次替换事件、第二次带
`capturing=True`;重放期每步延迟在 PTO 量级而非原生量级。再加 profile 里
`simpler_aicpu_kernel_exec` 每步出现一次作为直接证据。
