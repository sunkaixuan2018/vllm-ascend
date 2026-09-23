# PyPTO CSA 接入 —— 布局与语义问卷答复

回应 lib 侧的八项信息需求。每条都标了来源：**【读码】**给出 file:line，**【实测】**给出
artifact 路径。答不出来的列为待补，没有猜。

配置口径：vLLM `tensor_parallel_size=1`，单个 device，bs=4，prompt 1024，max_tokens 32，
`cudagraph_mode=FULL_DECODE_ONLY`，模型 official-l3（DSV4 截 3 层、去 MTP）。
算子 `PTO_ATTN_TP=4`（容量选择器，非四路并行）、`PTO_ATTN_SEQ=1`。
版本见 [pto-csa-integration.md](pto-csa-integration.md)。

## 怎么读这份文档

每项都经过一轮独立复核，复核方的任务是**逐条打开引用去反驳**。结论分三种：

| 标记 | 含义 |
| --- | --- |
| 已复核通过 | 复核方确认每条引用属实 |
| 复核未通过 | 复核方推翻了实质内容，本文已按更正改写，原答复保留在后面供对照 |

## 状态

| 项 | 内容 | 状态 |
| --- | --- | --- |
| 第 1 项 | 准确的代码版本 | 已复核通过 |
| 第 2 项 | 主压缩器 state 的完整布局 | 已复核通过 |
| 第 3 项 | inner compressor state 的完整布局 | **复核未通过，已按更正改写** |
| 第 4 项 | indexer key/scale 的共享存储契约 | **复核未通过，已按更正改写** |
| 第 5+6 项 | raw/cmp KV 分页语义 + 五类 slot mapping | 已复核通过 |
| 第 7 项 | 运行时 batch/token 维度关系 | 已复核通过 |
| 第 8 项 | 可复现 fixture | **复核未通过，已按更正改写** |

## 先看这三条

对方点名说第 2、3、4、6 项必须确认后才能改生产代码。目前：

**一、state block table 记的是绝对历史页，不是固定环形槽。**【第 2 项，已复核通过】
shape `[B, 1088]` INT32，列 j 对应 position ∈ [8j, 8j+8)。ring 语义**只存在于算子这一侧**，
vLLM 侧没有 wrap，它靠 sliding-window allocator 回收物理页。这就是对方担心「答错会在多步
decode 中静默写坏状态」的那条，现在有答案。

**二、inner state 与 indexer key 共享同一块 allocation，页尾 256 字节是 indexer scale。**
【第 3 项，复核更正】不是两块独立缓冲。`patch_kv_cache_utils.py:232-240` 按 page_size 分桶，
每桶只发一个 KVCacheTensor。inner-state 的视图覆盖每页 `[0, 16384)`，**从不寻址最后 256 字节**；
那 256 字节属于谁取决于该 block id 当前归哪个 group 所有。
**算子代码绝不能往那里写。**

**三、indexer key/scale 契约只存在于三层里的一层。**【第 4 项，复核更正】
`ops/dsa.py:284` 用 `if self.compress_ratio == 4` 门控 indexer 的三份 cache；
official-l3 里 layer 0/1 的 `compress_ratio=0`，它们的 `kv_cache[2..5]` 全是 null。
所有关于 indexer 的实测都只对 **layer 2** 成立。


---

# 第 1 项 · 准确的代码版本

> 复核状态：已复核通过

#### 问题 1 — 准确的代码版本（针对 fdoprof 那两次 bs=4 / prompt1024 / device15 的运行）

一句话结论：这套栈的"版本身份"不能只给 5 个 commit 就算交付——**vllm-ascend 与 pypto 两棵树都是脏的**，kernel 源码是**内联进 vllm-ascend 的 pypto-lib 副本**（不是 $LIB），而 `pypto` 这个包在测试用的 venv 里**根本没装**，只靠启动脚本的 PYTHONPATH 解析。下面逐项给锚点。

---

### a) vLLM ($VL = /data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-v0.20.2)

是 **git checkout（浅克隆，`.git/shallow` 存在），不是 release tarball**，但处于 **detached HEAD（no branch）**。

| 项 | 值 | 来源 |
| --- | --- | --- |
| commit | `bc150f50299199599673614f80d12a196f377655` | 实测 `git -C $VL rev-parse HEAD` |
| 分支 | 无（`* (no branch)`；`branch -a --contains HEAD` 只回 `(no branch)`） | 实测 |
| tag | HEAD 正好落在 **`v0.20.2`**（`git describe --tags` = `v0.20.2`，`tag --points-at HEAD` = `v0.20.2`） | 实测 |
| commit 主题/日期 | `[CI] Automate Docker Hub release image publishing (#40415)`，2026-05-07 18:25:36 -0700，作者 Kevin H. Luu | 实测 |
| remote | `origin = https://github.com/vllm-project/vllm.git` | 实测 |
| 工作区 | **干净**（`status --porcelain` 空） | 实测 |
| 自报版本 | `$VL/vllm/_version.py`:22 `__version__ = version = '0.20.2'`；:26 `__commit_id__ = 'gbc150f502'` | 读码 |
| 安装身份 | pip 里是 **`vllm 0.20.2+empty`**，editable，指向 `$VL`；venv 里有 `vllm-0.20.2+empty.dist-info` 与 `__editable__.vllm-0.20.2+empty.pth` | 实测 |

注意 `+empty` 后缀：这份 vLLM 是**纯 Python editable 安装**（未编原生扩展），所以 dist 版本号带 `+empty`，运行时读的是 `$VL` 源码本身。

### b) vLLM Ascend ($V)

| 项 | 值 | 来源 |
| --- | --- | --- |
| commit (HEAD) | `213933829aededf9e75359fa1de6e14426b81c5f` | 实测 |
| 分支 | **`feat/csa-attn-cut-20260920`**（本地分支） | 实测 |
| HEAD 主题/日期 | `Fix: let the substitution be recorded into an ACLGraph`，2026-09-20 19:22:58 -0700，作者 sunkaixuan2018 | 实测 |
| 基线 | `git describe --tags` = **`v0.20.2rc1-18-g213933829`**；tag `v0.20.2rc1` = `367b8e62da799870a7476ce34f5f7658589a8aad`（2026-06-03） | 实测 |
| remotes | `origin = https://github.com/vllm-project/vllm-ascend`，`fork = https://github.com/sunkaixuan2018/vllm-ascend.git` | 实测 |
| 是否已推远端 | **没有**。`git branch -a --contains HEAD` 只列出本地 `feat/csa-attn-cut-20260920`，没有任何 `remotes/…` 行 | 实测 |
| 仓库形态 | **浅克隆**（`.git/shallow`），所以 `git merge-base HEAD origin/main` 返回空、与 `origin/main` 无共同祖先 | 实测 |
| 安装身份 | `vllm_ascend 0.20.2rc2.dev11+g98528d81a.d20260920`，editable 指向 `$V`（`$V/vllm_ascend/_version.py` 同串） | 实测 |
| 工作区 | **脏**：`M csrc/utils/inc/kernel/moe_distribute_base.h`（+8/−47）；未跟踪 `build_output/`、`csrc/build_out/`、`tools/pto_csa/stack/env_own.sh.pre_cache`、`vllm_ascend/attention/pto_attn.py.pre_repage`、`vllm_ascend/ops/dsa.py.pre_attn` | 实测 |

两个坑要点名：
1. **dist 版本号里的 `g98528d81a` 是安装那一刻的 setuptools_scm 快照**（commit `98528d81a`，2026-09-20 01:58），比 HEAD 落后 7 个 commit。因为是 editable 安装，Python 侧跑的是源码树的最新状态，**版本字符串不能当作"正在跑什么"的依据**；要引用就引 commit。
2. `moe_distribute_base.h` 是 C++ 侧改动（编进 csrc 扩展，不在 Python 路径上），但它意味着**从 `213933829` 重新 clone 不能复现这台机上的栈**。

### c) CSA 适配代码所在仓/分支/commit

就是同一棵 vllm-ascend 树，分支 **`feat/csa-attn-cut-20260920`**。按时间顺序，引入本次 attention-only 适配的 commit（实测 `git log --numstat`）：

| commit | 时间(-0700) | 内容 | 改动量 |
| --- | --- | --- | --- |
| `98528d81a` | 09-20 01:58 | Add: vendor the PyPTO DSV4 kernels and cut an attention-only CSA entry | 新增 `pto_kernels/**` 共 22 文件 13121 行，其中 `dspark/decode_csa.py` 2244 行 |
| `8f9874dc3` | 09-20 03:32 | Add: translate vLLM's decode state into the attention-only CSA kernel's arguments | **新建 `pto_attn.py`（656 行）**，同时动 `pto_kernels/`、`ops/dsa.py` |
| `4c2543bc7` | 09-20 03:43 | Fix: read weights by shape, and let an empty forwarded switch fall back | `pto_attn.py` +71/−29，动 `pto_kernels/` |
| `5c42578cf` | 09-20 07:09 | Add: run the attention-only CSA kernel on a live vLLM decode step | `pto_attn.py` +138/−37，动 `ops/dsa.py` |
| `74e00de9c` | 09-20 07:35 | Add: substitute the attention on the decode path, and count what it is offered | `pto_attn.py` +20/−2，`ops/dsa.py`（替换钩子落地） |
| `9dfbcb113` | 09-20 17:41 | Style: store pto_attn.py with LF line endings | 817/817（纯行尾） |
| `687c7989d` | 09-20 17:42 | Fix: address a compacted KV cache in its own coordinates | `pto_attn.py` +112/−20 |
| `213933829` | 09-20 19:22 | Fix: let the substitution be recorded into an ACLGraph | `pto_attn.py` +117/−8，`ops/dsa.py` = **HEAD** |

更早的层级替换线（attention-only 之前，供背景）：`573529e08`(09-18) → `303e242bf` → `5ac6e5f2b` → `305337346` → `20764222e` → `c332808ca`，外加 YunjiQin 的 swimlane 线 `0bd103b23`/`d060cd7b1`/`8dc3e8561`/`1982d167e`(merge)。整条分支相对 `origin/main` 共 19 个 commit。

**kernel 副本的来历（重要）**：`$V/vllm_ascend/attention/pto_kernels/` 是 pypto-lib 的**内联副本**，不是引用 `$LIB`。
- `pto_kernels/__init__.py`:18 声明 `PYPTO_LIB_COMMIT = "15d9ae75aaba452594bff0fe16500fadc250564d"`；`VARIANTS` 里 `dspark → models/deepseek_v4_flash_dspark`，入口模块 `("decode_csa", "decode_metadata")`。
- **这个 commit 在 `$LIB` 里不存在**（`$LIB` 是 depth=1 浅克隆，只有 `675f027`），`git cat-file -t 15d9ae75…` → `bad object`。实测。
- 与 `$LIB/models/deepseek_v4_flash_dspark/decode_csa.py` 的差异：**原始 diff 4323 行（看着像整文件不同），但那是行尾造成的** —— 内联副本是 **CRLF**（`git ls-files --eol` → `i/crlf w/crlf`，2244 行全带 CR），`$LIB` 那份是 LF。用 `diff -u --strip-trailing-cr` 后只剩 **480 行 / +223 −55**，内容是：裸 import → 包内相对 import、多一个 `WEIGHTS_WORKERS as IDX_WEIGHTS_WORKERS`、新增 `TP1_CSA_WB_WORKERS = 8`，以及**新增的两个函数** `decode_csa_attn_tp1`（`@pl.jit.inline`）与 `decode_csa_attn_tp1_test`（`@pl.jit` 入口，:1188）。上游那份只有 `decode_csa_test`(:605) 和 `decode_csa_tp1_test`(:1009)，**没有 attn 入口** —— 所以这个入口是我们这边切出来的，不是 lib 现成的。
- `pto_attn.py`、`ops/dsa.py` 是 LF（`i/lf`），只有内联 kernel 是 CRLF。

### d) pypto / simpler / pypto-lib

| 仓 | 路径 | commit | 分支 | 哪个远端分支含它 | 工作区 |
| --- | --- | --- | --- | --- | --- |
| pypto | `$P` | `6b49cfd58de65f8a325339a30d6fa45e1973c237` | 本地 `feat/kernel-mode-20260920` | **`origin/feat/kernel-mode-integration-test`**，且该远端分支 tip **正好等于 HEAD**（两向 `rev-list --count` 均为 0） | **脏**：`M python/pypto/torch/shutdown.py` |
| simpler | `$P/runtime` | `17ea300256e2a6db5af397ce619a7d480b595d80` | **detached**（`HEAD detached at 17ea3002`） | **`origin/feat/kernel-mode-integration-test`**，tip 亦等于 HEAD | 干净 |
| pypto-lib | `$LIB` | `675f027f1bf9ce8f79740f0b2ed042b0c4b2c8b3` | `main` | `origin/main`（浅克隆 depth=1，只有这一个 ref） | 干净 |

- pypto HEAD 主题：`fix(runtime): adopt Perfetto-compatible swimlane IDs (#2835)`，2026-09-20 15:40:07 +0800，YunjiQin；remote `origin = https://github.com/hw-native-sys/pypto.git`。full clone（非浅）。
- simpler HEAD 主题：`Fix: preserve Perfetto flows with numeric trace IDs (#2393)`，2026-09-20 14:12:31 +0800；remote `origin = https://github.com/hw-native-sys/simpler`。**它是 pypto 的 submodule**：`$P/.gitmodules` 有 `[submodule "simpler"] path = runtime`，`git submodule status` 显示 `17ea3002… runtime (remotes/origin/feat/kernel-mode-integration-test)`，即 checkout 与 pypto 记录的 gitlink 一致。full clone。
- pypto-lib HEAD 主题：`Add: fuse DSpark decode into one device-state L2 (#1263)`，2026-09-17 17:02:33 +0800。
- **pypto 的本地未提交改动必须交代**（`git diff python/pypto/torch/shutdown.py`，实测）：`require_supported_framework` 的白名单由 `if version != "2.6.0.post2"` 改成 `if version not in ("2.6.0.post2", "2.10.0")`。这台机的 torch_npu 是 **2.10.0**，所以**没有这 3 行本地补丁，`pypto.torch.init()` 会直接抛 RuntimeError**——从 `6b49cfd` 干净 clone 跑不起来。
- **`$LIB` 在 vLLM 运行时不在 PYTHONPATH 上**：`$V/tools/pto_csa/serving/dsv4_case_inner.sh`:19-20 注释明写 "kernel 源码内联在 vllm_ascend/attention/pto_kernels/ 下，走包内相对 import，pypto-lib 不上路径"，:30 导出的 PYTHONPATH 也只有 `$PYPTO_ROOT/python:$PYPTO_ROOT/runtime:$PYPTO_ROOT/runtime/python`。`$LIB` 只被 `tools/pto_csa/kernel/*` 那批 a-tier/b-tier 脚本用到。**所以 lib 团队如果照着 `675f027` 改 kernel，改的不是服务进程跑的那份文件。**

### e) 注册代码、参数绑定、钩子点（全部逐字）

**kernel 导入与 TP 注入** — `$V/vllm_ascend/attention/pto_attn.py`:

```
29|def _env_int(name: str, default: int) -> int:
32|    return int(os.environ.get(name, "") or default)
35|_TP = _env_int("PTO_ATTN_TP", 4)
38|def _import_kernel():
39|    argv = sys.argv
40|    if not any(a == "--tp" or a.startswith("--tp=") for a in argv):
41|        sys.argv = [*argv, "--tp", str(_TP)]
42|    try:
43|        from .pto_kernels.dspark import config as kcfg
44|        from .pto_kernels.dspark import decode_csa as kcsa
45|    finally:
46|        sys.argv = argv
47|    return kcsa, kcfg
54|def kernel():                      # 进程内缓存 _KCSA/_KCFG
```

注意 :32 —— 空串等同于未设，**所以 `PTO_ATTN_TP` 即使被启动脚本以空值转发，`_TP` 仍是 4**（B = 64//4 = 16）。同理 `PTO_ATTN_SEQ` 默认 1（:909、:979）。

**注册函数**（问题里问的 `_registered`）— `pto_attn.py`:828-840，逐字：

```python
828|_OP = None
829|_DONE: set = set()
830|
831|
832|def _registered():
833|    global _OP
834|    if _OP is None:
835|        from pypto.torch import init, register
836|
837|        kcsa, _ = kernel()
838|        init()
839|        _OP = register(kcsa.decode_csa_attn_tp1_test, "pypto_csa::attention_csa")
840|    return _OP
```

`register` = `$P/python/pypto/torch/registration.py`:198（由 `$P/python/pypto/torch/__init__.py`:17 导出），签名 `register(kernel: JITFunction, name: str, *, constexpr=None, config=None) -> torch._ops.OpOverload`，返回 `torch.ops.pypto_csa.attention_csa.default`；docstring(:205-217) 明确"NPU graph capture requires prior execution of each specialization outside capture"、"Replay uses captured addresses and scalar values; update tensor contents in place"——这正是 `687c7989d`/`213933829` 两个修复对应的约束。

**ARG_ORDER（46 个，逐字复制，`pto_attn.py`:670-689）**：

```python
670|# --- the 46 arguments --------------------------------------------------------
671|
672|ARG_ORDER = (
673|    "x_normed",
674|    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
675|    "freqs_cos", "freqs_sin", "cmp_freqs_cos", "cmp_freqs_sin",
676|    "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
677|    "compress_state", "compress_state_block_table",
678|    "idx_wq_b", "idx_wq_b_scale", "weights_proj", "hadamard_idx",
679|    "inner_wkv", "inner_wgate", "inner_ape", "inner_norm_w",
680|    "inner_compress_state", "inner_compress_state_block_table",
681|    "kv_cache", "cmp_kv", "cmp_block_table",
682|    "idx_kv_cache", "idx_kv_scale", "idx_block_table",
683|    "ori_slot_mapping", "window_swa_indices",
684|    "cmp_slot_mapping", "idx_slot_mapping",
685|    "state_slot_mapping", "inner_state_slot_mapping",
686|    "position_ids", "kv_seq_lens", "attn_sink",
687|    "wo_a", "wo_b", "wo_b_scale",
688|    "attn_out",
689|)
```

逐行计数核对 = 1+6+4+4+2+4+4+2+3+3+2+2+2+3+3+1 = **46**，与 `decode_csa.py`:1188-1234 形参顺序**逐个对齐**（:1189 `x_normed` … :1234 `attn_out`）。kernel 侧 6 个 `pl.InOut`：`compress_state`(:1204)、`inner_compress_state`(:1214)、`kv_cache`(:1216)、`cmp_kv`(:1217)、`idx_kv_cache`(:1219)、`idx_kv_scale`(:1220)；1 个 `pl.Out`：`attn_out`(:1234)。

**绑定点**：`build_args`（`pto_attn.py`:712 定义）最后一行组装列表，:806-808：

```python
806|    return [a[name] for name in ARG_ORDER], (plan_m, plan_i, state_c, ist_c,
807|                                             main_dim, inner_dim, pos, ks, n_real,
808|                                             paged)
```

**调用点**：
- 替换路径 `pto_attn.py`:998 `_registered()(*args)`；紧接 :1000-1003 两次 `write_state_ring(...)` 回写压缩器 ring，:1004-1005 `for pg in paged: pg.commit()`，:1007-1009 用 `take = arange(n_real)*ks` 抽掉矩形 padding 行再写 `output`。
- 对照路径 `pto_attn.py`:920 `op = _registered()`，:924 `op(*args)`，:925 `torch.npu.synchronize()`。

**钩子点** — `$V/vllm_ascend/ops/dsa.py`（自定义算子 `dsa_forward`，:255-261 `direct_register_custom_op(op_name="dsa_forward", mutates_args=["output"], dispatch_key="PrivateUse1")`）：

```
195|    forward_context: ForwardContext = get_forward_context()
198|        attn_metadata = filter_metadata(forward_context.attn_metadata, self.prefix)
207|    kv_cache = _build_kv_cache(self, forward_context)
...
222|    if _os.environ.get("PTO_ATTN_REPLACE", "").strip() not in ("", "0"):
223|        from vllm_ascend.attention import pto_attn
227|        if pto_attn.substitute(self, hidden_states, kv_cache, attn_metadata, output):
228|            return
229|
230|    self.dsa_attn.impl.forward(
231|        self.dsa_attn.layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv, output
232|    )
```

同文件还有 :209-220 `PTO_ATTN_PROBE` 结构 dump 钩子、:234-242 `PTO_ATTN_COMPARE` 单次对照钩子（都在 native 之后、且 `debug_allowed()` 会在 capture 下拒绝）。

六元组 cache 的构造 `_build_kv_cache` 在 :269-303，**顺序是** `(compress_kv_cache, swa_kv_cache, state_cache, indexer_state_cache, indexer_k_cache, indexer_scale_cache)`，每个过 `unfold_kvcache`(:306-309，剥 len==1 的 list)；`indexer_k_cache/indexer_scale_cache` 来自 :286-289 `self.indexer.k_cache.kv_cache[0][0]` 与 `[0][1]`。`pto_attn.build_args` 在 :725 按 `cmp_kv_c, swa_kv_c, state_c, ist_c, idx_k_c, idx_s_c = kv_cache` 解包——**与 `_build_kv_cache` 的返回顺序一致**。metadata 侧 `filter_metadata`(:264-266) 按 key 排序，`build_args`:724 解为 `cmp_md, cst_md, ist_md, idx_md, swa_md`。

`substitute()` 的拒绝条件（:977、:982）：`ratio != 4` 或 `decode is None` 直接返回 False；`n_offered > kcsa.B` 也返回 False（capture 期的 padded batch）。

### f) 安装包解析位置（默认 vs 启动脚本覆盖）—— 两者确实不同，而且默认那份是**悬空的**

**默认（登录 shell，`/usr/bin/python3` = Python 3.10.9，user-site 开启，无 PYTHONPATH）**，实测 `importlib.util.find_spec`：

| 模块 | 解析到 |
| --- | --- |
| `pypto` | `/data/sunkaixuan/all_pyptos/auto-deps-pre-upstream-5983/python/pypto/__init__.py` |
| `simpler` | `/data/sunkaixuan/lcw_subdir/simpler-worker-async-pr7/python/simpler/__init__.py` |
| `simpler_setup` | `/data/sunkaixuan/lcw_subdir/simpler-worker-async-pr7/simpler_setup/__init__.py` |
| `vllm` / `vllm_ascend` | `None`（根本找不到） |

来源是 user-site 两个 editable `.pth`：`/data/sunkaixuan/.local/lib/python3.10/site-packages/_pypto_editable.pth`（指 `…/all_pyptos/auto-deps-pre-upstream-5983/python`）与 `_simpler_editable.pth`（指 `…/lcw_subdir/simpler-worker-async-pr7` 两条）。**并且这两个目标目录都已经不存在了**（`/data/sunkaixuan/all_pyptos` 整个 MISSING，`simpler-worker-async-pr7` MISSING）——editable finder 不校验存在性，所以 `find_spec` 照样返回路径，真 import 会炸。user-site 里另有一份实体 `pypto/`（含 `pypto_core.cpython-310-…so`）和 `simpler_setup/`。

**被测栈（`$OWN/.venv`，Python 3.11.16）**：`tools/pto_csa/stack/env_own.sh`:29 `export PYTHONNOUSERSITE=1`（注释 :28 明写"用户 site 里有别人的 editable .pth 元路径查找器，会劫持 import，必须关掉"），:69 激活 venv。在该 venv 内实测：

| 模块 | 无 PYTHONPATH 时 | 加上启动脚本 PYTHONPATH 后 |
| --- | --- | --- |
| `pypto` | **`None` —— 根本没装** | `…/own_stack_20260918/pypto/python/pypto/__init__.py` |
| `simpler` | `…/own_stack_20260918/pypto/runtime/python/simpler/__init__.py` | 同左 |
| `simpler_setup` | `…/own_stack_20260918/pypto/runtime/simpler_setup/__init__.py` | 同左 |
| `_task_interface` | `$OWN/.venv/lib/python3.11/site-packages/_task_interface.cpython-311-aarch64-linux-gnu.so` | 同左 |
| `vllm` / `vllm_ascend` | `$VL` / `$V`（editable finder） | 同左 |

即：
- **`pypto` 唯一的解析途径就是 PYTHONPATH**，由 `$V/tools/pto_csa/serving/dsv4_case_inner.sh`:30 注入：`export PYTHONPATH="$PYPTO_ROOT/python:$PYPTO_ROOT/runtime:$PYPTO_ROOT/runtime/python${PYTHONPATH:+:$PYTHONPATH}"`（仅在 `PTO_CSA`/`PTO_ATTN_COMPARE`/`PTO_ATTN_REPLACE` 任一非空非 0 时执行，见 :23-33）。
- **`simpler` 不靠 PYTHONPATH 也对**：venv 里有 scikit-build-core 的 `_editable_skbc_simpler.pth`，内容两行 `…/own_stack_20260918/pypto/runtime` 与 `…/pypto/runtime/python`；`pip list` 显示 `simpler 0.1.0  …/own_stack_20260918/pypto/runtime`。
- **`_task_interface`（nanobind 扩展）是 venv site-packages 里的实体 .so，不在源码树**——重编 simpler 而不重装，就会留着旧 binding（已知的 ABI 坑）。
- venv 其它相关版本：`torch 2.10.0`、`torch_npu 2.10.0`、`torchvision 0.25.0`、`torchaudio 2.10.0`、`nanobind 2.15.0`。

**这两次 profile 确实用的是上面这套（实测，非推断）**：`…/fdoprof/prof_pto/dsv4_mtp_vllm_20260920_192612_dp1_bs4/task_submit.log` 里有
`[own] python = /data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/.venv/bin/python Python 3.11.16`、
`[own] pypto = …/own_stack_20260918/pypto`、`[own] pypto-lib = …/all_libs/pypto-lib-csa-20260917`、`[own] ptoas = /usr/local/ptoas/0.61`、
`[dsv4-vllm] PTO 工具链已上路径 pypto=…/own_stack_20260918/pypto`。

### 把两份 msprof 产物钉到 commit 上（实测）

两个 run 的 `run_meta.txt` 都写着 **`VLLM_ASCEND_COMMIT=213933829aededf9e75359fa1de6e14426b81c5f`**（= 当前 HEAD），以及 `ASCEND_RT_VISIBLE_DEVICES=15 / TP=1 / DP=1 / LOCAL_BS=4 / PROMPT_TOKENS=1024 / MAX_TOKENS=32 / MAX_MODEL_LEN=8704 / SPECULATIVE_CONFIG=(disabled) / MODEL_NUM_HIDDEN_LAYERS=3 / MODEL_MTP_LAYERS=0`，`status.txt = status=passed`。时间戳 `prof_native` 19:24:38、`prof_pto` 19:26:13（HEAD commit 时间 19:22:58）。

替换是否真的生效也是实测的：`prof_pto/.../server.log` 有 `[pto-attn-offer] model.layers.2.self_attn.attn ratio=4 decode=True`、`[pto-attn-audit] tensors=20 requests=4`、`[pto-attn-ran] n=1 tokens=4 capturing=False` / `n=2 tokens=4 capturing=True`（即 capture 真的录进去了，对应 `213933829`），并且 `[pto-attn-audit] moved_between_steps=none`（vLLM 的 block_table/slot_mapping/seq_lens/input_positions 跨步不换地址）；`prof_native` 的 log 里一条都没有。三层里只有 layer 2 是 ratio=4，layer 0/1 是 ratio=0，被 :977 拒绝。

### 交给 lib 团队时请连带说明的三件事

1. 要改的 kernel 文件是 **`$V/vllm_ascend/attention/pto_kernels/dspark/decode_csa.py`（CRLF，2244 行，入口在 :1188）**，不是 `$LIB` 里的那份；两者差 480 行（CR 归一化后）。
2. 从 commit 干净复现不成立：vllm-ascend 有 `moe_distribute_base.h` 改动，pypto 有 `shutdown.py` 的 torch_npu 2.10.0 白名单补丁（缺它 `init()` 直接抛错）。
3. `pypto` 在 venv 里没有安装记录，版本身份只能由 `$P` 的 git 状态给出；任何自动化复现脚本必须自己带 PYTHONPATH。

#### FACTS
- 【实测】vLLM 树 $VL 是 git 浅克隆（.git/shallow 存在）且 detached HEAD，commit bc150f50299199599673614f80d12a196f377655，tag v0.20.2 正指向该 commit（git describe --tags = v0.20.2，tag --points-at HEAD = v0.20.2），工作区干净，remote origin = https://github.com/vllm-project/vllm.git
- 【读码】/data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-v0.20.2/vllm/_version.py:22 `__version__ = version = '0.20.2'`，:26 `__commit_id__ = commit_id = 'gbc150f502'`
- 【实测】pip 里 vLLM 的 dist 是 `vllm 0.20.2+empty`（editable，指向 $VL），venv 内有 vllm-0.20.2+empty.dist-info 与 __editable__.vllm-0.20.2+empty.pth
- 【实测】vllm-ascend 树 $V：HEAD = 213933829aededf9e75359fa1de6e14426b81c5f，分支 feat/csa-attn-cut-20260920，主题 'Fix: let the substitution be recorded into an ACLGraph'，2026-09-20 19:22:58 -0700
- 【实测】$V 的 git describe --tags = v0.20.2rc1-18-g213933829；tag v0.20.2rc1 = 367b8e62da799870a7476ce34f5f7658589a8aad（2026-06-03 22:14:23 +0800）
- 【实测】$V 的 `git branch -a --contains HEAD` 只返回本地分支 feat/csa-attn-cut-20260920，无任何 remotes/ 行 —— 该分支未推到 origin 或 fork
- 【实测】$V 是浅克隆，`git merge-base HEAD origin/main` 返回空（与 origin/main 无共同祖先）；remotes 为 origin=vllm-project/vllm-ascend、fork=sunkaixuan2018/vllm-ascend
- 【实测】$V 工作区脏：M csrc/utils/inc/kernel/moe_distribute_base.h（+8/−47），另有未跟踪 build_output/、csrc/build_out/、tools/pto_csa/stack/env_own.sh.pre_cache、vllm_ascend/attention/pto_attn.py.pre_repage、vllm_ascend/ops/dsa.py.pre_attn
- 【读码】$V/vllm_ascend/_version.py 报 0.20.2rc2.dev11+g98528d81a.d20260920，对应安装时刻的 commit 98528d81a，比 HEAD 落后 7 个 commit；因为是 editable 安装，运行的是源码树最新状态
- 【实测】适配 commit 序列（git log --numstat）：98528d81a(09-20 01:58 内联 kernel，新增 pto_kernels/** 22 文件 13121 行) → 8f9874dc3(03:32 新建 pto_attn.py 656 行) → 4c2543bc7(03:43 +71/−29) → 5c42578cf(07:09 +138/−37) → 74e00de9c(07:35 +20/−2) → 9dfbcb113(17:41 纯 LF 行尾 817/817) → 687c7989d(17:42 +112/−20) → 213933829(19:22 +117/−8)
- 【实测】$V/vllm_ascend/ops/dsa.py 的 git log：213933829、74e00de9c、5c42578cf、8f9874dc3 四个适配 commit，加上基线 367b8e62d
- 【读码】$V/vllm_ascend/attention/pto_kernels/__init__.py:18 `PYPTO_LIB_COMMIT = "15d9ae75aaba452594bff0fe16500fadc250564d"`，VARIANTS 中 dspark → models/deepseek_v4_flash_dspark，入口模块 ("decode_csa", "decode_metadata")
- 【实测】该声明的 pypto-lib commit 15d9ae75… 在 $LIB 中不存在：`git cat-file -t 15d9ae75…` 返回 bad object；$LIB 是 depth=1 浅克隆（rev-list --count HEAD = 1）
- 【实测】内联 decode_csa.py 是 CRLF（git ls-files --eol 显示 i/crlf w/crlf，2244 行全部含 CR），而 pto_attn.py 与 ops/dsa.py 是 i/lf；与 $LIB/models/deepseek_v4_flash_dspark/decode_csa.py 的原始 diff 4323 行，经 --strip-trailing-cr 归一化后只剩 480 行（+223/−55）
- 【实测】$LIB 的 dspark/decode_csa.py 只有 decode_csa_test(:605) 与 decode_csa_tp1_test(:1009)；内联副本额外有 decode_csa_attn_tp1_test(:1188)，以及新增的 decode_csa_attn_tp1、TP1_CSA_WB_WORKERS = 8、import WEIGHTS_WORKERS as IDX_WEIGHTS_WORKERS
- 【实测】pypto $P：HEAD 6b49cfd58de65f8a325339a30d6fa45e1973c237，本地分支 feat/kernel-mode-20260920，主题 'fix(runtime): adopt Perfetto-compatible swimlane IDs (#2835)' 2026-09-20 15:40:07 +0800；含该 commit 的远端分支是 origin/feat/kernel-mode-integration-test，且该远端 tip 与 HEAD 完全相等（双向 rev-list --count 均为 0）
- 【实测】pypto 工作区脏：M python/pypto/torch/shutdown.py，diff 把 require_supported_framework 的判断从 `if version != "2.6.0.post2"` 改为 `if version not in ("2.6.0.post2", "2.10.0")`（本机 torch_npu 为 2.10.0）
- 【实测】simpler $P/runtime：detached HEAD 17ea300256e2a6db5af397ce619a7d480b595d80，主题 'Fix: preserve Perfetto flows with numeric trace IDs (#2393)' 2026-09-20 14:12:31 +0800，工作区干净；含该 commit 的远端分支是 origin/feat/kernel-mode-integration-test，tip 与 HEAD 相等
- 【实测】$P/.gitmodules 有 [submodule "simpler"] path = runtime, url = https://github.com/hw-native-sys/simpler；git submodule status 显示 17ea3002… runtime (remotes/origin/feat/kernel-mode-integration-test)，即 checkout 与 pypto 记录的 gitlink 一致
- 【实测】pypto-lib $LIB：HEAD 675f027f1bf9ce8f79740f0b2ed042b0c4b2c8b3，分支 main，主题 'Add: fuse DSpark decode into one device-state L2 (#1263)' 2026-09-17 17:02:33 +0800，工作区干净，depth=1 浅克隆，含它的远端分支只有 origin/main
- 【读码】$V/vllm_ascend/attention/pto_attn.py:832-840 `_registered()`：`from pypto.torch import init, register`；`kcsa, _ = kernel()`；`init()`；`_OP = register(kcsa.decode_csa_attn_tp1_test, "pypto_csa::attention_csa")`
- 【读码】register 定义在 /data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/pypto/python/pypto/torch/registration.py:198，签名 register(kernel: JITFunction, name: str, *, constexpr=None, config=None) -> torch._ops.OpOverload；由 $P/python/pypto/torch/__init__.py:17 导出
- 【读码】pto_attn.py:672-689 ARG_ORDER 共 46 个名字，逐行计数 1+6+4+4+2+4+4+2+3+3+2+2+2+3+3+1 = 46，顺序与 decode_csa.py:1189-1234 的形参逐个对齐
- 【读码】decode_csa.py:1187-1235 `@pl.jit def decode_csa_attn_tp1_test(...)`：6 个 pl.InOut（compress_state:1204、inner_compress_state:1214、kv_cache:1216、cmp_kv:1217、idx_kv_cache:1219、idx_kv_scale:1220），1 个 pl.Out（attn_out:1234）
- 【读码】pto_attn.py:806-808 `return [a[name] for name in ARG_ORDER], (plan_m, plan_i, state_c, ist_c, main_dim, inner_dim, pos, ks, n_real, paged)` —— 这是唯一的参数组装点
- 【读码】替换路径调用点 pto_attn.py:998 `_registered()(*args)`；:1000-1003 两次 write_state_ring 回写 compress_state/inner_compress_state；:1004-1005 `for pg in paged: pg.commit()`；:1007-1009 按 arange(n_real)*ks 抽行写回 output
- 【读码】对照路径调用点 pto_attn.py:920 `op = _registered()`、:924 `op(*args)`、:925 torch.npu.synchronize()
- 【读码】钩子点 $V/vllm_ascend/ops/dsa.py:222-228：`if _os.environ.get("PTO_ATTN_REPLACE", "").strip() not in ("", "0"):` → `if pto_attn.substitute(self, hidden_states, kv_cache, attn_metadata, output): return`；native 路径在 :230-232
- 【读码】dsa.py:255-261 direct_register_custom_op(op_name="dsa_forward", op_func=dsa_forward, mutates_args=["output"], fake_impl=dsa_forward_fake, dispatch_key="PrivateUse1")
- 【读码】dsa.py:269-303 _build_kv_cache 返回六元组顺序为 (compress_kv_cache, swa_kv_cache, state_cache, indexer_state_cache, indexer_k_cache, indexer_scale_cache)，每项过 unfold_kvcache(:306-309)；indexer k/scale 来自 :286-289 self.indexer.k_cache.kv_cache[0][0] 与 [0][1]
- 【读码】pto_attn.py:725 解包 `cmp_kv_c, swa_kv_c, state_c, ist_c, idx_k_c, idx_s_c = kv_cache`，与 _build_kv_cache 返回顺序一致；:724 `cmp_md, cst_md, ist_md, idx_md, swa_md = (m.decode for m in metadata_list)`，metadata 由 dsa.py:264-266 filter_metadata 按 key 排序给出
- 【读码】pto_attn.py:29-32 `_env_int` 把空串当未设，:35 `_TP = _env_int("PTO_ATTN_TP", 4)` —— 即使启动脚本转发空值，TP 仍为 4；PTO_ATTN_SEQ 默认 1（:909、:979）
- 【读码】pto_attn.py:38-47 _import_kernel 在 import 前把 `--tp <TP>` 注入 sys.argv，然后 `from .pto_kernels.dspark import decode_csa as kcsa`（包内相对 import，不经 pypto-lib）
- 【读码】pto_attn.py:977、:982 substitute 的拒绝条件：ratio != 4 或 decode is None 返回 False；n_offered > kcsa.B 也返回 False
- 【实测】登录 shell 默认解析（/usr/bin/python3 = 3.10.9，user-site 开，无 PYTHONPATH）：pypto → /data/sunkaixuan/all_pyptos/auto-deps-pre-upstream-5983/python/pypto/__init__.py，simpler → /data/sunkaixuan/lcw_subdir/simpler-worker-async-pr7/python/simpler/__init__.py，vllm 与 vllm_ascend 均为 None
- 【实测】上述两条来自 /data/sunkaixuan/.local/lib/python3.10/site-packages/_pypto_editable.pth 与 _simpler_editable.pth，而两个目标目录都已不存在（/data/sunkaixuan/all_pyptos MISSING，/data/sunkaixuan/lcw_subdir/simpler-worker-async-pr7 MISSING）
- 【读码】$V/tools/pto_csa/stack/env_own.sh:29 `export PYTHONNOUSERSITE=1`（注释 :28 说明 user site 的 editable .pth 会劫持 import），:69 激活 $OWN/.venv；:16 PYPTO_ROOT、:20 PYPTO_LIB_ROOT、:23 PTOAS_ROOT=/usr/local/ptoas/0.61
- 【实测】$OWN/.venv（Python 3.11.16）内，无 PYTHONPATH 时 find_spec('pypto') 返回 None（pypto 根本未安装）；simpler → $P/runtime/python/simpler/__init__.py、simpler_setup → $P/runtime/simpler_setup/__init__.py（来自 _editable_skbc_simpler.pth 的两行路径）
- 【读码】$V/tools/pto_csa/serving/dsv4_case_inner.sh:30 `export PYTHONPATH="$PYPTO_ROOT/python:$PYPTO_ROOT/runtime:$PYPTO_ROOT/runtime/python${PYTHONPATH:+:$PYTHONPATH}"`，仅当 PTO_CSA/PTO_ATTN_COMPARE/PTO_ATTN_REPLACE 任一非空非 0 时执行（:23-33）
- 【实测】加上该 PYTHONPATH 后 pypto 解析到 $P/python/pypto/__init__.py
- 【实测】venv 内 pip list：simpler 0.1.0 → $P/runtime；vllm 0.20.2+empty → $VL；vllm_ascend 0.20.2rc2.dev11+g98528d81a.d20260920 → $V；torch 2.10.0、torch_npu 2.10.0、torchvision 0.25.0、torchaudio 2.10.0、nanobind 2.15.0
- 【实测】_task_interface 解析到 $OWN/.venv/lib/python3.11/site-packages/_task_interface.cpython-311-aarch64-linux-gnu.so —— 是 venv 内的实体构建产物，不在源码树
- 【实测】/data/sunkaixuan/skx_log_output/dsv4_vllm/fdoprof/{prof_native/dsv4_mtp_vllm_20260920_192438_dp1_bs4,prof_pto/dsv4_mtp_vllm_20260920_192612_dp1_bs4}/run_meta.txt 均记录 VLLM_ASCEND_COMMIT=213933829aededf9e75359fa1de6e14426b81c5f，以及 ASCEND_RT_VISIBLE_DEVICES=15 / TP=1 / DP=1 / LOCAL_BS=4 / PROMPT_TOKENS=1024 / MAX_TOKENS=32 / MAX_MODEL_LEN=8704 / SPECULATIVE_CONFIG=(disabled) / MODEL_NUM_HIDDEN_LAYERS=3 / MODEL_MTP_LAYERS=0；status.txt 均为 status=passed
- 【实测】prof_pto 运行目录的 task_submit.log 含 `[own] python = /data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/.venv/bin/python Python 3.11.16`、`[own] pypto = …/own_stack_20260918/pypto`、`[own] pypto-lib = …/all_libs/pypto-lib-csa-20260917`、`[own] ptoas = /usr/local/ptoas/0.61`、`[dsv4-vllm] PTO 工具链已上路径 pypto=…/own_stack_20260918/pypto`
- 【实测】prof_pto 的 server.log 含 `[pto-attn-offer] model.layers.2.self_attn.attn ratio=4 decode=True`、`[pto-attn-audit] tensors=20 requests=4`、`[pto-attn-ran] n=1 tokens=4 capturing=False` 与 `n=2 tokens=4 capturing=True`、`[pto-attn-audit] moved_between_steps=none`；layer 0/1 显示 ratio=0；prof_native 的 server.log 无任何 pto-attn 行
- 【实测】/data/sunkaixuan/skx_log_output/csa_cut_20260920/compare/compare__model_layers_2_self_attn_attn.json 的 stage=complete、seq=1，arg_shapes 里 x_normed 为 [8,4096] bfloat16 contiguous；文件 mtime 为 2026-09-20 04:13，落在 commit 4c2543bc7(03:43) 与 5c42578cf(07:09) 之间
- 【读码】run_meta.txt 由 $V/tools/pto_csa/serving/dsv4_case_inner.sh:107-122 写出，其中 :120 只记录 vllm-ascend 的 commit（`git -C "$DSV4_VLLM_ASCEND_SRC" log -1 --format="VLLM_ASCEND_COMMIT=%H"`），未记录 pypto / simpler / pypto-lib 的 commit，也未记录 PTO_ATTN_* 开关
- 【读码】$V/tools/pto_csa/serving/run_dsv4_mtp_vllm.sh:77-79 通过 task-submit --env 转发 PTO_ATTN_REPLACE / PTO_ATTN_SEQ / PTO_ATTN_TP，均使用 ${VAR:-} 形式（可为空串）
- 【读码】$V/tools/pto_csa/serving/dsv4_case_inner.sh:19-20 注释明写 kernel 源码内联在 vllm_ascend/attention/pto_kernels/ 下、走包内相对 import、pypto-lib 不上路径；:30 导出的 PYTHONPATH 确实不含 PYPTO_LIB_ROOT

#### NEEDS_LIVE
- 【最高优先级】run_meta.txt 只记录 vllm-ascend 的 commit，pypto / simpler / pypto-lib 的身份对任何一次历史运行都只能靠'现在这棵树是什么'去推断，而这三棵树是共享且可被其他会话改动的。修法：在 $V/tools/pto_csa/serving/dsv4_case_inner.sh 第 120 行之后（`git -C "$DSV4_VLLM_ASCEND_SRC" log -1 …` 与 :121 的 `date` 之间）补打印：`git -C "$PYPTO_ROOT" log -1 --format="PYPTO_COMMIT=%H"`、`git -C "$PYPTO_ROOT/runtime" log -1 --format="SIMPLER_COMMIT=%H"`、`git -C "$PYPTO_LIB_ROOT" log -1 --format="PYPTO_LIB_COMMIT_ONDISK=%H"`，外加 `echo "PYPTO_DIRTY=$(git -C "$PYPTO_ROOT" status --porcelain | wc -l)"`、`echo "VLLM_ASCEND_DIRTY=$(git -C "$DSV4_VLLM_ASCEND_SRC" status --porcelain | wc -l)"` 与 `echo "PTO_ATTN_REPLACE=${PTO_ATTN_REPLACE:-} PTO_ATTN_TP=${PTO_ATTN_TP:-} PTO_ATTN_SEQ=${PTO_ATTN_SEQ:-} PTO_CSA=${PTO_CSA:-}"`。
- 两棵树都是脏的，从 commit 干净 clone 复现不出被测栈。修法：同样在 dsv4_case_inner.sh 写 run_meta.txt 的那个花括号块（:107-122）之后，把两份 diff 落盘到运行目录：`git -C "$DSV4_VLLM_ASCEND_SRC" diff > "$OUT/vllm_ascend.diff"`、`git -C "$PYPTO_ROOT" diff > "$OUT/pypto.diff"`。目前已知需要保留的两处：vllm-ascend 的 csrc/utils/inc/kernel/moe_distribute_base.h，pypto 的 python/pypto/torch/shutdown.py（torch_npu 2.10.0 白名单，缺它 pypto.torch.init() 直接抛 RuntimeError）。
- pto_kernels/__init__.py:18 声明的上游 PYPTO_LIB_COMMIT=15d9ae75aaba452594bff0fe16500fadc250564d 在本机无法证实：$LIB 是 depth=1 浅克隆，只有 675f027。要settle 必须做其中之一：在 $LIB 执行 `git fetch --unshallow origin`（需要网络，本次未做，属于写操作故未执行）后 `git cat-file -t 15d9ae75…`；或直接请 lib 团队确认 15d9ae75 是哪个分支上的哪次提交、与 675f027 的关系。在此之前，唯一可测量的陈述只有'CR 归一化后与 $LIB HEAD 差 480 行'。
- 实际注册并执行的入口、以及 TP/B/S/T 的真实实例化值，没有任何一行日志打印过，目前全部是读码 + 默认值推断（pto_attn.py:32 把空串当未设，故 PTO_ATTN_TP 即使为空也得 4）。修法：在 $V/vllm_ascend/attention/pto_attn.py 第 839 行（`_OP = register(...)`）与第 840 行 `return _OP` 之间插入一行 `print(f"[pto-attn-reg] entry={kcsa.decode_csa_attn_tp1_test.__name__} TP={_TP} B={kcsa.B} S={kcsa.S} T={getattr(kcsa,'T',None)} CMP_MAX_BLOCKS={kcsa.CMP_MAX_BLOCKS} IDX_MAX_BLOCKS={kcsa.IDX_MAX_BLOCKS}", flush=True)`。注册只发生一次且在 capture 之外，所以这行不影响 ACLGraph，也不进热路径。
- vllm_ascend 的 dist 版本戳 g98528d81a 是安装时刻（98528d81a）的快照，之后 7 个 commit 与本地 moe_distribute_base.h 改动是否已重新编进 venv 里的 csrc 扩展，从源码树看不出来。要 settle：打印 venv 内 vllm_ascend 相关 .so 的 mtime 与 98528d81a/213933829 的提交时间对比（`find $OWN/.venv/lib/python3.11/site-packages -name '*vllm_ascend*' -newermt '2026-09-20 01:58'`），或直接重装一次再取版本串。
- compare 产物 /data/sunkaixuan/skx_log_output/csa_cut_20260920/compare/compare__model_layers_2_self_attn_attn.json 对应的代码状态是按 mtime(09-20 04:13) 推断为 4c2543bc7 时刻，文件本身不含版本字段。修法：在 pto_attn.py 的 compare_once 内，紧接 :901 `rec = {"layer": layer, "stage": "start"}` 之后加入 commit 字段（由环境变量传入，避免在该函数里调 git），并让 dsv4_case_inner.sh 把 VLLM_ASCEND_COMMIT 也 export 给服务进程。
- 分支 feat/csa-attn-cut-20260920 是否存在于 fork/origin 远端，本次未用 `git ls-remote`（需要网络）确认，只能说本机 refs 里没有。若需要给 lib 团队一个可拉取的 commit，需先 `git push fork feat/csa-attn-cut-20260920` 并回报 fork 上的 ref 名。

#### UNKNOWNS
- feat/csa-attn-cut-20260920 是否已推到任何远端：本机 `git branch -a --contains HEAD` 只有本地分支，但未执行 `git ls-remote`（需网络），所以只能断言'本机所见的 refs 里没有'，不能断言远端一定没有。
- $VL 的 bc150f502 属于上游哪条分支：该树是浅克隆且 detached，除 tag v0.20.2 外无分支信息可查。
- $LIB 的 675f027 除 origin/main 外还被哪些远端分支包含：depth=1 浅克隆只 fetch 了 main，无法回答。
- pypto-lib commit 15d9ae75（内联副本声明的上游来源）与 $LIB HEAD 675f027 的先后与差异内容：该对象不在本机任何仓库中。
- venv 里 vllm_ascend 的 csrc 原生扩展是在哪个 commit 上编的、是否包含本地 moe_distribute_base.h 改动：源码树与 dist-info 都回答不了。
- 这次 profile 运行时 PTO_ATTN_TP / PTO_ATTN_SEQ 的字面取值：脚本以 ${VAR:-} 转发、run_meta 未记录、日志未打印。按 pto_attn.py:32/35/909 的默认语义，TP=4、SEQ=1 成立（空串与未设同义），但这是代码推断而非该次运行的实测记录。
- 除 shutdown.py 与 moe_distribute_base.h 外，两棵树是否还有已 stash 或已提交但未推送的本地差异：本次只核对了 working tree 的 porcelain 状态，未检查 stash 栈（且按规则不动共享 stash）。


## 复核补充的细节更正

- $VL/vllm/_version.py line numbers are off by one and three: `__version__ = version = '0.20.2'` is at line 21 (answer says :22) and `__commit_id__ = commit_id = 'gbc150f502'` is at line 24 (answer says :26). Content is correct.
- $V/vllm_ascend/attention/pto_kernels/__init__.py: `PYPTO_LIB_COMMIT = "15d9ae75aaba452594bff0fe16500fadc250564d"` is at line 17, not :18. Line 18 is `VARIANTS = {`. Both facts are correct, the anchor for the first is off by one.
- The normalized diff against $LIB is +222/−54, not +223/−55. Counted with `awk '/^\+\+\+/{next} /^---/{next} /^\+/{a++} /^-/{r++}'` on `diff -u --strip-trailing-cr`. The answer's numbers include the `+++`/`---` header lines. The 480-line total and the 4323-line raw diff are both exact.
- Commit 98528d81a touches 22 files totalling 13121 added lines, but only 21 of them are under pto_kernels/. The answer reads as if all 22 are pto_kernels files. (Verified: `git show --numstat --format= 98528d81a | grep pto_kernels | wc -l` = 21.)
- The run_meta.txt brace block in $V/tools/pto_csa/serving/dsv4_case_inner.sh is `collect_meta()` spanning :94-122, not :107-122. The specific anchors the answer relies on — :120 for the lone VLLM_ASCEND_COMMIT line, :121 for `date`, :122 for the redirect — are all exact, so the proposed patch location still lands correctly; only the block's stated start is wrong.
- The server.log quotation is truncated and selective. The real lines end with a request count: `[pto-attn-offer] model.layers.2.self_attn.attn ratio=4 decode=True n=1` (also n=2, n=3). And there are six `[pto-attn-ran]` lines, not two: n=1 tokens=4 capturing=False, n=2 tokens=4 capturing=True, n=3 tokens=2 capturing=False, n=4 tokens=2 capturing=True, n=5 tokens=1 capturing=False, n=6 tokens=1 capturing=True. This strengthens rather than weakens the answer — it shows a warm-up plus a capture for each of the three capture sizes [4,2,1], not just for size 4 — but the answer should quote all six, because "only two lines" would otherwise read as "only one capture size was recorded".
- pto_attn.py:977 is paraphrased as `ratio != 4`. The code is `if ratio != COMPRESS_RATIO or decode is None:`, with `COMPRESS_RATIO = 4` defined at pto_attn.py:65 (module-local, also used at :278, :591, :893). The paraphrase is true but the constant name should be given, since a kernel-side reader will otherwise look for the literal.
- Caution for whoever consumes this downstream: the answer's gloss of the metadata order — "attn, compressor state, indexer-compressor state, indexer k, sliding window" matching `cmp_md, cst_md, ist_md, idx_md, swa_md` at pto_attn.py:724 — is taken from the docstring at :715-717 and the variable names, not from any observed key list. What is actually verified is only that dsa.py:264-266 sorts by key and that :724 unpacks five values. The six-tuple *cache* order at :725 IS verified against the construction code (dsa.py:291-303); the five-tuple *metadata* order is not. Settling it needs one live read: print `sorted(k for k in forward_context.attn_metadata if k.startswith(self.prefix))` at dsa.py:198, right after the `filter_metadata` call. This belongs in the unknowns list, and matters for any later question that indexes metadata_list positionally.

---

# 第 2 项 · 主压缩器 state 的完整布局

> 复核状态：已复核通过

#### 问卷第 2 项 —— 主压缩器 state 的完整布局

先给结论摘要（四个必须确认项里属于我这一项的）：
- state_c 实际传给 kernel 的是 **[num_blocks, 8, 1, 2048] FP32，stride=(32768, 2048, 2048, 1)，非连续，storage_offset=0**。msprof 里那个 `[34730,16,2048]` 是**底层存储 / base tensor**，不是算子看到的逻辑张量；native 路径的算子看到的是 `34730,8,2048`。
- 一页 16 行里**只有前 8 行是真实内容，后 8 行是纯 padding，vLLM 没有把任何第二份 cache 放进去**（和 indexer k_cache 的 padding 不同——那里确实塞了 scale）。
- **state block table 记的是整段历史的绝对逻辑页号**，shape `[B, 1088]` INT32，列 j 对应 position ∈ [8j, 8j+8)，**不是**固定的小 ring。ring 语义只存在于我们 kernel 这一侧。
- 8 行页边界 = vLLM 的页边界（每 8 个 position 换一个物理页）；16 行 ring wrap 是 **kernel 私有**的，vLLM 侧没有 wrap，它靠 sliding-window allocator 回收物理页。

---

### (a) state_c 的 shape / dtype / stride / storage_offset

**构造链（读码）**

1. `dsa.py:279` `state_cache = self.compressor.state_cache.kv_cache`，是六元组的第 2 项（index 2），再经 `dsa.py:306-309 unfold_kvcache` 把单元素 list 拆开。
2. 该 layer 的 spec：`vllm_ascend/models/deepseek_v4.py:519-525` —— `CompressorStateCache(state_dim=2*coff*head_dim, dtype=fp32, compress_ratio=4, block_size=8)`。coff=1+(ratio==4)=2，head_dim=512 ⇒ **state_dim = 2*2*512 = 2048**，**block_size = 8**。
3. `patch_deepseek_compressor.py:55-65` 用 `AscendCompressorStateCache.get_kv_cache_spec` 覆盖，返回 `SlidingWindowMLASpec(block_size=8, num_kv_heads=1, head_size=2048, dtype=fp32, sliding_window=8, alignment=None, page_size_padded=131072)`（第 56 行：state_dim==512 且 ratio==4 才走 16640，主压缩器 state_dim=2048 ⇒ 走 131072）。
4. 真实页大小：`vllm/v1/kv_cache_interface.py:479-484`（`SlidingWindowMLASpec.real_page_size_bytes`，model_version 为 None 分支）= `storage_block_size(=block_size/compress_ratio，spec 的 compress_ratio 字段默认 1 ⇒ =8) * num_kv_heads(1) * head_size(2048) * 4` = **65536 B**。
5. `kv_cache_interface.py:147-149` `page_size_bytes` 直接返回 `page_size_padded=131072`。
6. 分配：`model_runner_v1.py:3647-3658`（`use_compress` 分支）给整个 KVCacheTensor 开一块 `torch.zeros(size, dtype=int8)`；无 kv_transfer_config ⇒ **没有 `_align_memory` 切片，`storage_offset()==0`**。
7. 成形：`model_runner_v1.py:3811-3841` → `_adjust_kv_layout(kv_tensor, [kv_cache_shape], [fp32], page_size_bytes=131072)`；`_adjust_kv_layout` 在 `model_runner_v1.py:3753-3779`：
   - `num_element_per_page = 131072//4 = 32768`
   - `stride = torch.empty([nb,8,1,2048]).stride() = (16384,2048,2048,1)`
   - `target_stride = (32768, 2048, 2048, 1)`
   - `storage_offset = raw.storage_offset()//4 = 0`

**实测（artifact 证据）**

- `/data/sunkaixuan/skx_log_output/csa_cut_20260920/probe/model_layers_2_self_attn.json` 的 `kv_cache[2]`：
  `shape=[21777, 8, 1, 2048]`，`dtype=torch.float32`，`contig=false`，`stride=[32768, 2048, 2048, 1]`。与上面逐项吻合。
- native run 的 msprof（`prof_native/.../kernel_details.csv`，`Name=Compressor`）Input Shapes：
  `4,4096; 1024,4096; 1024,4096; **34730,8,2048**; 4,1024; 512; 4,64; 4,64; **4,1088**; 5; ; 4`，dtype `...;FLOAT;...;INT32;INT32;...`。算子拿到的 state 是 `state_cache.squeeze(-2)`（`dsa_v1.py:2306`）⇒ `[34730, 8, 2048]`。
- 那个 `[34730,16,2048]` 出现在 **PTO 替换 run** 的两个算子里：
  `aclnnIndexSelect_SliceAiCore_Slice  IN "34730,16,2048;3;3" OUT "34730,8,2048" FLOAT`
  `aclnnInplaceIndexCopy_SliceAiCore_Slice IN "34730,16,2048;3;3" OUT "34730,8,2048" FLOAT`
  也就是 `pto_attn.py:378 make_state_ring` 的 `index_select` 和 `:433 write_state_ring` 的 `index_copy_`，profiler 打的是 **base/storage 形状**。34730*16*2048 = 1,138,032,640，与你给的数字一致。

**"16" 从哪来（明确回答）**：16 = `page_size_padded / (MAIN_STATE_DIM * sizeof(fp32))` = `131072 / (2048*4)`。真实内容只有 `65536/(2048*4) = 8` 行。所以 16 = 2 × 8，纯粹是 `patch_deepseek_compressor.py:56` 那个硬编码 131072 把一页撑成了两倍。**不是** K/V 两份，也**不是** num_kv_heads。

**修正你们的说法**：`[34730,16,2048]`、1,138,032,640 元素 —— 数值正确，但它是**父存储**，不是 `state_c`。`state_c` 本身是 `[34730, 8, 1, 2048]`，numel = 569,016,320，storage_offset = 0，page stride = 32768 个 float。

---

### (b) 8 行的逻辑顺序：position → (page, row) 的映射

**决定性证据来自 CANN 算子源码**（这份在树里，不是闭源二进制）：
`vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/compressor/arch32/compressor_block_vec_perf.h:802-858`

```
blockTablebaseOffset = batchIdx * maxBlockNumPerBatch          // :807 / :835
blockIdOffset  = curSeqIdx / blockSize                          // :812 / :840
remainRowCnt   = curSeqIdx % blockSize                          // :813 / :841
idInBlockTable = blockTableGm[blockTablebaseOffset+blockIdOffset] // :814 / :842
stateOffset    = idInBlockTable * stride
               + remainRowCnt * 2 * coff * headDim
               + stateIdx * coff * headDim + dStartIdx           // :819-821 / :848-850
```

其中 `curSeqIdx = sliceInfo.bStartPos + sliceInfo.sIdx`（`:866`），`bStartPos` 来自 `start_pos`，而 `start_pos_decode = seq_lens - seq_lens_q`（`dsa_v1.py:898`）= 本 step 第一个 query token 的**绝对 position**。

所以：

- **row = position % 8**（`remainRowCnt`），**page = state_block_table[req, position // 8]**。
- 行是 position p, p+1, ... 递增排列，且**页对齐到 8 的倍数**（position 0..7 在列 0，8..15 在列 1，…），**不是**相对于请求窗口起点的偏移。
- **没有任何 modulo**。这正是 `pto_attn.py:288-300 _state_slots` docstring 说的 "ordinary paged lookup -- no modulo, no ring"。
- 行内 2048 个 float 的结构：`stateIdx ∈ {0,1}` 各占 `coff*headDim = 1024` ⇒ `[0:1024] = kv_state`，`[1024:2048] = score_state`。与 `vllm/model_executor/layers/deepseek_compressor.py:396` 的注释（"state_cache last dim packs [kv_state, score_state], each STATE_WIDTH wide"）一致，也与我们 kernel 的写回一致（`decode_compressor_ratio4.py:304-307`：`[0:OUT_DIM]=kv`，`[OUT_DIM:2048]=score+ape`）。
- 1024 内部再分两半：`decode_compressor_ratio4.py:188-190` 的 `state_half = 0 或 HEAD_DIM(512)`，对应 CANN 侧 `SaveState` 的 OVERLAP 分支（`:870-878` 先写 `dStartIdx`，再 `dStartIdx += headDim` 写第二次）。即 `[0:512]` 与 `[512:1024]` 是 overlap 的前/后半窗。

---

### (c) 另外 8 行是什么：**纯 padding**，独立复核通过

**独立复核路径（不依赖之前那次调查）：**

1. `_adjust_kv_layout`（`model_runner_v1.py:3753-3779`）是唯一把 padding 变成"第二份 cache"的机制：它按 `kv_cache_shape_list` 逐个建 view，每建一个就 `storage_offset_bytes += stride[0]*dtype_size` 往页内推进。
2. 调用点 `model_runner_v1.py:3818-3833`：只有当 `hasattr(spec,"scale_dim") and spec.scale_dim != 0` 时 list 才有**两**个元素。`SlidingWindowMLASpec` **没有** `scale_dim` 字段（`kv_cache_interface.py:456-462` 只有 cache_dtype_str / alignment / compress_ratio / model_version）；`scale_dim` 只在 `AscendMLAAttentionSpec` 上（`patch_deepseek_compressor.py:93`，indexer k_cache 专用）。
   ⇒ **主 state cache 只建了一个 view，页内 65536..131071 字节没有任何 view 指向。**
3. 反向对照（同一份 probe artifact，同一次 run）：indexer 的 k/scale **确实**共页——
   `kv_cache[4] = [21777,128,1,128] int8 stride [16640,128,128,1]`，
   `kv_cache[5] = [21777,128,1,1] fp16 stride [8320,1,1,1]`（8320 fp16 = 16640 B）。
   scale 起点 = 128*128 = 16384 B，正好是 k 的内容末尾，占掉 16640-16384=256 B 的 padding。这说明 `_adjust_kv_layout` 的"往 padding 里塞第二份"确实会发生——但对 state cache **没发生**。
4. msprof 两个 run 全表扫描：除上面那两个 `aclnnIndexSelect/IndexCopy`（我们自己的 ring seed/writeback）外，**没有任何算子**碰 `34730,16,2048` 或该分配的偏移 view。

**但有一个必须写进结论的限定（这直接回答"写错子页会不会毁真数据"）：**

页池是**跨 group 共享**的。`vllm/v1/core/kv_cache_utils.py:1285-1288` 原文：不同 group 的 layer 共享同一个 Tensor，"As layers of different groups have different block table, they will use different parts of the shared Tensor"。vllm-ascend 的 DeepSeek-V4 版本 `patch/platform/patch_kv_cache_utils.py:232-240`：`KVCacheTensor(size=ps*num_blocks, shared_by=[每个 group 在该 (tuple_idx, page_size) 槽位的 layer])`。131072 这个 bucket 里同时住着 full-MLA attn layer、SWA layer 和主 state layer（probe 里 `kv_cache[0]` 的 stride0 = 65536 个 bf16 = 131072 B，正是同一个 bucket）。

所以：
- **同一页内写到第 8..15 行**（页号正确、行号越界）⇒ 只碰 padding，**不会毁真数据**。
- **写到错误的页号** ⇒ 那一页很可能正被 attn 或 SWA layer 用着，**131072 B 全是真数据**，会静默损坏别的 layer/请求。
- 分配器保证同一时刻一个 block id 只属于一个 (request, group)，所以"自己合法持有的页"的 8..15 行永远是安全的垃圾区。

---

### (d) state block table：真实 shape / dtype / 值的样子

**Shape/dtype（实测，两处独立）**
- probe JSON（prefill step，B=1）第 2 个 metadata 的 `prefill.block_table = {shape:[1,1088], dtype:torch.int32, contig:true, stride:[1088,1]}`；第 3 个（indexer state）同样 `[1,1088]`。
- msprof native decode（bs=4）`Compressor` 的第 9 个输入 = `4,1088` INT32。

**构造与列含义（读码）**
- 缓冲区：`vllm/v1/worker/block_table.py:70-72` `self.block_table = _make_buffer(max_num_reqs, max_num_blocks_per_req, dtype=torch.int32)`。
- 宽度：`vllm/v1/worker/gpu_model_runner.py:6480-6482` `max_num_blocks_per_req = cdiv(max_model_len, block_size)` = `cdiv(8704, 8)` = **1088**。（对照：block_size=128 的几个 cache 宽度是 `cdiv(8704,128)=68`，probe 里也确实是 `[1,68]`。）
- decode 取用：`dsa_v1.py:1083` `block_table=self.block_table[:block_table_size, ...]`，源头是 `common_attn_metadata.block_table_tensor`（`dsa_v1.py:569`），即该 KV-cache-group 自己的 block table。
- **列 j 的含义：position ∈ [8j, 8j+8) 这一段逻辑页的物理页号**（由 (b) 的算子公式 `blockIdOffset = curSeqIdx/blockSize` 决定）。
- **值域与哨兵**：物理页号 ∈ [0, num_blocks)。**0 是 null_block**（`vllm/v1/core/block_pool.py:173-177`：`# To represent a placeholder block with block_id=0`，`null_block = free_block_queue.popleft()`）。被 sliding window 淘汰的列会被**原地替换成 null_block**（`single_type_kv_cache_manager.py:418-425`：`blocks[i] = self._null_block`），列号**不平移**。
- CANN 算子明确把 0 当 hole：写路径 `compressor_block_vec_perf.h:847` `if (idInBlockTable != 0) { ... }`；**读路径（:802-828）没有这个判断**，hole 会直接读物理页 0。
- ACLGraph padding 行被填 0：`dsa_v1.py:971` `self.block_table[num_reqs_actual:self.num_decodes, ...].fill_(0)`。

**典型一行长什么样（由上述规则推导，非实测）**
bs=4、prompt=1024、正在解码 position p=1024 的请求 r：
`num_computed = 1024` ⇒ `get_num_skipped_tokens = 1024-8+1 = 1017`（`single_type_kv_cache_manager.py:632`，sliding_window = coff*ratio = 8）⇒ `num_skipped_blocks = 1017//8 = 127` ⇒ 列 0..126 全被置成 0；列 127（position 1016..1023）和列 128（position 1024..1031）是真实物理页号；列 129..1087 是**上一个占用该行的请求留下的陈旧值**（`block_table.py:120-122 add_row` 只重置计数不清尾部，`:124-128 clear_row` 也只清 `:num_blocks`）。
即一行大致是 `[0,0,...,0, P_a, P_b, <stale...>]`，只有 1~2 个非零。

**对比我们喂给 kernel 的那张表**（compare artifact 实测）：
`compare__model_layers_2_self_attn_attn.json` 的 `compress_state_block_table = [[1,8], torch.int32, contig]`，值由 `pto_attn.py:436-440 state_block_table` 生成 = `bt[r][j] = r*8 + j`，B=1 时就是 `[[0,1,2,3,4,5,6,7]]`。**两张表的语义完全不同，不能互换。**

---

### (e) 【关键】绝对逻辑页 还是 固定 ring 槽位？

**答案：vLLM 侧是绝对逻辑页；ring 只存在于我们 kernel 侧。两者不可互换。**

**vLLM / native 侧 = 绝对（算术如下）**

```
curSeqIdx      = start_pos + sIdx           = 绝对 position p        (block_vec_perf.h:866 + dsa_v1.py:898)
blockIdOffset  = p / 8                       无 modulo               (block_vec_perf.h:812, :840)
remainRowCnt   = p % 8                                               (block_vec_perf.h:813, :841)
phys           = bt[req, p/8]                                        (block_vec_perf.h:814, :842)
addr           = phys*32768 + (p%8)*2048 + stateIdx*1024 + d         (block_vec_perf.h:819-821)
```
列索引 `p/8` 随 p 无界增长，上限 1088 = cdiv(max_model_len, 8)。**这就是整段历史的绝对逻辑页编号**。之所以"看起来只有几页"，是因为 sliding_window=8 让 `remove_skipped_blocks` 把老列换成 null_block，而**不是**因为表变短或循环复用列。

**kernel 侧 = 固定 16 行 ring（算术如下）**

```
ring_row  = logical_pos % STATE_STORAGE_LEN(=16)                     (decode_compressor_ratio4.py:192)
page_off  = ring_row / COMPRESS_STATE_BLOCK_SIZE(=2)                 (:193)
phys      = compress_state_block_table[c_idx, page_off]              (:194-195)
state_row = phys*2 + ring_row%2                                      (:198-199)
```
常量来源：`decode_compressor_ratio4.py:44-53` / `decode_csa.py:160-174` ——
`COFF=2, COMPRESS_RATIO=4 ⇒ MAIN_STATE_LEN=8`；`MAIN_STATE_STORAGE_LEN = 8 + S(=8) = 16`；`MAIN_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE = 2`；`MAIN_STATE_MAX_BLOCKS = ceil(16/2) = 8`。
kernel 自带的 metadata 生成器也确认有 modulo：`decode_metadata.py:225-244`
`csa_state_logical = position//2`；`bt[request, csa_state_logical % CSA_STATE_REQUIRED_BLOCKS(=8)]`；`slot = phys*2 + position%2`。
`(position//2) % 8 == (position % 16)//2`，两式等价。

**为什么不能直接把 vLLM 的表喂进去（数字）**
kernel 只读表的前 8 列（`MAIN_STATE_MAX_BLOCKS=8`）。把 `[B,1088]` 的表交给它，它只会用到列 0..7 —— 而这 8 列对应 position 0..63，在 decode 到 p=1024 时**全部是 null_block(0)**。于是每个请求的 16 行 state 全落到物理页 0 的 0..15 行，所有请求互相踩，而且 `phys >= 0` 的守卫（`:196`）**放行 0**（vLLM 的 hole 哨兵是 0，不是 -1）。这就是 `pto_attn.py:332-334` 注释说的 "collide on the null block -- silently"。**读码确认该注释属实。**

所以现状的做法（`pto_attn.py:774-779`）是：用 `state_ring_plan` 从 vLLM 的绝对表里 gather 出 16 行 ⇒ 拷成连续私有 ring ⇒ 给 kernel 一张自造的恒等表 `bt[r][j]=r*8+j` ⇒ 算完再 scatter 回去。

---

### (f) 8 行页边界 与 16 行 ring wrap

**vLLM 侧（8 行边界）**：每 8 个 position 换一个新的物理页（`p/8` 进位），block table 追加一列；被移出 sliding window 的旧列原地变 null_block(0)，物理页归还池子。**vLLM 自己没有任何 wrap**——列号单调增长到 1088；"循环"发生在物理页复用层面，对 block table 的列编号不可见。

**kernel 侧（16 行 wrap）**：`ring_row = pos % 16`，每 16 个 position 覆盖一圈。因为 kernel 只回看 `MAIN_STATE_LEN-1 = 7` 个历史位置（`decode_compressor_ratio4.py:166,184-185`：`window_start = token_pos-7`，`state_idx ∈ [0, 6]` ⇒ `logical_pos ∈ [token_pos-7, token_pos-1]`），16 行 ring 永远不会被自己覆盖掉还要用的行。

**`write_state_ring` 在边界上做了什么（`pto_attn.py:395-433`）**
- ring 覆盖 position `[first-8, first+7]`（`:363` `pos = first-(16-8)+i`，i∈[0,16)）。
- 这 16 个连续 position 跨 `span = 16/8 + 1 = 3`（`:410`）个 vLLM 逻辑页：`first ≡ 0 (mod 8)` 时实际只跨 2 页，否则跨 3 页。
- 它不按 `blk` 去重，而是按 `base = (first-8)//8` 加 `arange(span)` 构造槽位，再用 `scatter_` 把每个槽位的物理页填进去（`:413-423`），然后 `index_select → 改行 → index_copy_` 整页读改写（`:426-433`）。docstring（`:399-404`）解释这是为了避开 `index_copy_` 重复下标互相覆盖。

**这对 vLLM 行为的三点推论（都请在改 kernel 前确认）**

1. **一页 8 行、ring 16 行 ⇒ ring 恰好等于 2 个 vLLM 页。** 把 `MAIN_STATE_STORAGE_LEN` 从 16 改成别的值，或把 `S` 从 8 改掉，`span` 和整个 gather/scatter 计划都要跟着改；`VLLM_STATE_PAGE=8` 是 `pto_attn.py:64` 的硬编码，它等于 `deepseek_v4.py:524` 的 `block_size=8`，**这两个数必须一起改**。
2. **ring 的最老一行在 `first ≡ 7 (mod 8)` 时落在已被回收的列上。** 算术：被置空的列是 `0 .. (first-7)//8 - 1`；ring 最低列是 `(first-8)//8 = q-1`（first=8q+r）。r<7 时 `(first-7)//8 = q-1`，两者相等，全部命中活页；r=7 时 `(first-7)//8 = q`，于是列 `q-1` 已是 null_block(0)，那一行从物理页 0 读来。kernel 不会读它（它只读 `[first-7, first-1]`，全在列 q 内），写回又是把同样的字节写回去，所以**当前是良性的**——但 `state_ring_plan` 的守卫是 `blk >= 0`（`:369`），**放行 0**，一旦 ring 长度或回看窗口变长，这条路就会直接读到 null block。
3. **ring 的"未来"7 行（position `first+1 .. first+7`）落在尚未分配的列上。** `state_ring_plan` 的 `ok` 只检查 `lblk < vllm_bt.shape[1]`（1088），**没有**检查 `lblk < num_blocks_per_row[req]`。而 block table 的尾部列既不被 `add_row` 清零也不被 `clear_row` 清零（`block_table.py:116-128`），装的是**上一个占用该行下标的请求留下的陈旧物理页号**。目前这也良性，因为写回是 read-modify-write 的恒等操作。**但有一个真实的静默风险**：如果某个陈旧列的值恰好等于同一 span 内某个活页的页号，`flat_pages` 就出现重复下标，`index_copy_`（`:433`）后写覆盖先写，会**丢掉真实那一页的 state 更新**——而 `:399-404` 的 docstring 正是宣称"构造上无重复"。这个前提只在 span 内三列物理页互不相同时成立，陈旧列可以打破它。这是 2(f) 里我认为最需要你们在动手前实测的一条。

---

### 给你们改 kernel 时的一句话总结

`state_c` 是 `[num_blocks, 8, 1, 2048]` fp32、page stride 32768 float、offset 0 的**非连续** view；一页 16 行里后 8 行是死 padding（写它不会坏数据，写错页号会）；vLLM 的 state block table 是 `[B, cdiv(max_model_len,8)] = [4,1088]` INT32 的**绝对逻辑页表**，列 j ↔ position [8j,8j+8)，hole 哨兵是 **0** 不是 -1；kernel 的 `[B,8]` ring 表和它语义完全不同，中间必须保留 `state_ring_plan/make_state_ring/write_state_ring` 这一层转换（或者改 kernel 让它直接吃绝对表 + 把 `phys>=0` 守卫换成 `phys!=0`，并把 `MAIN_STATE_MAX_BLOCKS` 放宽到 1088）。

#### FACTS
- 读码: state_c 是 dsa.py 六元组 index 2，来自 self.compressor.state_cache.kv_cache，经 unfold_kvcache 拆单元素 list — /data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-ascend-v0.20.2rc1/vllm_ascend/ops/dsa.py:279, :291-303, :306-309
- 读码: 主压缩器 state_dim = 2*coff*head_dim = 2*2*512 = 2048，block_size=8 — vllm-ascend-v0.20.2rc1/vllm_ascend/models/deepseek_v4.py:519-525
- 读码: page_size_padded = 131072（state_dim!=512 分支），alignment=None — vllm-ascend-v0.20.2rc1/vllm_ascend/patch/worker/patch_deepseek_compressor.py:56, :57-65
- 读码: SlidingWindowMLASpec.real_page_size_bytes = storage_block_size*num_kv_heads*head_size*4 = 8*1*2048*4 = 65536 B；page_size_bytes 直接返回 page_size_padded — vllm-v0.20.2/vllm/v1/kv_cache_interface.py:479-484, :147-149
- 读码: 16 = page_size_padded(131072) / (MAIN_STATE_DIM 2048 * 4 B)；真实内容 8 行 = 65536/(2048*4)。padding 恰好是真实页的一倍
- 读码: raw tensor 为 torch.zeros(size, int8)（kv_transfer_config 为 None 时不切片）⇒ storage_offset()==0 — vllm-ascend-v0.20.2rc1/vllm_ascend/worker/model_runner_v1.py:3647-3658
- 读码: _adjust_kv_layout 用 target_stride=(page_size_bytes/dtype_size, *contig_stride[1:])、storage_offset=raw.storage_offset()//dtype_size 建 as_strided view — vllm-ascend-v0.20.2rc1/vllm_ascend/worker/model_runner_v1.py:3753-3779
- 读码: state cache 走 use_compress 分支，kv_cache_shape_list 只有 1 个元素（scale_dim 分支只对有 scale_dim 的 AscendMLAAttentionSpec 生效），所以页内 65536..131071 B 没有任何 view — vllm-ascend-v0.20.2rc1/vllm_ascend/worker/model_runner_v1.py:3811-3841
- 实测: probe artifact kv_cache[2] = shape [21777,8,1,2048], dtype float32, contig=false, stride [32768,2048,2048,1] — /data/sunkaixuan/skx_log_output/csa_cut_20260920/probe/model_layers_2_self_attn.json
- 实测: native run 的 Compressor 算子输入第 4 项是 34730,8,2048 FLOAT（= state_cache.squeeze(-2)），第 9 项是 4,1088 INT32（state block table）— /data/sunkaixuan/skx_log_output/dsv4_vllm/fdoprof/prof_native/dsv4_mtp_vllm_20260920_192438_dp1_bs4/profile/dp0_pp0_tp0_dcp0_ep0_rank0_3896415_20260920192555213_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv
- 实测: [34730,16,2048] 只出现在 PTO run 的 aclnnIndexSelect_SliceAiCore_Slice 和 aclnnInplaceIndexCopy_SliceAiCore_Slice，IN 34730,16,2048 / OUT 34730,8,2048，即 profiler 打的是 base storage，不是算子逻辑张量 — .../prof_pto/.../ASCEND_PROFILER_OUTPUT/kernel_details.csv
- 实测(反例): indexer 的 k/scale 确实共页 —— kv_cache[4]=[21777,128,1,128] int8 stride[16640,...]，kv_cache[5]=[21777,128,1,1] fp16 stride[8320,...]，scale 起点在页内 16384 B，占掉 16640-16384=256 B padding — 同一份 probe JSON
- 读码: CANN compressor 算子的 state 寻址 blockIdOffset=curSeqIdx/blockSize, remainRowCnt=curSeqIdx%blockSize, stateOffset=idInBlockTable*stride + remainRowCnt*2*coff*headDim + stateIdx*coff*headDim + dStartIdx，全程无 modulo — vllm-ascend-v0.20.2rc1/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/compressor/arch32/compressor_block_vec_perf.h:812-821 (读) / :840-850 (写)
- 读码: 写路径有 if (idInBlockTable != 0) 守卫（把 0 当 hole 跳过），读路径没有同样的守卫 — compressor_block_vec_perf.h:847 vs :802-828
- 读码: curSeqIdx = sliceInfo.bStartPos + sliceInfo.sIdx，bStartPos 来自 start_pos；start_pos_decode = seq_lens - seq_lens_q = 绝对 position — compressor_block_vec_perf.h:866 + vllm-ascend-v0.20.2rc1/vllm_ascend/attention/dsa_v1.py:898
- 读码: 行内 2048 float 分 [0:1024]=kv_state / [1024:2048]=score_state（stateIdx*coff*headDim），每半再按 OVERLAP 分 [0:512]/[512:1024] — compressor_block_vec_perf.h:819-821, :870-878; vllm-v0.20.2/vllm/model_executor/layers/deepseek_compressor.py:396; pto_kernels/dspark/decode_compressor_ratio4.py:188-190, :304-307
- 读码: block table 缓冲区 [max_num_reqs, max_num_blocks_per_req] int32 — vllm-v0.20.2/vllm/v1/worker/block_table.py:70-72
- 读码: max_num_blocks_per_req = cdiv(max_model_len, block_size) = cdiv(8704,8) = 1088 — vllm-v0.20.2/vllm/v1/worker/gpu_model_runner.py:6480-6482
- 实测: probe JSON 五个 metadata 的 prefill.block_table 依次为 [1,68] / [1,1088] / [1,1088] / [1,68] / [1,68]，对应 sorted key 顺序 attn / compressor.state_cache / indexer.compressor.state_cache / indexer.k_cache / swa_cache（与 dsa_v1.py:1885 注释一致）
- 读码: null_block 的 block_id 是 0 —— '# To represent a placeholder block with block_id=0'，null_block = free_block_queue.popleft() — vllm-v0.20.2/vllm/v1/core/block_pool.py:173-177
- 读码: sliding window 淘汰是原地替换 blocks[i] = self._null_block，列号不平移 — vllm-v0.20.2/vllm/v1/core/single_type_kv_cache_manager.py:401-426
- 读码: state spec 的 sliding_window = coff*compress_ratio = 8；get_num_skipped_tokens(n) = max(0, n-8+1) — patch_deepseek_compressor.py:51-52 + vllm/v1/core/single_type_kv_cache_manager.py:632
- 读码: ACLGraph padding 行被 fill_(0)，即指向 null block — vllm-ascend-v0.20.2rc1/vllm_ascend/attention/dsa_v1.py:971
- 读码: add_row 只重置计数、append_row 只写 num_blocks 个元素、clear_row 只清 :num_blocks —— 行尾列不清零，保留上一个请求的陈旧页号 — vllm-v0.20.2/vllm/v1/worker/block_table.py:116-128
- 读码: 不同 group 的 layer 共享同一块 KVCacheTensor，靠各自的 block table 用到不同部分 — vllm-v0.20.2/vllm/v1/core/kv_cache_utils.py:1285-1288；vllm-ascend 的 DSV4 版按 (tuple_idx, page_size) 发 KVCacheTensor，shared_by 是各 group 该槽位 layer 的并集 — vllm_ascend/patch/platform/patch_kv_cache_utils.py:232-240
- 读码: kernel 侧 ring 寻址 ring_row = logical_pos % STATE_STORAGE_LEN(16); page_off = ring_row/2; state_row = bt[c_idx,page_off]*2 + ring_row%2；守卫是 state_blk_id_i32 >= 0（会放行 0） — pto_kernels/dspark/decode_compressor_ratio4.py:192-199
- 读码: kernel 常量 MAIN_STATE_LEN = COFF*COMPRESS_RATIO = 8, MAIN_STATE_STORAGE_LEN = 8+S = 16, MAIN_STATE_BLOCK_SIZE = 2, MAIN_STATE_MAX_BLOCKS = 8 — pto_kernels/dspark/decode_csa.py:160-174 和 decode_compressor_ratio4.py:44-53
- 读码: kernel 只回看 7 个历史 position —— window_start = token_pos - STATE_LEN + 1，state_idx in range(STATE_LEN-1) ⇒ logical_pos ∈ [token_pos-7, token_pos-1] — pto_kernels/dspark/decode_compressor_ratio4.py:166, :184-185
- 读码: kernel 自带 metadata 也用 modulo: csa_state_logical = position//2; bt[request, csa_state_logical % 8]; slot = phys*2 + position%2 — pto_kernels/dspark/decode_metadata.py:225-244
- 读码: pto_attn 的 ring 覆盖 position [first-8, first+7]，span = 16/8+1 = 3 个 vLLM 逻辑页；valid 守卫只有 pos>=0 / lblk<vllm_bt.shape[1](1088) / blk>=0，没有 lblk < 该请求已分配列数 — vllm_ascend/attention/pto_attn.py:363-373, :410
- 读码: pto_attn 给 kernel 的是自造恒等表 bt[r][j] = r*8+j，state_slots 折叠成 r*16 + pos%16 — vllm_ascend/attention/pto_attn.py:436-440, :443-454
- 实测: compare artifact 里 compress_state=[8,2,2048] fp32 contig, compress_state_block_table=[1,8] int32（B=1, T=8），与 [B,1088] 的 vLLM 表语义完全不同 — /data/sunkaixuan/skx_log_output/csa_cut_20260920/compare/compare__model_layers_2_self_attn_attn.json
- 算术: first=8q+r 时，被置空列为 0..(first-7)//8-1；r<7 ⇒ (first-7)//8 = q-1 = ring 最低列，全部命中活页；r=7 ⇒ (first-7)//8 = q，ring 最低列 q-1 已是 null_block(0)

#### NEEDS_LIVE
- storage_offset / data_ptr / 底层 storage 大小（唯一还是读码推断的项）。在 /data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-ascend-v0.20.2rc1/vllm_ascend/attention/pto_attn.py 的 describe()（dump_structure 用的那个，约 :640 附近）里给每个 tensor 追加 storage_offset()、data_ptr()、untyped_storage().size()；或直接在 vllm_ascend/ops/dsa.py:291 的 return 之前对六个 cache 逐个 print。期望：state_c.storage_offset()==0，untyped_storage().size()==num_blocks*131072。
- padding 是否真的没人写。在 vllm_ascend/ops/dsa.py:279 拿到 state_cache 后，用 torch.as_strided(state_c 的 base, size=(nb,16,2048), stride=(32768,2048,1), storage_offset=0) 取出整页视图，跑 5~10 个 decode step 后 print parent[:, 8:16, :].abs().max() 与 .count_nonzero()。若恒为 0 则 (c) 的『纯 padding』得到运行时确认。
- 真实的 state block table 一行。在 vllm_ascend/attention/pto_attn.py:774（plan_m = state_ring_plan(...) 之后）加打印，并用 pto_attn.debug_allowed('PTO_ATTN_PROBE') 门控（会 D2H，ACLGraph capture 下必须跳过）：print(cst_md.block_table.shape, cst_md.block_table.dtype); p0=int(pos[0]); j=p0//8; print(cst_md.block_table[0, max(0,j-3):j+3], p0, (cst_md.block_table[0]==0).sum()). 期望看到『前面一长串 0 + 1~2 个非零 + 尾部陈旧值』。
- 确认 0 就是 hole 而不是合法页。同一处打印 (cst_md.block_table[:b, :j+1]==0).sum(dim=1) 与 j+1 的对比；再打印 cmp_md.seq_lens[:b]。若 zeros 数 ≈ j-1 则 (d)/(e) 的 null_block 推导成立。
- ring 跨 8 行页边界与 16 行 wrap 的实际行为。对同一个请求连续记录 ~40 个 decode step 的 (pos, pos%16, pos//8, bt[r, pos//8], bt[r, pos//8 - 1], bt[r, (pos+8)//8])，打印点同上（pto_attn.py:774）。要看的是：pos%16 归零那一步 bt 的哪一列变化、以及 pos%8 归零那一步是否出现新的物理页号。
- 【最重要】write_state_ring 的 span 内三列物理页是否可能重复。在 vllm_ascend/attention/pto_attn.py:425（flat_pages 构造之后、index_copy_ 之前）加 assert/打印：phys 形状 [b,span]，检查每行是否有重复值 —— print(phys, (phys[:, :, None] == phys[:, None, :]).sum(dim=(1,2))). 任何一行的计数 > span 就说明 :399-404 docstring 的『duplicate-free』前提被陈旧列打破，会静默丢失一页 state 更新。
- 陈旧尾列是否真的被读写。在 pto_attn.py:774 打印 lblk 的最大值与该请求 num_blocks_per_row（后者需要从 InputBatch 侧取，或用 cdiv(int(cmp_md.seq_lens[0]),8) 近似），确认 ring 的未来 7 行确实落在已分配范围之外。
- 次要但值得顺手看：compare artifact 记录的 cosine=0.36 / max_abs_diff=5.71 却 ok=true（/data/sunkaixuan/skx_log_output/csa_cut_20260920/compare/compare__model_layers_2_self_attn_attn.json）。这不属于第 2 项，但若 state 布局是对的，这个数字说明别处还有问题，建议交给做第 3/4 项的人。

#### UNKNOWNS
- state_c.storage_offset() 的运行时值未实测。代码链（torch.zeros → 单元素 shape list → _adjust_kv_layout 只调用一次、offset 不推进）唯一地给出 0，但 probe artifact 的 describe() 不记录 storage_offset，msprof 也不记录。若开启了 kv_transfer_config（PD 分离），_align_memory 的切片会让 offset 非 0 —— 本配置下未确认该开关的值。
- CANN compressor 算子 tiling 里的 constInfo_.stride 具体取值无法从源码读到：op_host tiling 实现不在树里（只有 op_proto/lib 与 op_api/lib 的二进制）。我按 stateOffset = idInBlockTable*stride 必须等于 32768 个 float 才能与 probe 实测的 stride(0)=32768 自洽来推断；若 tiling 用的是 blockSize*rowdim=16384，native 路径早就错了。属于强推断而非直读。
- pageAttentionParams.blockSize 同理不可直读，由 spec 的 block_size=8 与 block table 宽度 1088=cdiv(8704,8) 反推为 8。
- 块表里被 sliding window 置空的列数、以及尾部陈旧列的具体内容，都是按 remove_skipped_blocks / add_row 的代码推导出来的，没有任何 artifact 记录过一行真实的 state block table 值。(d) 里给的那一行是推导样例，不是实测。
- 131072 这个 bucket 里到底哪几个 layer 共享同一个 KVCacheTensor（full-MLA attn / SWA / 主 state 的具体 shared_by 组合），只读了构造代码 patch_kv_cache_utils.py:232-240，没有 dump 过运行时的 KVCacheConfig.kv_cache_tensors。所以『写错页号会毁另一 layer 的真数据』这一条在机制上成立，但具体是哪个 layer 未确认。
- 第 2(f) 里点出的重复页 index_copy_ 覆盖风险、以及未来 7 行写回陈旧列的风险，都是按代码推导的『可能路径』；在当前 bs=4 / prompt=1024 / max_model_len=8704 下是否真的触发，没有实测。两者今天看起来都是 read-modify-write 恒等操作因而良性，但重复页那条一旦触发就是静默丢更新。
- probe/compare artifact 来自 B=1、prompt 8192 的 run（num_blocks=21777），msprof 来自 bs=4 的 run（num_blocks=34730）。两者 num_blocks 不同属正常（随可用显存变），但本答案里凡涉及 num_blocks 的绝对数字都要按目标 run 重新取。


## 复核补充的细节更正

- SUMMARY BULLET IS WRONG AS WRITTEN (the one precision defect a kernel author could act on): "state_c 实际传给 kernel 的是 [num_blocks, 8, 1, 2048]". [nb,8,1,2048] is what vLLM hands the CONVERSION LAYER (dsa.py:279 tuple index 2) and, after .squeeze(-2), what the NATIVE CANN Compressor op receives ([34730,8,2048], dsa_v1.py:2306, confirmed in prof_native kernel_details.csv). What our registered kernel receives is the private ring: compress_state = [8,2,2048] fp32 contiguous (measured in compare__model_layers_2_self_attn_attn.json), built by make_state_ring at pto_attn.py:776 with layout [b*RR/2, 2, MAIN_STATE_DIM]. The body of the answer gets this right in sections (d)/(e); only the opening summary conflates them. Anyone sizing a kernel argument from the summary alone would size the wrong tensor.
- WRONG ANCHOR for the shared-pool claim, though the conclusion survives. vllm-v0.20.2/vllm/v1/core/kv_cache_utils.py:1285-1288 does contain the quoted text ("As layers of different groups have different block table, they will use different parts of the shared Tensor"), but it sits in the `else:` general-case branch. Lines 1275-1283 route `all(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs))` — i.e. DeepSeek-V4 — to `_get_kv_cache_config_deepseek_v4`, which vllm_ascend replaces at patch/platform/patch_kv_cache_utils.py:249 with its own :184-244. So that comment is NOT the code path for this configuration; only the patch_kv_cache_utils.py:232-240 citation is load-bearing.
- The shared-tensor mechanism is SHARPER than the answer states, and the kernel team should be told the sharper version. model_runner_v1.py:3656-3658 assigns the SAME torch.Tensor object to every layer_name in kv_cache_tensor.shared_by, and _adjust_kv_layout (:3771-3776) then gives each of those layers a view at storage_offset 0 with page stride = page_size_bytes. So layers sharing the 131072-byte bucket (state cache; compress-attn and swa caches, whose probe stride0 of 65536 bf16 = 131072 B confirms the same bucket) are aliased PAGE FOR PAGE onto one allocation. The only thing keeping state page p's upper 64 KB free is global block-id exclusivity: there is a single BlockPool for all groups (vllm/v1/core/kv_cache_coordinator.py:50). State it that way — "padding" is not reserved dead space, it is another group's page-p second half, unowned only because the allocator never hands block id p to two groups at once.
- FORMULA MISSTATED (numerically harmless here). gpu_model_runner.py:6480-6482 is `cdiv(max_model_len, block_size * get_total_cp_world_size())`, and `max_model_len` at :6474 is `max(self.max_model_len, self.max_encoder_len)` — not `cdiv(max_model_len, block_size)` as quoted. With TP=1 / no CP it still gives cdiv(8704,8)=1088, and 1088 is independently MEASURED (probe [1,1088]; msprof `4,1088` INT32), so the number is safe. The formula as written would be wrong on any CP-enabled deployment.
- MSPROF ATTRIBUTION INCOMPLETE, and the omitted part is a cost the requester should see. Per layer per decode step the prof_pto CSV shows THREE rows carrying the base shape 34730,16,2048: two aclnnIndexSelect_SliceAiCore_Slice and one aclnnInplaceIndexCopy_SliceAiCore_Slice. The answer maps them to pto_attn.py:378 and :433 only; the second index_select is write_state_ring's read-modify-write at :426. Each row measures ~3.52-3.59 ms, so the main-state ring conversion alone costs ~10.6 ms per layer per step in the profiled run. The answer never quantifies the conversion layer's price while recommending keeping it.
- MINOR CITATION DRIFT, no effect: real_page_size_bytes in kv_cache_interface.py is the property at :472-485 with the return expression at :480-485, cited as :479-484 (lands inside). decode_metadata.py has two guards the answer did not quote — :228 `csa_state_count >= CSA_STATE_REQUIRED_BLOCKS` before the table read and :239 `csa_state_physical_block >= 0` — neither changes the modulo semantics the answer derives.
- ONE ADDITION TO THEIR LIVE-READ LIST: they propose adding storage_offset()/data_ptr() to describe(). Confirmed describe() at pto_attn.py:626 records none of these — but pto_attn.py:867-869 already maintains `_SEEN_ADDRS[n] = t.data_ptr()`, so a data_ptr audit hook exists and can be extended rather than written from scratch.
- CONFIRMED-AS-HONEST, not a defect: the two items the answer flags as strong inference really are unreadable in-tree. compressor_kernel_perf.h:258 `constInfo.blockSize = tilingData_->pageAttentionParams.blockSize` and :264 `constInfo.stride = tilingData_->baseParams.stride`; the only file matching `maxBlockNumPerBatch` on the host side is the binary op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so. So blockSize=8 and stride=32768 floats genuinely cannot be read, only inferred — exactly as the answer says.

---

# 第 3 项 · inner compressor state 的完整布局

> 复核状态：**复核未通过，已按更正改写**

## 复核的更正（以这一节为准）

- REPLACE the core of (c). The inner compressor state and the indexer k_cache are NOT two allocations. `_get_kv_cache_config_deepseek_v4` buckets every group's layers by page_size_bytes and emits ONE KVCacheTensor per (tuple_idx, page_size), shared_by the union of groups at that slot — /data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-ascend-v0.20.2rc1/vllm_ascend/patch/platform/patch_kv_cache_utils.py:196-199 (docstring) and :232-240 (implementation); the identical function is upstream at /data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-v0.20.2/vllm/v1/core/kv_cache_utils.py:1174-1244, dispatched at :1275-1283. ist_c and idx_k_c therefore share one buffer of num_blocks*16640 B, both mapped at storage_offset 0 by model_runner_v1.py:3647-3658 + :3761.
- REPLACE the core of (b). Do not tell the lib team the trailing 256 B of the inner-state page is a dead zone. Those bytes are the indexer scale (storage_offset_bytes 16384, model_runner_v1.py:3778) for the same physical page. The correct statement: the inner-state *view* covers bytes [0,16384) of each page and never addresses the last 256 B; what occupies those bytes depends on whether that block id is currently owned by the indexer group or the inner-state group. Nothing may be written there by kernel code.
- DELETE the msprof AsStrided-vs-Slice argument as evidence about payload/padding. It shows only that 4160 fp32 is not a multiple of 512, so torch could not express the view as a row Slice. It says nothing about what writes those bytes, and it was used to reach the wrong conclusion.
- REPLACE the 'why 16640' mechanism. It is not `get_uniform_page_size` (kv_cache_utils.py:955) nor `unify_kv_cache_spec_page_size` (:1007-1044) — DSV4 returns earlier at :1642-1649 and those live in the `else` branch at :1284-1295. `unify_kv_cache_spec_page_size` would in fact raise NotImplementedError here (it multiplies block_size and needs 131072 % 16640 == 0). The real mechanism is patch_kv_cache_utils.py:145-156: each SWA-MLA layer's page_size_padded is set by object.__setattr__ to `min(x for x in full_mla_group.get_page_sizes() if x >= current)`, i.e. the nearest-larger page size among the full-MLA group's buckets — which is the indexer k's 16640. The hardcoded 16640 at patch_deepseek_compressor.py:56 is what makes that a no-op.
- CORRECT the quotation from deepseek_compressor.py. The sentence 'compressor states share the same physical tensor as KV blocks, they must use the same page size' is at lines 147-149, not 150-153 (150-153 is the block-shape arithmetic and a TODO). More importantly, that sentence is evidence AGAINST the answer's separate-allocation claim and should be presented as such.
- INVERT the expected outcome of the answer's own live-read #1. The prediction 'inner state 与 indexer k/scale 落在不同 storage，预期 untyped_storage().data_ptr() 不同' is backwards. Predict ist_c.data_ptr() == idx_k_c.data_ptr() and ist_c.untyped_storage().data_ptr() == idx_k_c.untyped_storage().data_ptr(), both storage nbytes == num_blocks*16640. Keep the probe — it is the right experiment — but state the code-derived prediction correctly.
- ADD a live check the answer does not have, and it is the one that matters for silent corruption: print the block-id sets actually in use by the two groups in the same step — idx_md.block_table (indexer group) and ist_md.block_table (inner-state group) at pto_attn.py:774-775 / :760 — and assert their intersection is empty. Aliasing is safe ONLY while those id spaces stay disjoint. Nothing in pto_attn.py checks it.
- FIX line-number errata before sending: page_size_padded is patch_deepseek_compressor.py:56 (not :55); the indexer spec is :84-95 (not :83-93); COMPRESS_STATE_DIM is decode_indexer_compressor.py:58 (not :57); value/score reads are :192-199 (not :190-191); C4A_COMPRESSOR_BLOCK_SIZE is config.py:253 (not :252); CSA_STATE_BLOCKS_PER_REQUEST is config.py:266 (not :262); the inner_compress_state parameter is decode_csa.py:1214 (not :1219); num_element_per_page is model_runner_v1.py:3764-3766 and target_stride :3769 (answer said :3762-3767).
- FIX two dead citations. There is no `_paged_slots` in pto_attn.py — the function is `_state_slots` at :288-314, and `grep -n '_state_slots('` shows it is DEFINED BUT NEVER CALLED. Its docstring is still a correct statement about vLLM's formula, but cite it as an in-repo comment, not as live code. There is also no `_apply_alignment_padding` anywhere under vllm_ascend/ — drop that sentence.
- TIGHTEN the (a) storage_offset claim. storage_offset()==0 is now doubly supported (torch.zeros at model_runner_v1.py:3648-3651 with kv_transfer_config None, plus the fact that the shared-tensor path re-reads raw_tensor.storage_offset() at :3761), but it remains a code inference. Keep it in unknowns until the live print lands, and say so plainly rather than tabling it as a value.
- CORRECT the framing of the opening line. '问者把 16640 的含义搞混了' overstates it. The questioner's guess (16640 = 16384 key + 256 scale) describes the physical page correctly; what they missed is that the inner state is a second interpretation of those same pages, covering only the first 16384 B. Lead with that, because it is the fact a kernel editor needs.
- KEEP as-is, verified verbatim: the clamp_(min=0) finding at deepseek_compressor.py:118 and its effect on state_ring_plan's `ok = ok & (blk >= 0)` at pto_attn.py:369; the unprotected MAIN/INNER coupling via state_ring_len() at pto_attn.py:343-346 reused on the inner path at :777/:436/:1002 with no assert; the two-independent-state_block_table-calls vs one-aliased-slot-mapping distinction at :778-779 vs :781/:804; the dead `_inert_later` at :780; the per-row [kv_state 256 | score_state 256] with 128-halves structure (decode_indexer_compressor.py:39 HEAD_DIM=M.index_head_dim, :49 OUT_DIM, :58, :180-199); and the kernel addressing formula at :184-191. Every one of these I opened and confirmed.

<details>
<summary>原答复（已被上面的更正推翻，保留供对照）</summary>

### 问卷第 3 项 —— inner compressor state（ist_c）的完整布局

结论先行：**问者把 16640 的含义搞混了。inner state 页的 16640 B = 16384 B 有效载荷 + 256 B 纯填充（死区，无人读写）；而 indexer 的 16640 B 才是真正的 16384 B key + 256 B scale 两张量共页。两者页大小相同不是巧合，是 vLLM hybrid allocator 要求全局统一页大小、vllm-ascend 为此把 inner state 的 `page_size_padded` 硬编码成 16640 的结果。**

---

#### (a) ist_c 的父缓冲区：shape / dtype / stride / storage_offset

### 这个张量是怎么来的（读码）

1. `$V/vllm_ascend/ops/dsa.py:285` —— `indexer_state_cache = self.indexer.compressor.state_cache.kv_cache`，再经 `dsa.py:293` 的 `unfold_kvcache` 把单元素 list 拆成张量。它是 `_build_kv_cache` 返回的 6 元组第 4 位（index 3），即 `pto_attn.py:725` 的 `ist_c`。
2. 该 cache 的 spec 由 `$V/vllm_ascend/patch/worker/patch_deepseek_compressor.py:54-66` `AscendCompressorStateCache.get_kv_cache_spec()` 给出：
   - `block_size=8`（由 `$V/vllm_ascend/models/deepseek_v4.py:519-525` 传入，compress_ratio==4 分支）
   - `num_kv_heads=1`，`head_size=state_dim`，`dtype=torch.float32`
   - `state_dim = 2 * coff * head_dim = 2*2*128 = 512`（`deepseek_v4.py:521`，indexer 的 compressor `head_dim=config.index_head_dim=128`，见 `deepseek_v4.py:419` / `:456-463`）
   - **关键行 `patch_deepseek_compressor.py:55`**：
     `page_size_padded = 16640 if self.state_dim == 2 * 256 and self.compress_ratio == 4 else 131072`
     inner state 命中 `state_dim==512==2*256` → **16640**；main state（state_dim=2048）走 else → **131072**。
   - `alignment=None`（`:63`），所以 `_apply_alignment_padding` 不产生任何额外补齐，全部 padding 只来自 `page_size_padded`。
3. 实页字节数：`AscendSlidingWindowMLASpec.real_page_size_bytes`（`$V/vllm_ascend/patch/platform/patch_kv_cache_interface.py`，`storage_block_size = block_size`）= `8 * 1 * 512 * 4 = 16384 B`。
   基类 `AttentionSpec.page_size_bytes`（`$VL/vllm/v1/kv_cache_interface.py:137-150`）有 `assert self.page_size_padded >= real_page_size`，然后**返回 padded 值 16640**。
4. 实际张量在 `$V/vllm_ascend/worker/model_runner_v1.py:3753-3780` `_adjust_kv_layout` 里生成：
   - `num_element_per_page = page_size_bytes // dtype_size = 16640 // 4 = 4160`（`:3762-3764`）
   - `stride = torch.empty([N,8,1,512]).stride() = (4096,512,512,1)`；
     `target_stride = (num_element_per_page, *stride[1:]) = (4160, 512, 512, 1)`（`:3766-3767`）
   - `torch.as_strided(raw.view(float32), size=[N,8,1,512], stride=(4160,512,512,1), storage_offset=storage_offset_bytes//4)`（`:3771-3776`）
   - `storage_offset_bytes = raw_tensor.storage_offset()`（`:3761`）。raw_tensor 来自 `model_runner_v1.py:3648-3651`：`torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=...)`（`use_compress` 分支，`self.use_compress` 在 `:261-263` 因 hf_config 有 `compress_ratios` 而为 True）。本配置 `kv_transfer_config is None`，不走 `_align_memory` 切片，所以 **`storage_offset() == 0` → ist_c 的 storage_offset = 0**。
   - `kv_cache_shape_list = [kv_cache_shape]` 只有一项（`:3822`），因为 `hasattr(spec,"scale_dim")` 对 `AscendSlidingWindowMLASpec` 为 False（scale_dim 只在 `AscendMLAAttentionSpec` 上）。**即这块 allocation 上只映射了这一个视图。**

### 实测（artifact）

- 结构 dump `/data/sunkaixuan/skx_log_output/csa_cut_20260920/probe/model_layers_2_self_attn.json` → `kv_cache[3]`：
  `{"shape": [21777, 8, 1, 512], "dtype": "torch.float32", "contig": false, "stride": [4160, 512, 512, 1]}`
  （该次运行 num_blocks=21777；dump 不含 storage_offset / data_ptr）
- msprof `prof_pto/.../kernel_details.csv`：
  `aclnnIndexSelect_AsStridedAiCore_AsStrided | in "144476800;4;4;1" | out "34730,8,1,512" | FLOAT`
  `aclnnInplaceIndexCopy_ViewCopyAiCore_ViewCopy | "144476800;…;34730,8,1,512;…" | "144476800"`

### 问者的算术确认

`144,476,800 / 34,730 = 4,160` fp32/页 → `4160 × 4 = 16,640 B/页`。**成立**，且与代码完全一致：4160 就是 `_adjust_kv_layout` 的 `num_element_per_page = 16640 // 4`。父缓冲总字节 `34730 × 16640 = 577,907,200 B`（= 144,476,800 fp32）。注意 **34730 不是常量**，它是该显存配比下的 `num_blocks`；同一路径在另一次运行是 21777。

**汇总（本配置）**：
| 项 | 值 |
| --- | --- |
| shape | `[num_blocks, 8, 1, 512]`（8 = block_size 行数，1 = num_kv_heads，512 = state_dim） |
| dtype | `torch.float32` |
| stride | `(4160, 512, 512, 1)` |
| storage_offset | `0`（读码推定，见 live 清单第 1 条） |
| 父 storage | flat int8 `torch.zeros(34730*16640)`，按 float32 view 即 144,476,800 元素 |
| is_contiguous | False（4160 ≠ 8*512） |

---

#### (b) 每 16640 B 页内：前 16384 与后 256 分别是什么

**答案：前 16384 B 是全部有效载荷（8 行 × 512 fp32），后 256 B 是纯 padding，本仓库任何路径都不读也不写。问者把它和 indexer 页的 key/scale 切分弄混了。**

三条独立证据：

1. **读码**：`_adjust_kv_layout` 对该 spec 只收到一个 shape（`model_runner_v1.py:3822`），循环只产生一个张量、`storage_offset_bytes` 不再前进（`:3778` 的累加在单元素列表里无后继消费者）。没有第二个张量被映射到这块 allocation。
2. **语义**：16640 来自字段名就叫 `page_size_padded`，且 `kv_cache_interface.py:147-149` 的断言是 `page_size_padded >= real_page_size`——它是补齐参数，不是"两段载荷之和"。real = 16384。
3. **实测反证（很有力）**：msprof 里 main state 和 indexer 的 padding 都被 torch 分解成 **Slice**，而 inner state 是 **AsStrided**：
   - main state：`aclnnIndexSelect_SliceAiCore_Slice | "34730,16,2048;3;3" | "34730,8,2048"` → 父页按 [16,2048] fp32 看 = 131072 B，取前 8 行 = 65536 B，**后 65536 B 是 padding**。
   - indexer key：`aclnnIndexSelect_SliceAiCore_Slice | "34730,130,128;3;3" | "34730,128,128"` → 父页按 [130,128] int8 看 = 16640 B，取前 128 行 = 16384 B。
   - indexer scale：`aclnnIndexSelect_SliceAiCore_Slice | "34730,8320;2;2" | "34730,128"` → 父页按 8320 个 fp16 看 = 16640 B，取 128 个 fp16 = 256 B。
   - inner state：**无法**表达成干净 Slice，只能 `AsStrided over 144476800`——因为 4160 不是 512 的整数倍（4160/512 = 8.125），尾巴是 **64 个 fp32 = 256 B**，切不出整行。这本身就证明尾部不是"另一个按行排布的张量"。

### 前 16384 B 的内部结构（读码）

8 行 × 512 fp32，每行 512 fp32 = `[kv_state 256 | score_state 256]`，每个 256 又是 `[half0 128 | half1 128]`：
- `$LIB/models/deepseek_v4_flash_dspark/decode_indexer_compressor.py:49` `OUT_DIM = COFF * HEAD_DIM = 2*128 = 256`，`:57` `COMPRESS_STATE_DIM = 2*OUT_DIM = 512`
- `:190-191` 取 value = `compress_state_flat[row, state_half + h0 : …]`，score = `compress_state_flat[row, OUT_DIM + state_half + h0 : …]`，其中 `state_half ∈ {0, HEAD_DIM=128}`（`state_idx >= COMPRESS_RATIO` 时取 128 那半）。
- vLLM 侧同构：`$V/vllm_ascend/models/deepseek_v4.py:521` 注释 `state_dim = 2*coff*head_dim  # kv_state + score_state`。

---

#### (c) 与 indexer key/scale 是否同一种分页格式

**页大小相同（都是 16640 B）是设计上的强制，不是巧合；但分页格式不同。**

### 两处分配的对照（都是读码 + 实测双证）

| | inner compressor state | indexer k_cache / scale |
| --- | --- | --- |
| spec | `AscendSlidingWindowMLASpec`，`patch_deepseek_compressor.py:55-66` | `AscendMLAAttentionSpec`，`patch_deepseek_compressor.py:83-93` |
| block_size | 8（行） | 128（token） |
| 页字节数来源 | `page_size_padded = 16640` 硬编码 | `page_size_bytes` 公式算出：`128*1*(128*1 + 1*2) = 16640`（`patch_kv_cache_interface.py` `AscendMLAAttentionSpec.page_size_bytes` 非 sparse_c8 分支） |
| 页内张量数 | **1 个**（`model_runner_v1.py:3822`） | **2 个**（`model_runner_v1.py:3832-3833`，`[indexer_k_shape, indexer_scale_shape]`） |
| 载荷 | 16384 B fp32 state | 16384 B int8 key（offset 0）+ 256 B fp16 scale（offset 16384） |
| 尾 256 B | **死区 padding** | **scale 张量** |
| 实测 stride | `[4160,512,512,1]` fp32 | k `[16640,128,128,1]` int8；scale `[8320,1,1,1]` fp16 |

- indexer 的 scale 偏移由 `_adjust_kv_layout:3778` `storage_offset_bytes += stride[0]*dtype_size` 得到：k 的 `stride[0]=16384`、dtype_size=1 → scale 的 `storage_offset_bytes = 16384` → fp16 元素偏移 **8192**（`:3774`）。msprof 的 `Slice "34730,8320" → "34730,128"` 与之吻合（8320-8192=128）。
- indexer k 的 dtype 是 **int8**（A2/A3；A5 是 fp8_e4m3fn），见 `$V/vllm_ascend/models/deepseek_v4.py:444`：`k_dtype = torch.float8_e4m3fn if A5 else torch.int8`。这正是 `128*128*1 = 16384` 的来源。
- 两块 allocation 是**两块独立显存**，只是字节数恰好相同：inner state `577,907,200 B`（= 144,476,800 fp32），indexer k+scale `577,907,200 B`（msprof `aclnnIndexPutImpl_ViewCopyAiCore_ViewCopy | "577907200;…"`，scale 侧同一块按 fp16 看是 `288,953,600` 元素）。

### 为什么必须相等

vLLM hybrid allocator 要求同一分组集合内页大小唯一：`$VL/vllm/v1/core/kv_cache_utils.py:950-955` `get_uniform_page_size` 里 `assert len(page_sizes) == 1`，`:1023-1042` 的 unify 逻辑把小页 spec 补齐到 max_page_size。上游 `CompressorStateCache` 的注释也直说了（`$VL/vllm/model_executor/layers/deepseek_compressor.py:150-153`）："compressor states share the same physical tensor as KV blocks, they must use the same page size"。
同理 main state 的 131072 是为了对齐 **bf16 KV 页**：`128 tokens × 1 head × 512 × 2 B = 131072 B`（实测 `SparseAttnSharedkv` 输入 `34730,128,1,512` bf16，probe `kv_cache[0]/[1]` stride `[65536,512,512,1]`）。

**所以：16640 这个数字在 inner state 侧是"被迫跟随 indexer 页大小"的结果，在 indexer 侧是"key+scale 自然算出来"的结果。改 kernel 时必须按各自语义处理，不能套用同一段 unpack 代码。**

---

#### (d) inner state 的 block table：shape / 语义 / 更新规则 / 是否与 main 共用 / 转换层是否别名

这里有两层 block table，必须分开说，问者容易混。

### 第 1 层：vLLM 侧的 state block table（只用于 seed / write-back，不进 kernel）

- 对象：`ist_md.block_table`，来自 `metadata_list[2]`（排序键顺序见 `$V/vllm_ascend/attention/dsa_v1.py:1885` 注释：`[attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]`）。
- **与 main state 的 block table 是不同对象**：`pto_attn.py:774` 用 `cst_md.block_table`（index 1），`:775` 用 `ist_md.block_table`（index 2）。两个 cache group 各自的 metadata。
- shape（实测，prefill probe）：`[num_reqs, 1088]` int32。1088 = `max_model_len 8704 / block_size 8`。对比同一 dump 里 attn/indexer/swa 的 `[1, 68]`（= 8704/128）。
- 语义：**线性、无 ring**。`pto_attn.py:290-313`（`_paged_slots` 的 docstring + 实现）说明 vLLM 侧公式就是普通分页查表 `blk = bt[req, pos // 8]`，没有取模。`state_ring_plan`（`pto_attn.py:349-374`）照此实现：`lblk = pos // VLLM_STATE_PAGE(8)`，`intra = pos - lblk*8`。
- 更新规则：**我们从不写它**。由 vLLM 的 KVCacheManager 按 `SlidingWindowMLASpec(sliding_window = coff*compress_ratio = 8)` 分配/淘汰。`CompressorMetadata` 在 `$VL/vllm/model_executor/layers/deepseek_compressor.py:118` 做了 `block_table_tensor.clamp_(min=0)`，**把 -1 空洞哨兵抹成 0**。⚠️ 这使 `state_ring_plan:369` 的 `ok = ok & (blk >= 0)` 形同虚设——一个被淘汰的逻辑页会表现为"物理页 0"，静默读到别人的 state。这是本项里唯一能造成"跨 decode step 静默污染"的真实风险点，见 live 清单第 3/4 条。

### 第 2 层：kernel 侧的 block table（真正传进算子的那个）

- 形参：`inner_compress_state_block_table: pl.Tensor[[B_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32]`，`decode_csa.py:1215`（注册入口 `decode_csa_attn_tp1_test`，`:1188`）。`INNER_STATE_MAX_BLOCKS = 8`（`decode_csa.py:171`）。
- 由 `pto_attn.py:779` `a["inner_compress_state_block_table"] = state_block_table(b, pos.device)` 构造；`state_block_table` 在 `pto_attn.py:436-441`：`bt[r, k] = r * (16//2) + k = r*8 + k`，shape `[b, 8]` int32。
- 语义（kernel 侧寻址，`$LIB/.../decode_indexer_compressor.py:184-191`）：
  `ring_row = logical_pos % STATE_STORAGE_LEN(16)` → `page_off = ring_row // 2` → `blk = bt[req, page_off]` → `row = blk*2 + ring_row % 2`。
- 我们喂的 ring 是**每请求私有、连续**的：`make_state_ring`（`pto_attn.py:383-393`）产出 `[b*16//2, 2, 512]`，`write_state_ring`（`:395-433`）调用后写回 vLLM cache。为什么不能直接把 vLLM 的表喂进去，`pto_attn.py:316-338` 的注释已经写清楚（非连续 stride + 绝对页号 + null_block 冲突）。

### ⚠️ 转换层的别名情况（问者关心的那处 review 意见）

**两个 block table：不是同一个对象。** `pto_attn.py:778` 与 `:779` 是**两次独立调用** `state_block_table(b, pos.device)`，产生两个不同的 tensor（内容逐位相同）。

**两个 slot mapping：确实是同一个对象。**
- `pto_attn.py:780-781`：
  `a["state_slot_mapping"] = _inert_later = state_slots(pos, ks)`
  `a["inner_state_slot_mapping"] = a["state_slot_mapping"]`
- `pto_attn.py:803-804` 再做一次同样的别名赋值（`_inert` 之后）。

**这在当前实例化下数值正确，但是一处无保护的耦合：**
- 正确性依据：`decode_csa.py:163-164` vs `:170-171`，`MAIN_STATE_STORAGE_LEN == INNER_STATE_STORAGE_LEN == COFF*COMPRESS_RATIO + S == 2*4+8 == 16`，`MAIN_STATE_BLOCK_SIZE == INNER_STATE_BLOCK_SIZE == C4A_COMPRESSOR_BLOCK_SIZE == 2`（`$LIB/.../config.py:252`），`MAIN_STATE_MAX_BLOCKS == INNER_STATE_MAX_BLOCKS == 8`。两侧 block table 又都由同一个 `state_block_table` 生成，所以 kernel 公式 `bt[r,(pos//2)%8]*2 + pos%2` 对两者都化简为 `r*16 + pos%16`，正是 `state_slots` 算的值（`pto_attn.py:443-454` 的注释亦如此声明）。
- 风险：`state_ring_len()`（`pto_attn.py:343-346`）**只返回 `kcsa.MAIN_STATE_STORAGE_LEN`**，却被用在 inner 路径上：`:777` `make_state_ring(ist_c, plan_i, b, inner_dim)`、`:436-441` `state_block_table`、`:1002` `write_state_ring(ist_c, …)`。pto_attn.py 里**没有任何 assert** 确保 `INNER_STATE_STORAGE_LEN == MAIN_STATE_STORAGE_LEN`。一旦 lib 侧把 inner 的 `S`/`COFF`/`COMPRESS_RATIO` 改成与 main 不同（例如 MTP 分支的 DECODE_SEQ 变化），seed/ring/slot/writeback 会同时错位且**不报错**——正是"跨 decode step 静默污染 state"的那类失败。建议在改 kernel 前先加一行断言 `kcsa.INNER_STATE_STORAGE_LEN == kcsa.MAIN_STATE_STORAGE_LEN and kcsa.INNER_STATE_BLOCK_SIZE == kcsa.MAIN_STATE_BLOCK_SIZE`。
- 另：`:780` 的 `_inert_later` 是未被使用的遗留绑定（`:803` 已覆盖），无功能影响。

### 相关尺寸（用于对表）

`CSA_STATE_BLOCKS_PER_REQUEST = (8 + DECODE_SEQ + 2 - 1)//2 = (8+8+1)//2 = 8`（`$LIB/.../config.py:262`），`CSA_INNER_STATE_BLOCKS_PER_REQUEST = CSA_STATE_BLOCKS_PER_REQUEST`（`:270`）。kernel 的 `inner_compress_state` dim0 是动态轴 `INNER_STATE_BLOCK_NUM_DYN`（`decode_csa.py:1219` + `:1242` `bind_dynamic(0, …)`），host 传的是 `b*8` 页（bs=4 → `[32, 2, 512]`），block table 最大索引 31，自洽。

#### FACTS
- 读码 $V/vllm_ascend/ops/dsa.py:285 — ist_c 来自 self.indexer.compressor.state_cache.kv_cache，是 _build_kv_cache 6 元组的 index 3（对应 pto_attn.py:725 的 ist_c）
- 读码 $V/vllm_ascend/patch/worker/patch_deepseek_compressor.py:55 — `page_size_padded = 16640 if self.state_dim == 2 * 256 and self.compress_ratio == 4 else 131072`：inner state（state_dim=512）得 16640，main state（state_dim=2048）得 131072
- 读码 $V/vllm_ascend/models/deepseek_v4.py:519-525 — compress_ratio==4 时 CompressorStateCache(state_dim=2*coff*head_dim, dtype=float32, block_size=8)；indexer 的 compressor head_dim=index_head_dim=128 → state_dim=2*2*128=512
- 读码 $V/vllm_ascend/models/deepseek_v4.py:419 与 :456-463 — indexer.head_dim = config.index_head_dim = 128，其内部 Compressor(head_dim=128, rotate=True)
- 读码 $V/vllm_ascend/patch/platform/patch_kv_cache_interface.py AscendSlidingWindowMLASpec.real_page_size_bytes = storage_block_size*num_kv_heads*head_size*dtype_size = 8*1*512*4 = 16384 B（inner），8*1*2048*4 = 65536 B（main）
- 读码 $VL/vllm/v1/kv_cache_interface.py:137-150 — AttentionSpec.page_size_bytes 在 page_size_padded 非 None 时 assert padded >= real 后直接返回 padded；即 16640 是 padding 上界而非两段载荷之和
- 读码 $V/vllm_ascend/worker/model_runner_v1.py:3762-3776 — num_element_per_page = page_size_bytes//dtype_size = 16640//4 = 4160；target_stride = (4160, 512, 512, 1)；torch.as_strided(..., storage_offset=storage_offset_bytes//dtype_size)
- 读码 $V/vllm_ascend/worker/model_runner_v1.py:3761 + :3648-3651 — storage_offset_bytes = raw_tensor.storage_offset()，而 use_compress 分支的 raw_tensor 是 torch.zeros(size, dtype=int8)（kv_transfer_config 为 None 时不做 _align_memory 切片）→ ist_c 的 storage_offset 推定为 0
- 读码 $V/vllm_ascend/worker/model_runner_v1.py:3822 vs :3832-3833 — state cache 的 kv_cache_shape_list 只有 1 项（SlidingWindowMLASpec 无 scale_dim 字段）；indexer 的是 [indexer_k_shape, indexer_scale_shape] 共 2 项
- 读码 $V/vllm_ascend/worker/model_runner_v1.py:3778 — storage_offset_bytes += stride[0]*dtype_size；indexer k 的 stride[0]=16384、int8 → scale 的 storage_offset_bytes=16384，fp16 元素偏移 8192
- 实测 /data/sunkaixuan/skx_log_output/csa_cut_20260920/probe/model_layers_2_self_attn.json kv_cache[3] = {shape [21777,8,1,512], float32, contig false, stride [4160,512,512,1]}（无 storage_offset/data_ptr）
- 实测 同一 probe kv_cache[4] idx_k = {[21777,128,1,128], int8, stride [16640,128,128,1]}；kv_cache[5] idx_scale = {[21777,128,1,1], float16, stride [8320,1,1,1]} — 两者共用同一 16640 B 页
- 实测 同一 probe kv_cache[2] main state = {[21777,8,1,2048], float32, stride [32768,2048,2048,1]} → 页 131072 B、载荷 65536 B
- 实测 /data/sunkaixuan/skx_log_output/dsv4_vllm/fdoprof/prof_pto/.../kernel_details.csv：aclnnIndexSelect_AsStridedAiCore_AsStrided in "144476800;4;4;1" out "34730,8,1,512" FLOAT — 144476800/34730 = 4160 fp32 = 16640 B/页，父缓冲 577,907,200 B
- 实测 同一 CSV：main state 走 Slice "34730,16,2048"→"34730,8,2048"（131072 B 页取前 65536 B）；indexer key 走 Slice "34730,130,128"→"34730,128,128"（16640 B 页取前 16384 B）；indexer scale 走 Slice "34730,8320"→"34730,128"（16640 B 页取 256 B，偏移 8192 个 fp16）
- 实测 同一 CSV：inner state 唯独无法表达为 Slice、只能 AsStrided over 144476800，因为 4160 不是 512 的整数倍（尾部 64 个 fp32 = 256 B 切不出整行）——反证尾部不是按行排布的第二张量
- 实测 同一 CSV：indexer k 的父缓冲 aclnnIndexPutImpl_ViewCopyAiCore_ViewCopy "577907200;…"、scale 侧 "288953600;…"（fp16 计数）— 即 34730×16640 B 同一块；与 inner state 的 577,907,200 B 字节数相同但是两块独立显存
- 读码 $VL/vllm/v1/core/kv_cache_utils.py:950-955 get_uniform_page_size 内 `assert len(page_sizes) == 1`，:1023-1042 把小页 spec 补齐到 max_page_size — 这是 inner state 被迫 padding 到 16640 的制度性原因
- 读码 $VL/vllm/model_executor/layers/deepseek_compressor.py:150-153 注释明示 compressor state 与 KV block 共用同一物理张量、必须同页大小
- 读码 $V/vllm_ascend/models/deepseek_v4.py:444 — indexer k_dtype = float8_e4m3fn(A5) else int8；A2/A3 上 int8 → indexer 页 = 128*(128*1 + 1*2) = 16640 B
- 读码 $LIB/models/deepseek_v4_flash_dspark/decode_indexer_compressor.py:49,:57 — OUT_DIM=COFF*HEAD_DIM=256，COMPRESS_STATE_DIM=2*OUT_DIM=512；:190-191 value 取 [row, state_half+h0]、score 取 [row, OUT_DIM+state_half+h0]，state_half∈{0,128} → 每行 512 fp32 = [kv_state 256 | score_state 256]，每 256 再分 half0/half1 各 128
- 读码 $LIB/models/deepseek_v4_flash_dspark/decode_indexer_compressor.py:184-191 — kernel 侧寻址 ring_row = logical_pos % STATE_STORAGE_LEN(16)；page_off = ring_row // 2；blk = bt[req,page_off]；row = blk*2 + ring_row%2
- 读码 $V/vllm_ascend/attention/pto_attn.py:774-775 — plan_m 用 cst_md.block_table（metadata index 1），plan_i 用 ist_md.block_table（index 2）：vLLM 侧 main/inner 的 block table 是两个不同对象
- 读码 $V/vllm_ascend/attention/dsa_v1.py:1885 — 5 个 metadata 的排序键顺序为 [attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]，确认 index 2 即 inner state
- 实测 probe model_layers_2_self_attn.json attn_metadata[1] 与 [2] 的 prefill.block_table 均为 {shape [1,1088], int32}（= max_model_len 8704 / block_size 8），而 [0]/[3]/[4] 为 [1,68]（= 8704/128）
- 读码 $V/vllm_ascend/attention/pto_attn.py:778-779 — compress_state_block_table 与 inner_compress_state_block_table 是两次独立的 state_block_table(b, device) 调用，是不同 tensor 对象（内容相同）
- 读码 $V/vllm_ascend/attention/pto_attn.py:781 与 :804 — a["inner_state_slot_mapping"] = a["state_slot_mapping"]，两个 slot mapping 参数确实别名同一个 tensor 对象
- 读码 $V/vllm_ascend/attention/pto_attn.py:436-441 state_block_table 返回 [b,8] int32，bt[r,k] = r*8+k；:443-454 state_slots 返回 r*16 + pos%16，与 kernel 公式 bt[r,(pos//2)%8]*2+pos%2 恒等
- 读码 $V/vllm_ascend/attention/pto_kernels/dspark/decode_csa.py:163-164 vs :170-171 — MAIN_STATE_STORAGE_LEN == INNER_STATE_STORAGE_LEN == COFF*COMPRESS_RATIO+S == 16，MAIN/INNER_STATE_BLOCK_SIZE 同为 C4A_COMPRESSOR_BLOCK_SIZE=2，MAX_BLOCKS 同为 8 — 别名当前数值正确的唯一依据
- 读码 $V/vllm_ascend/attention/pto_attn.py:343-346 state_ring_len() 只返回 kcsa.MAIN_STATE_STORAGE_LEN，却被 :777 / :436 / :1002 的 inner 路径复用，且文件内无任何 assert 保证两者相等 — 无保护耦合
- 读码 $VL/vllm/model_executor/layers/deepseek_compressor.py:118 — CompressorMetadata 构造时做 block_table_tensor.clamp_(min=0)，把 -1 空洞抹成 0，使 pto_attn.py:369 的 `ok = ok & (blk >= 0)` 永不生效；被淘汰的逻辑页会表现为物理页 0
- 读码 $LIB/models/deepseek_v4_flash_dspark/config.py:262,:270 — CSA_STATE_BLOCKS_PER_REQUEST = (8+DECODE_SEQ+2-1)//2 = 8，CSA_INNER_STATE_BLOCKS_PER_REQUEST 与之相同
- 读码 $V/vllm_ascend/attention/pto_kernels/dspark/decode_csa.py:1219,:1242 — 注册入口 decode_csa_attn_tp1_test 的 inner_compress_state 为 pl.InOut[[INNER_STATE_BLOCK_NUM_DYN,2,512] FP32]，dim0 由 bind_dynamic 绑定，host 传 b*8 页
- 读码 $V/vllm_ascend/attention/pto_attn.py:316-338 注释 — 明确记录 vLLM state cache 为非连续（main 65536 B 载荷 / 131072 B stride，inner 16384 B 载荷 / 16640 B stride），这正是本项答案的仓内既有文字依据
- 读码 $V/vllm_ascend/attention/pto_attn.py:63-64 — VLLM_PAGE=128（swa/cmp/indexer KV 页，按 slot），VLLM_STATE_PAGE=8（两个 compressor state cache，按行）
- 读码 $V/vllm_ascend/attention/dsa_v1.py:2548 — 原生路径把 indexer_state_cache.squeeze(-2) 直接交给 torch.ops._C_ascend.compressor，配 state_block_table=ist 的 decode block_table；算子按 strided 描述符寻址，不感知页尾

#### NEEDS_LIVE
- ist_c.storage_offset() 是否真为 0：在 $V/vllm_ascend/attention/pto_attn.py:725（`cmp_kv_c, swa_kv_c, state_c, ist_c, idx_k_c, idx_s_c = kv_cache` 之后）加一次性打印，对 6 个 cache 各打 `t.shape, t.dtype, t.stride(), t.storage_offset(), t.data_ptr(), t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()`。用 PTO_ATTN_PROBE 门控、只打一层一次。这同时回答：inner state 与 indexer k/scale 是否落在不同 storage（预期 untyped_storage().data_ptr() 不同，nbytes 都是 num_blocks*16640）。
- 页尾 256 B 是否真的无人写：同一位置加 `base = ist_c.untyped_storage()`，`tail = torch.as_strided(torch.empty(0, dtype=torch.float32, device=ist_c.device).set_(base, 0, (ist_c.shape[0], 64), (4160, 1), 4096)`，打印 `tail.abs().sum().item()` 与 `tail.count_nonzero().item()`。在 prefill 之后、跑若干 decode step 之后各取一次。若恒为 0，(b) 的『尾部为死 padding』就从读码推定升级为实测。对照组：对 idx_k_c 的同一偏移做同样的 as_strided（预期非 0，因为那是 scale）。
- inner 与 main 的 vLLM block table 是否同一对象、内容是否不同：在 pto_attn.py:774-775 之间打印 `cst_md.block_table.data_ptr(), ist_md.block_table.data_ptr(), cst_md.block_table.shape, ist_md.block_table.shape`，以及 request 0 的 `bt[0, lblk_lo : lblk_hi+1]`，其中 `lblk_lo = (first-8)//8`、`lblk_hi = first//8`（first = pos.view(b,seq)[:,0][0]）。预期两个 data_ptr 不同、两行物理页号不同。
- clamp_(min=0) 造成的 null-block 混淆是否在本负载上真的发生：在 pto_attn.py:775 之后打印 `plan_i` 的 `(blk, ok)`——具体为 `blk.view(b,16)[0]` 与 `valid.view(b,16)[0]`，并统计 `(valid & (blk == 0)).sum()`。任何一个 valid=True 且 blk==0 的条目就是静默读到物理页 0，即跨 step state 污染的直接证据。main/inner 各查一次。
- ring 的跨 step 往返是否无损：在 pto_attn.py:1002（write_state_ring(ist_c, …) 之前）与其后各取一次 `args[ARG_ORDER.index('inner_compress_state')].reshape(-1,512)[state_slots_row]` 的 checksum，并在下一个 decode step 的 make_state_ring 之后（pto_attn.py:777 之后）对同一 ring_row 取 checksum，验证 write→seed 往返逐位一致。这是『state 静默损坏』唯一能端到端证伪的检查。
- 跨 ring wrap（logical_pos % 16 回绕）行为：本次配置 prompt 1024 / max_tokens 32，position 必然多次跨过 16 的倍数。在 pto_attn.py:780 之后打印 `pos[:8]` 与 `a['state_slot_mapping'][:8]`，在 pos%16 从 15 跨到 0 的那一步前后各抓一次，确认 slot 从 r*16+15 正确回绕到 r*16+0 且 seed 取到的是上一步写回的行。
- INNER 与 MAIN state 几何是否仍相等（防回归）：在 pto_attn.py:343-346 附近打印 `kcsa.MAIN_STATE_STORAGE_LEN, kcsa.INNER_STATE_STORAGE_LEN, kcsa.MAIN_STATE_BLOCK_SIZE, kcsa.INNER_STATE_BLOCK_SIZE, kcsa.MAIN_STATE_MAX_BLOCKS, kcsa.INNER_STATE_MAX_BLOCKS`。若要改 kernel，建议直接把它改成 assert——:781/:804 的别名与 state_ring_len() 复用 MAIN 常量的正确性全部依赖这六个值两两相等。
- 本次配置下的 num_blocks 实际值：msprof 记录是 34730、结构 dump 是 21777，两者来自不同运行。若答复要给具体数字，需在 _adjust_kv_layout 之后（model_runner_v1.py:3836 附近）打印 `layer_name, kv_cache_shape_list, page_size_bytes, num_blocks`，取本配置（gpu-memory-utilization 0.60, max_model_len 8704）的那一次。

#### UNKNOWNS
- ist_c 的 storage_offset 与 data_ptr 未被任何 artifact 记录（probe dump 明确不含这两项）。答案中的 storage_offset=0 是从 model_runner_v1.py:3761 + :3648-3651（torch.zeros 新分配、kv_transfer_config 为 None 故不走 _align_memory 切片）推出的，未实测。
- 16640 B 页尾 256 B 恒为 0 / 从不被写，是三条间接证据（单张量映射、page_size_padded 的 assert 语义、msprof 只出现 AsStrided 而非第二个 Slice）的合取，没有直接读过那 256 B 的内容。
- inner state 的 allocation 与 indexer k/scale 的 allocation 是否物理上是两块独立显存，仅由『字节数相同但 msprof 里以不同 dtype/元素数出现』推断；未比对 untyped_storage().data_ptr()。
- vLLM 把这些 spec 分到几个 KV cache group、哪些 layer 共用同一个 KVCacheTensor（kv_cache_tensor.shared_by 的实际内容），未从运行期读出。答案只依赖『每个 layer 各自 _adjust_kv_layout 一次、state spec 只映射一个张量』这一点，不依赖分组细节。
- main state 的 131072 B 页尾 65536 B 同样未实测其内容；由 msprof 的 Slice "34730,16,2048"→"34730,8,2048" 推断为 padding。
- state block table 在 sliding_window=8 下的实际驻留页数与淘汰时机（以及 clamp_(min=0) 抹掉的 -1 是否真会落在 state_ring_plan 需要读的 3 个逻辑页内）未实测，仅指出机制上可能。
- probe artifact 抓到的是 prefill step（decode 为 null），因此 decode 路径的 block_table / slot_mapping 实际形状与取值没有 artifact 佐证，只有 prefill 的 [1,1088]。
- 答案中的『34730 不是常量』已确认，但本任务指定配置（bs=4, max_model_len 8704, gpu-mem-util 0.60）对应的确切 num_blocks 未从该配置的运行里读出。

</details>

---

# 第 4 项 · indexer key/scale 的共享存储契约

> 复核状态：**复核未通过，已按更正改写**

## 复核的更正（以这一节为准）

- FALSE AS STATED — 'model_layers_{0,1,2}...三层一致'. In /data/sunkaixuan/skx_log_output/csa_cut_20260920/probe/model_layers_0_self_attn.json and _1_, kv_cache[2] through kv_cache[5] are all null; only kv_cache[1] is populated. All of kv_cache[3]/[4]/[5] come from model_layers_2 alone. Root cause is a real configuration fact the answer misses: /data/sunkaixuan/skx_log_output/csa_b_tier/probe1/oproj__model_layers_{0,1,2}_self_attn_attn.json report compress_ratio = 0, 0, 4, and $V/vllm_ascend/ops/dsa.py:284 gates indexer_state_cache / indexer_k_cache / indexer_scale_cache on `if self.compress_ratio == 4`. In official-l3 the indexer key/scale contract exists on exactly ONE of the three layers. Restate every (a) measurement as 'model_layers_2 only'.
- REFUTED — (d) '每次调用 ~58 µs 用于取 scale + 写回 scale' and 'key 路径相对好一些…走的是真 gather 而非全表 Slice'. Full aggregation of .../fdoprof/prof_pto/dsv4_mtp_vllm_20260920_192612_dp1_bs4/profile/dp0_pp0_tp0_dcp0_ep0_rank0_3965829_20260920192745483_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv. KEY path per call: aclnnIndexSelect_SliceAiCore_Slice in="34730,130,128;3;3" out="34730,128,128" x8 @ 813.88-818.98 us; aclnnIndexPutImpl_SliceAiCore_Slice same shapes x8 @ 889.18-892.62 us; aclnnIndexPutImpl_ViewCopyAiCore_ViewCopy in="577907200;4;4;1;34730,128,1,128;4;4;1" x8 @ 922.04-931.54 us; aclnnIndexSelect_GatherV3 in="34730,128,1,128;272;1" x8 @ 14.08-14.42 us; aclnnIndexPutImpl_IndexPutV2 in="34730,128,1,128;32,1,128;..." x8 @ 9.92-10.28 us. Total ~2.66 ms. SCALE path per call: IndexSelect_Slice "34730,8320;2;2" x8 @ 28.60-30.32; IndexPutImpl_Slice same x8 @ 28.32-29.94; IndexPutImpl_ViewCopy "288953600;...;34730,128,1,1" x8 @ 22.00-23.16; GatherV3 "34730,128,1,1;272;1" x8 @ 10.66-11.02; IndexPutV2 x8 @ 9.02-9.46. Total ~100 us. The key is ~25x the scale. The whole-page argument is stronger than the answer makes it, but for the opposite tensor.
- WRONG ATTRIBUTION — 'aclnnIndexSelect_AsStridedAiCore_AsStrided ×8' listed under scale movement. That op is x16 with out="34730,8,1,512" and in="144476800;4;4;1" (144476800 = 34730*4160 fp32), i.e. the INNER COMPRESSOR STATE cache (kv_cache[3]), a different allocation. It has nothing to do with indexer scale. Likewise aclnnIndexSelect_SliceAiCore_Slice "34730,16,2048"->"34730,8,2048" x16 @ 3516-3596 us is the MAIN state cache, not the indexer.
- COUNTS WRONG — 'x8' is right only for the specific shape variant, not the op. Actual totals in that csv: IndexSelect_Slice 32 (three shapes), IndexPutImpl_Slice 16 (two shapes), IndexPutImpl_ViewCopy 16 (two shapes), AsStrided 16, IndexPutV2 32 (four shapes), GatherV3 208 (seventeen shapes). Quote op+shape together or the numbers cannot be reproduced.
- STRENGTHEN (d), measured evidence the answer missed — the [N,130,128] page geometry is not merely arithmetic; ACL already materializes it. aclnn*_Slice rows carry Input Shapes "34730,130,128;3;3" -> "34730,128,128", which is the driver's own rendering of the key view's padded base, exactly as "34730,8320" is for the scale. Produced by $V/vllm_ascend/attention/pto_attn.py:602 `cache.index_select(0, src).contiguous()` on the strided key. This is a measured confirmation that the padded page is 130 rows of 128 int8.
- OVER-LISTED AS needs_live #1 (partially) — `storage_bytes == num_blocks * 16640` is already settled by artifact. The same csv shows ViewCopy destinations of 577907200 elements (int8 view) and 288953600 elements (fp16 view) = 34730*16640 bytes and 34730*8320 fp16, i.e. both views span exactly num_blocks pages with no slack. Keep the live read for storage_offset / data_ptr / storage_ptr, drop the nbytes expectation from the unknown column.
- OVER-LISTED AS needs_live #4 — 'confirm native gives key and scale the same block_table object'. dsa_v1.py:2669/2675 are the prefill/decode branches of ONE fetch, and that single `block_table` is passed to torch.ops._C_ascend.npu_quant_lightning_indexer alongside both key=indexer_k_cache (2680) and key_dequant_scale=indexer_scale_cache.squeeze(-2) (2683) in the same call at 2678-2687. One call site settles it; no live read needed. Keep the live print of idx_md.block_table[0,:8] / slot_mapping[:8] for the separate purpose of seeing a real row.
- UNDER-CLAIMED IN (c) — the 'no other layer shares this allocation' point is provable from code, not merely inferred from spec types. model_runner_v1.py:3813-3815 computes num_blocks = kv_tensor.numel() // current_kv_cache_spec.page_size_bytes and asserts it equals kv_cache_config.num_blocks. If the indexer layer's raw tensor also carried the inner-state pages, numel would be ~2x and the assert would fire. The indexer layer's allocation is exactly num_blocks * 16640 bytes and holds nothing else. Move this out of unknowns.
- MINOR — deepseek_v4.py:448 constructs `DeepseekV4IndexerCache`, the upstream class, not `AscendDeepseekV4IndexerCache`. The spec chain only reaches patch_deepseek_compressor.py:84-95 because line 142 of that patch rebinds `deepseek_v4_attention.DeepseekV4IndexerCache = AscendDeepseekV4IndexerCache`. Cite :142 alongside :84-95 or a reader following the constructor lands in vllm-v0.20.2 and sees alignment=576 still live.
- MINOR — the (a) page-size derivation should also cite patch_kv_cache_interface.py:147/151, where MLAAttentionSpec.merge propagates scale_dim and cache_sparse_c8 from specs[0]. That is the second place a future c8 flip would reach the indexer page, and the answer's list of eight breakers only names 4282.

<details>
<summary>原答复（已被上面的更正推翻，保留供对照）</summary>

#### 问卷第 4 项 —— indexer key/scale 的共享存储契约

配置口径：vLLM TP=1、单卡、DSV4 official-l3（3 层、去 MTP）、dtype bf16、quantization ascend、A2/A3 硅、无 PD 分离（kv_transfer_config is None）。

---

### 结论速览

| 子问 | 判定 |
| --- | --- |
| a) 页 16640 B = 16384 key + 256 scale | **确认**（读码 + 实测 stride 双证） |
| b) idx_kv_scale 是整页 FP16 视图，scale 在 8192:8320 | **偏移猜测完全正确**；但 `[34730, 8320]` 不是 vLLM 造出来的张量，是 msprof 对 as_strided base storage 的渲染 |
| c) key/scale 共用一块 allocation、共用一个物理页号 | **确认**（分配侧 + native 读写侧都成立） |
| d) lib 整页吃 `[N,130,128]` INT8 | **算术上可行、且能过 PyPTO 的 contiguous 校验**；但这个布局是分配实现的副产品，**不是算子契约**，有 8 条开关会静默破坏它 |
| e) 两张量的 offset/data_ptr 关系 | 现有 dump **没有** offset/ptr；给出必须活打的清单 |

---

### a) 页布局：确认

**读码链条（每步带 file:line）**

1. indexer cache 的 spec 来源：`$V/vllm_ascend/patch/worker/patch_deepseek_compressor.py:84-95`
   `AscendDeepseekV4IndexerCache.get_kv_cache_spec` 返回 `AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=self.head_dim, dtype=self.dtype, scale_dim=1 if head_dim==128 else 0, scale_dtype=torch.float16)`。
2. head_dim / dtype 来源：`$V/vllm_ascend/models/deepseek_v4.py:443-454`
   `k_dtype = torch.float8_e4m3fn if A5 else torch.int8`；`head_dim=self.head_dim`（=128, IDX_HEAD_DIM）；并且 **只有 `compress_ratio == 4` 才创建 k_cache**（第 446 行）。故本配置 dtype=int8、head_dim=128 → `scale_dim=1`。
3. 页字节数：`$V/vllm_ascend/patch/platform/patch_kv_cache_interface.py:72-76`
   `page_size_bytes = block_size * num_kv_heads * (head_size*sizeof(dtype) + scale_dim*sizeof(scale_dtype))`
   `= 128 * 1 * (128*1 + 1*2) = 128 * 130 = 16640`。
   注意：走的是这条（非 c8）分支，因为 `cache_sparse_c8` 在 DSV4 下恒为 dataclass 默认 `False`（见 d 中第 2 条）。
4. 两个子视图的 shape：`$V/vllm_ascend/worker/model_runner_v1.py:3818-3833`
   `AscendDSABackend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size) = (num_blocks, block_size, num_kv_heads, head_size)`（`$V/vllm_ascend/attention/dsa_v1.py:197-198`）
   → key shape `(N,128,1,128)`，scale shape `(N,128,1,1)`（第 3827-3831 行用 `scale_dim` 顶替 head_size）。
   **顺序固定为 key 在前**：`kv_cache_shape_list = [indexer_k_shape, indexer_scale_shape]`（第 3832 行）。
5. 真正决定页内偏移的是 `_adjust_kv_layout`，`$V/vllm_ascend/worker/model_runner_v1.py:3753-3779`：

```
3761  storage_offset_bytes = raw_tensor.storage_offset()
3764  num_element_per_page = page_size_bytes // dtype_size
3768  stride = torch.empty(shape).stride()          # 该 shape 的连续 stride
3769  target_stride = (num_element_per_page, *stride[1:])
3771  tensor = torch.as_strided(raw_tensor.view(dtype), size=shape,
                                stride=target_stride,
                                storage_offset=storage_offset_bytes // dtype_size)
3778  storage_offset_bytes += stride[0] * dtype_size
```

   - 第 1 轮（key，int8，dtype_size=1）：`num_element_per_page = 16640`，`stride[0] = 128*1*128 = 16384`，offset = O（无 PD 分离时 O=0）→ stride `(16640,128,128,1)`；循环末 `storage_offset_bytes = O + 16384`。
   - 第 2 轮（scale，fp16，dtype_size=2）：`num_element_per_page = 16640//2 = 8320`，`stride[0] = 128`，offset = `(O+16384)//2` → stride `(8320,1,1,1)`。

**实测确认**（artifact `/data/sunkaixuan/skx_log_output/csa_cut_20260920/probe/model_layers_{0,1,2}_self_attn.json`，三层完全一致）：

```
kv_cache[4]  shape [21777,128,1,128]  torch.int8     contig=false  stride [16640,128,128,1]
kv_cache[5]  shape [21777,128,1,1]    torch.float16  contig=false  stride [8320,1,1,1]
```

`16640 B/页`、`16384 B key`、`256 B scale` 全部对上。页内没有填充：16384 + 256 = 16640 正好。key 是 token-major 的 128×128 INT8（内层 stride `[128,128,1]`），scale 是每 token 一个 FP16、128 个连续值。

**一个必须点出来的混淆陷阱**：`patch_deepseek_compressor.py:56` 里那个 `page_size_padded = 16640` **不是 indexer 的**，是 inner compressor state cache 的（`state_dim == 2*256 且 compress_ratio == 4`）。probe 里 `kv_cache[3]`（inner state）是 `[21777,8,1,512] fp32 stride [4160,...]`，4160×4 = 16640 B/页，内容只占 16384 B。两个 16640 同值、不同代码路径、不同 allocation，别把 `patch_deepseek_compressor.py:56` 当成 indexer 的出处。

---

### b) `[34730, 8320]` 的真相 + 8192:8320 猜测

**先纠正前提**：vLLM 从不创建 `[34730, 8320]` FP16 张量。scale 张量的真实 shape 是 `[N,128,1,1]`、stride `[8320,1,1,1]`（probe 实测，见上）。

`[34730, 8320]` 来自 msprof 对 as_strided view 的 **base storage footprint** 的渲染。实测出处：`/data/sunkaixuan/skx_log_output/dsv4_vllm/fdoprof/prof_pto/.../ASCEND_PROFILER_OUTPUT/kernel_details.csv` 第 250 行、第 557 行：

```
aclnnIndexSelect_SliceAiCore_Slice   Input Shapes "34730,8320;2;2"  FLOAT16;INT64;INT64  →  Output "34730,128" FLOAT16
aclnnIndexPutImpl_SliceAiCore_Slice  同上
```

即：34730 页 × 每页 8320 个 FP16 步长，输出把每页的那 128 个 scale 抠出来。`8320 * 2 = 16640` —— 所以“这个视图跨越整页、包含 key 字节”的理解**在语义上是对的**（它就是整页的 FP16 计数），只是它不是一个 vLLM 对象。（34730 vs probe 的 21777 是两次不同 memory 配置下的 num_blocks，见 e 第 5 条。）

**8192:8320 —— 猜测正确，而且是能从代码算出来的，不是巧合。**
由 `_adjust_kv_layout`：scale 的 `storage_offset = (O + 16384) / 2`，页步长 8320 个 fp16。以 raw tensor 按 fp16 看，第 p 页的 scale 起点 = `8192 + p*8320`（O=0 时），长度 128 个 fp16。所以**相对页首的 fp16 下标区间正是 `[8192, 8320)`**，对应字节 `[16384, 16640)`。

关于 O：无 PD 分离时 `tensor = torch.zeros(size, dtype=int8)`，O=0（`model_runner_v1.py:3648-3651`）；开 `kv_transfer_config` 时是 2 MiB 对齐后切片，O≠0（`model_runner_v1.py:3652-3655`）。**相对偏移 +16384 恒成立，绝对基址会挪**——lib 若要自己做 as_strided，必须用活的 `storage_offset()`，不能写 0。

---

### c) 同一 allocation、同一物理页号：确认（两侧都查了，不是只看 shape）

**分配侧（同一块内存）**

`$V/vllm_ascend/worker/model_runner_v1.py:3647-3658`：

```
3647  elif "attn" in layer_name and self.use_compress and layer_name not in kv_cache_raw_tensors:
3649      tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
3658      kv_cache_raw_tensors[layer_name_inner] = tensor
```

`use_compress = hasattr(hf_config, "compress_ratios")`（`model_runner_v1.py:261-263`）在 DSV4 下为 True，且这个分支排在 `"attn" in layer_name ...`（3659）之前。所以 indexer k_cache 这一 layer 拿到的是**一个** flat int8 张量。

随后 `model_runner_v1.py:3811-3841` 把**同一个** `kv_tensor` 交给 `_adjust_kv_layout`，两个 view 都是 `raw_tensor.view(dtype)`（3772 行）。→ key 与 scale 是同一块 allocation 的两个 strided view，不存在两次分配。

**寻址侧（同一页号）**

- native 写：`$V/vllm_ascend/attention/dsa_v1.py:2638 + 2641`（prefill）与 `2647 + 2650`（decode）—— 两次 `npu_scatter_nd_update_v2` 用的是**同一个** `indexer_kv_scale_metadata.{prefill,decode}.slot_mapping`。
- native 读：`dsa_v1.py:2669 / 2675` 取同一个 `block_table`；`dsa_v1.py:2069 / 2359 / 2683` 传 `key_dequant_scale=indexer_scale_cache.squeeze(-2)`。
- 我们的转换层同样只用一套：`$V/vllm_ascend/attention/pto_attn.py:760 和 763` 两次 `repage_kv` 都传 `idx_md.block_table`；`pto_attn.py:795-797` scale 复用 key 的 slot（`pg_ids.track(idx_flat, src)`，注释“The scale is written at the key's slots but takes no mapping of its own”）。

唯一没实测到的一条是“没有**别的** layer 与它共享同一块 allocation”（尤其 `kv_cache[3]` inner state 的页也是 16640 B）。代码上不可能合并（spec 类型不同：`AscendSlidingWindowMLASpec` vs `AscendMLAAttentionSpec`，无法 merge），但这是推断，列入 needs_live。

---

### d) lib 整页吃 `[N,130,128]` INT8：可行，但“保证”不成立

**可行性（vLLM 侧无障碍）**

- 算术：`130*128 = 16640`。行 0..127 = key 的 128 个 token 行；行 128..129 = 256 B = 那 128 个 FP16 scale。
- 构造式：`torch.as_strided(idx_k_c, (N,130,128), (16640,128,1), idx_k_c.storage_offset())`。
- **关键优势**：`(16640,128,1)` 恰好就是 `(N,130,128)` 的连续 stride → `is_contiguous() == True`，能直接过 PyPTO 那三处校验（最内层 `torch_npu_adapter.cpp` 的 `Require(tensor.is_contiguous())`，记录在 `pto_attn.py:321-326`）。这正是当前必须 compact 的原因被消除。

**收益是可量化的（实测）**

`kernel_details.csv`（prof_pto）中 scale 的搬运开销：

- `aclnnIndexSelect_SliceAiCore_Slice`：8 次，28.6–30.3 µs，48 block
- `aclnnIndexPutImpl_SliceAiCore_Slice`：8 次，28.3–29.9 µs，48 block
- 另有 `aclnnIndexSelect_AsStridedAiCore_AsStrided` ×8、`aclnnIndexPutImpl_ViewCopyAiCore_ViewCopy` ×8

即每次调用 ~58 µs 用于“取 scale + 写回 scale”。而且 Slice 的 **Input Shapes 是整张 `34730,8320`**，不是 block_table 命中的那 68 页 —— 每次都要把整张 scale cache 走一遍。key 路径相对好一些（`aclnnIndexSelect_GatherV3AiCore_GatherV3` ×8，输入形状带 `34730,128,1,128`，走的是真 gather 而非全表 Slice）。

**两个前置条件（lib 侧要答）**

1. kernel 当前的 idx 页是 **32 slot**（`$V/vllm_ascend/attention/pto_kernels/dspark/decode_csa.py:1219-1220`，`idx_kv_cache: [IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE(=32), 1, 128] INT8`、`idx_kv_scale: [..., 32, 1, 1] FP32`；compare artifact 实测 `idx_kv_cache [272,32,1,128]`），vLLM 页是 **128 slot**。要直接吃整页，IDX 侧的块大小得改成 128，否则 repage 还是省不掉。
2. 行 128..129 要在 kernel 内把 INT8 bit-cast 成 FP16；而且 kernel 现在声明的 scale 是 **FP32**（`decode_csa.py:1220`），我们在 `pto_attn.py:763-764` 是用 `dtype=torch.float32` 强转过去的。整页方案要求 kernel 改吃 FP16。这两条我无法从 vLLM 侧确认 PyPTO 是否支持，列入 unknowns。

**会静默破坏这个布局的 8 条开关**

1. **`use_sparse`（V3.2 路径）** —— `model_runner_v1.py:325-329`：条件是 `hf_text_config` 有 `index_topk` 且**没有** `compress_ratios`。此时走 3668-3711，`dsa_k_tensor` 与 `dsa_k_scale_tensor` 是**两次独立的 `torch.zeros`**（3706-3711），打包成 4-tuple（3735-3738）。**提问者引用的 3706-3711 确实存在、确实分开分配 —— 但 DSV4 走不到**：`hf_text_config` 有 `compress_ratios` ⇒ `use_sparse=False`；且 3647 的 `use_compress` 分支排在 3659 之前先命中。
2. **`cache_sparse_c8`** —— 只在 `use_sparse` 分支里被赋值（`model_runner_v1.py:4282`），DSV4 下恒为 dataclass 默认 `False`（`patch_kv_cache_interface.py:48`）。所以 `page_size_bytes` 走 72-76 而非 54-70。若哪天 c8 在 compress 路径上被打开，页会变成 `kv_lora+rope+key+scale` 四段合一，整页假设作废。
3. **A5 硅** —— `models/deepseek_v4.py:443-444`，`k_dtype = torch.float8_e4m3fn`。字节布局不变（仍 16640 = 16384+256），但 `[N,130,128] INT8` 的前 128 行是 FP8 不是 INT8。
4. **`head_dim != 128`** —— `patch_deepseek_compressor.py:93` → `scale_dim = 0` → `model_runner_v1.py:3825` 分支不进 → `kv_cache_shape_list` 只有一项，页里只有 key，scale 子视图根本不存在（`_build_kv_cache` 取 `kv_cache[0][1]` 会直接炸，`$V/vllm_ascend/ops/dsa.py:286-289`）。
5. **`compress_ratio != 4`** —— `models/deepseek_v4.py:446` → 根本不创建 indexer k_cache。
6. **PD 分离（`kv_transfer_config` 非 None）** —— `model_runner_v1.py:3652-3655`，2 MiB 对齐后切片，`storage_offset != 0`。相对 +16384 仍成立，但 as_strided 必须用活的 offset。
7. **上游 `alignment=576` 的打包意图** —— `$VL/vllm/model_executor/layers/deepseek_v4_attention.py:1021-1023` 注释明说这是为了让 indexer 页能“pack with the indexer's compressor state cache”。vllm-ascend 当前把它丢掉了（`patch_deepseek_compressor.py:84-95` 不传 `alignment`），所以没打包 —— 而 inner state 的页大小恰好也是 16640（`patch_deepseek_compressor.py:56`），说明打包是设计里留着的。一旦启用，indexer 的页 stride 就不再是 16640，**整页读会静默读到 state 字节**。
8. **最根本的一条：没有任何算子契约钉住这个融合。** native 侧把 key 和 scale 当两个独立张量收（`dsa_v1.py:2069 / 2359 / 2683` 的 `key_dequant_scale=`；写侧 `2638`+`2641` 两次独立 scatter）。融合纯粹是 `_adjust_kv_layout`（`model_runner_v1.py:3753-3779`）这段**分配代码**的产物。改这段不会让任何 native 路径编译失败、不会报错，只会让整页假设悄悄错掉。

→ 给 lib 的建议措辞：**可以按整页实现，但必须在 host 侧加一条运行期断言**（`scale.data_ptr() - key.data_ptr() == 16384 且 key.stride(0) == 16640`），断言失败就退回现在的双张量路径。否则第 7、8 条会以“数值悄悄变坏”的形式出现，不会以崩溃出现。

---

### e) 两张量的 shape/dtype/stride/offset/ptr 关系

**现有 dump 里有的**（`pto_attn.py:626-630` 的 `describe()` 只记 shape / dtype / is_contiguous / stride 四项）：

```
artifact: /data/sunkaixuan/skx_log_output/csa_cut_20260920/probe/model_layers_{0,1,2}_self_attn.json
kv_cache[4] (indexer key)   [21777,128,1,128]  torch.int8     contig=false  stride [16640,128,128,1]
kv_cache[5] (indexer scale) [21777,128,1,1]    torch.float16  contig=false  stride [8320,1,1,1]
（同一文件里 kv_cache[3] inner state [21777,8,1,512] fp32 stride [4160,512,512,1]，页同为 16640 B，另一块 allocation）
```

```
artifact: .../fdoprof/prof_pto/.../kernel_details.csv 第 250/557 行
scale 的 base storage footprint = [34730, 8320] FLOAT16（num_blocks 随 memory 配置变化）
```

`$V/vllm_ascend/attention/pto_kernels/dspark/decode_csa.py` 与 compare artifact 侧（喂给 kernel 的、已 compact 过的）：
```
artifact: /data/sunkaixuan/skx_log_output/csa_cut_20260920/compare/compare__model_layers_2_self_attn_attn.json
idx_kv_cache [272,32,1,128] torch.int8   contig=true
idx_kv_scale [272,32,1,1]   torch.float32 contig=true
```

**缺的、必须活跑打印的**：`storage_offset()`、`data_ptr()`、`untyped_storage().data_ptr()`、`untyped_storage().nbytes()`。清单见 `needs_live`。

**代码推出的期望关系**（活打是为了证伪它）：
```
kv_cache[4].storage_offset() == O                       # 无 PD 分离时 O == 0
kv_cache[5].storage_offset() == (O + 16384) // 2        # O==0 时 == 8192
kv_cache[5].data_ptr() - kv_cache[4].data_ptr() == 16384
kv_cache[4].untyped_storage().data_ptr() == kv_cache[5].untyped_storage().data_ptr()
untyped_storage().nbytes() == num_blocks * 16640        # + 2MiB 若开 kv_transfer
```

#### FACTS
- 读码 | indexer cache 的 spec：block_size=128, num_kv_heads=1, head_size=head_dim, dtype=k_dtype, scale_dim=1 if head_dim==128 else 0, scale_dtype=torch.float16 —— /data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-ascend-v0.20.2rc1/vllm_ascend/patch/worker/patch_deepseek_compressor.py:84-95
- 读码 | k_dtype = float8_e4m3fn if A5 else int8；head_dim=128；且仅 compress_ratio==4 才创建 indexer k_cache —— vllm-ascend-v0.20.2rc1/vllm_ascend/models/deepseek_v4.py:443-454（446 行是 compress_ratio==4 的门）
- 读码 | page_size_bytes = block_size*num_kv_heads*(head_size*sizeof(dtype)+scale_dim*sizeof(scale_dtype)) = 128*1*(128*1+1*2) = 16640 —— vllm_ascend/patch/platform/patch_kv_cache_interface.py:72-76
- 读码 | cache_sparse_c8 在 DSV4 下恒为 dataclass 默认 False（唯一赋值点在 use_sparse 分支内）—— patch_kv_cache_interface.py:48 与 worker/model_runner_v1.py:4282，故 page_size_bytes 走 72-76 而非 54-70 的 c8 分支
- 读码 | AscendDSABackend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size) 原样返回四元组 —— vllm_ascend/attention/dsa_v1.py:197-198
- 读码 | key/scale 的 shape 列表固定为 [key_shape, scale_shape]，scale 用 scale_dim 顶替 head_size —— worker/model_runner_v1.py:3825-3833
- 读码 | _adjust_kv_layout：storage_offset_bytes 从 raw_tensor.storage_offset() 起算，num_element_per_page = page_size_bytes//dtype_size，target_stride=(num_element_per_page,*contig_stride[1:])，每轮末 storage_offset_bytes += stride[0]*dtype_size —— worker/model_runner_v1.py:3753-3779（3761/3764/3769/3771/3778）
- 读码 | 由 3778 推得 scale 的 storage_offset = (O+16384)/2；O=0 时为 8192 个 fp16 元素，页步长 8320 → 相对页首 fp16 下标 [8192,8320)，字节 [16384,16640)。提问者的 8192:8320 猜测正确
- 读码 | use_compress 分支为该 layer 只分配一个 flat int8 张量，两个 view 都是 raw_tensor.view(dtype) —— worker/model_runner_v1.py:3647-3658（3649 无 PD 分离；3652-3655 有 PD 分离时 2MiB 对齐切片使 storage_offset!=0）与 3772
- 读码 | use_compress = hasattr(hf_config,'compress_ratios') —— worker/model_runner_v1.py:261-263；use_sparse 额外要求『没有 compress_ratios』 —— worker/model_runner_v1.py:325-329。故 DSV4 下 use_sparse=False
- 读码 | 提问者引用的分离分配确实存在（dsa_k 与 dsa_k_scale 两次独立 torch.zeros，打包成 4-tuple），但只在 use_sparse 路径上 —— worker/model_runner_v1.py:3696-3711 与 3735-3738；且 3647 的 use_compress 分支排在 3659 之前先命中
- 读码 | native 写侧用同一个 slot_mapping 分两次 npu_scatter_nd_update_v2 写 key 与 scale —— attention/dsa_v1.py:2638+2641（prefill）、2647+2650（decode）；读侧同一 block_table —— dsa_v1.py:2669/2675
- 读码 | native 算子把 scale 当独立张量收（key_dequant_scale=indexer_scale_cache.squeeze(-2)）—— attention/dsa_v1.py:2069 / 2359 / 2683，故『同页』不是算子契约
- 读码 | 转换层两次 repage_kv 都用 idx_md.block_table，scale 复用 key 的 slot 且被强转 FP32 —— attention/pto_attn.py:760、763-764、795-797
- 读码 | PyPTO 绑定要求 contiguous（最内层 torch_npu_adapter.cpp 的 Require(tensor.is_contiguous())）—— attention/pto_attn.py:321-326 的注释记录
- 读码 | kernel 的 idx 页是 32 slot、scale 声明为 FP32 —— attention/pto_kernels/dspark/decode_csa.py:1219-1220（入口 decode_csa_attn_tp1_test 在 1188，bind_dynamic 在 1245-1246）
- 读码 | 上游 alignment=576 的注释意图是让 indexer 页与 indexer 的 compressor state cache 打包；vllm-ascend 的 patch 不传 alignment 所以当前没打包 —— vllm-v0.20.2/vllm/model_executor/layers/deepseek_v4_attention.py:1021-1023 vs patch_deepseek_compressor.py:84-95
- 读码 | patch_deepseek_compressor.py:56 的 page_size_padded=16640 属于 inner compressor state cache（state_dim==512 且 cr==4），与 indexer 同值但是另一条路径、另一块 allocation —— 勿混淆
- 读码 | probe 的 describe() 只记录 shape/dtype/is_contiguous/stride 四项，没有 storage_offset 与 data_ptr —— attention/pto_attn.py:626-630；触发点 ops/dsa.py:209-220
- 实测 | indexer key = [21777,128,1,128] torch.int8 contig=false stride [16640,128,128,1]；indexer scale = [21777,128,1,1] torch.float16 contig=false stride [8320,1,1,1] —— /data/sunkaixuan/skx_log_output/csa_cut_20260920/probe/model_layers_{0,1,2}_self_attn.json 的 kv_cache[4]、kv_cache[5]，三层一致
- 实测 | inner compressor state = [21777,8,1,512] torch.float32 stride [4160,512,512,1]，页同为 16640 B、内容 16384 B —— 同上 json 的 kv_cache[3]
- 实测 | [34730,8320] FP16 是 msprof 对 as_strided base storage 的渲染，不是 vLLM 对象：aclnnIndexSelect_SliceAiCore_Slice 的 Input Shapes 为 "34730,8320;2;2" FLOAT16，Output 为 "34730,128" FLOAT16 —— /data/sunkaixuan/skx_log_output/dsv4_vllm/fdoprof/prof_pto/dsv4_mtp_vllm_20260920_192612_dp1_bs4/profile/dp0_pp0_tp0_dcp0_ep0_rank0_3965829_20260920192745483_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv 第 250、557 行
- 实测 | scale 搬运开销：aclnnIndexSelect_SliceAiCore_Slice 8 次 28.6–30.3 µs、aclnnIndexPutImpl_SliceAiCore_Slice 8 次 28.3–29.9 µs，均 48 block，且输入是整张 34730×8320 而非 block_table 命中的页 —— 同一 kernel_details.csv
- 实测 | key 搬运走的是真 gather：aclnnIndexSelect_GatherV3AiCore_GatherV3 ×8、aclnnIndexPutImpl_IndexPutV2 ×8、ViewCopy ×8，输入形状含 34730,128,1,128 —— 同一 kernel_details.csv
- 实测 | 喂给 kernel 的已 compact 张量：idx_kv_cache [272,32,1,128] torch.int8 contig=true，idx_kv_scale [272,32,1,1] torch.float32 contig=true —— /data/sunkaixuan/skx_log_output/csa_cut_20260920/compare/compare__model_layers_2_self_attn_attn.json 的 arg_shapes
- 实测 | num_blocks 随 memory 配置变化：probe 配置 21777，prof 配置 34730 —— 上述两组 artifact

#### NEEDS_LIVE
- 【最小改动、一次拿全】在 /data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-ascend-v0.20.2rc1/vllm_ascend/attention/pto_attn.py:628-630 的 describe() Tensor 分支里补四个字段："offset": obj.storage_offset(), "ptr": obj.data_ptr(), "storage_ptr": obj.untyped_storage().data_ptr(), "storage_bytes": obj.untyped_storage().nbytes()。触发点现成：ops/dsa.py:209-220（PTO_ATTN_PROBE，每层只跑一次，已在 debug_allowed 门后）。这四个都是 host 侧属性读，不触发 device 同步。期望：kv_cache[5].ptr - kv_cache[4].ptr == 16384；两者 storage_ptr 相等；kv_cache[4].offset==0 且 kv_cache[5].offset==8192；storage_bytes == num_blocks*16640。这一条同时结掉 4a 的页内顺序、4b 的 8192 偏移、4c 的同一 allocation。
- 【非 alias 证明】在同一份 dump 里比较 6 个 cache 的 storage_ptr 是否两两不等，重点是 kv_cache[3]（inner state，页也是 16640 B）与 kv_cache[4]/[5]。现在只有『spec 类型不同（AscendSlidingWindowMLASpec vs AscendMLAAttentionSpec）所以无法 merge 成同一 KVCacheTensor』的推断，没有实测。同一处 describe() 改动即可覆盖。
- 【整页语义的字节级验证，4d 的决定性实验】在 pto_attn.py:763（pg_ids = repage_kv(idx_s_c, ...) 之前）插入，门用现成的 debug_allowed：取一个活的 block id p = int(idx_md.block_table[0,0])；构造 whole = torch.as_strided(idx_k_c, (idx_k_c.shape[0],130,128), (16640,128,1), idx_k_c.storage_offset())；断言 whole.is_contiguous() 为 True；断言 whole[p,128:130,:].reshape(-1).view(torch.float16) 与 idx_s_c[p].reshape(-1) 逐位相等。这一条同时证明『页尾 256 B 是本页的 scale 而非填充』『key 与 scale 共用同一页号』『[N,130,128] 视图确实 contiguous 因而能过 PyPTO 校验』。注意它会读设备数据，必须在非 aclgraph capture 的 eager step 上跑。
- 【block table 实体】打印 idx_md.block_table[0,:8]、idx_md.slot_mapping[:8]，以及 idx_md.block_table.data_ptr()；并在 dsa_v1.py:2669/2675 处打印 indexer_kv_scale_metadata.{prefill,decode}.block_table.data_ptr()，确认 native 给 key 和 scale 用的是同一个 tensor 对象而不只是同值副本。
- 【num_blocks 不可写死】probe 配置下 21777、prof 配置下 34730，由 gpu-memory-utilization 决定。kernel 侧已 bind_dynamic(0, IDX_CACHE_BLOCK_NUM_DYN)（decode_csa.py:1245-1246），需要确认 lib 若改整页方案后 dim0 仍保持 dynamic。
- 【若要启用整页方案，必须加的运行期守卫】在 pto_attn.py 的 _build_args 里（763 行附近）加断言：idx_s_c.data_ptr() - idx_k_c.data_ptr() == 16384 且 idx_k_c.stride(0) == 16640 且 idx_s_c.stride(0) == 8320；失败即退回现有双张量路径。理由是 4d 第 7、8 条（alignment 打包、native 算子不约束融合）失效时不会崩溃，只会数值悄悄变坏。
- 【wrap 行为】本题范围内 indexer cache 不是 ring，无 wrap 语义（ring 只存在于 compressor state，见 pto_attn.py:317-338）。若仍要确认页复用后的一致性，需要跑到 block_table 回收复用同一 p 的场景，再重复上面第 3 条的字节级断言。

#### UNKNOWNS
- PyPTO 是否支持在 kernel 内把 INT8 行 bit-cast 成 FP16（整页 [N,130,128] 方案的前提之一）。我只查了 vLLM/vllm-ascend 侧，PyPTO 语言能力未核。
- kernel 侧能否把 IDX 块大小从 32 改成 128 以直接吃 vLLM 的整页（当前 decode_csa.py:1219-1220 是 32），以及改了以后 IDX_MAX_BLOCKS=8192 的列宽是否仍够用。属于 lib 侧决策。
- kernel 的 idx_kv_scale 声明是 FP32（decode_csa.py:1220），整页方案要求改吃 FP16。是否可行未核。
- 6 个 cache 互不 alias：代码上不可能合并（spec 类型不同），但未实测 storage data_ptr，已列入 needs_live 第 2 条。
- kv_cache_config.kv_cache_tensors[*].shared_by 在本配置下的实际内容（哪些 layer 共享同一块 raw tensor）没有实测，只从 3647-3658 的分配逻辑推断 indexer 独占一块。
- 开启 PD 分离（kv_transfer_config 非 None）时 raw_tensor.storage_offset() 的实际值未实测；代码上 model_runner_v1.py:3770 的 assert 只保证它是偶数（fp16 可整除），相对 +16384 不变。
- A5 硅上（k_dtype=float8_e4m3fn）的实际 probe 未做——本次所有 artifact 都是 A2/A3 口径。字节布局按代码推应完全相同，仅 key 的 dtype 变 FP8。

</details>

---

# 第 5+6 项 · raw/cmp KV 分页语义 + 五类 slot mapping

> 复核状态：已复核通过

### ITEM 5 — raw KV 与 compressed KV 的 128 行分页语义

约定:【读码】= 从源码读出;【实测】= 从 artifact 量出。所有行号为 host 上当前文件的真实行号。

#### 5(a) 两张 block table 的真实形状、来源、一行装什么

**来源(读码)**
- 两张表都由 vLLM 的 per-cache-group `BlockTable` 产出,DSA 侧的页大小固定 128:`vllm_ascend/attention/dsa_v1.py:349` `block_size: int | None = 128`;表宽 `dsa_v1.py:371` `self.max_blocks = (max_model_len + block_size - 1) // block_size` = ceil(8704/128) = **68**。
- 五个 cache group 各有独立的表。raw KV(swa) / compressed KV(cmp) / indexer-k 三组页=128、宽=68;两组 compressor state 页=8、宽=8704/8=**1088**。

**形状(实测)**
- prefill 结构 dump `/data/sunkaixuan/skx_log_output/csa_cut_20260920/probe/model_layers_2_self_attn.json`:metadata[0](cmp)、metadata[3](idx)、metadata[4](swa) 的 `prefill.block_table` = `[1, 68] int32 contiguous stride [68,1]`;metadata[1](主 state)、metadata[2](inner state) = `[1, 1088] int32`。
- native 算子实参 `/data/sunkaixuan/skx_log_output/csa_b_tier/probe1/attn__model_layers_2_self_attn_attn.json`:`ori_block_table [1,68] int32`、`cmp_block_table [1,68] int32`(bs=1 decode, positions=[4097], seqused_kv=[4098])。
- msprof `prof_native/.../kernel_details.csv` 的 `SparseAttnSharedkv` Input Shapes:`"4,64,512;34730,128,1,512;34730,128,1,512;;4,1,512;4,68;4,68;5;..."` → **bs=4 时两张表都是 `[4, 68] int32`**;`QuantLightningIndexer` 的 idx 表同为 `4,68`。

**一行装什么(读码)**
第 r 行 = 第 r 个请求;第 j 列 = 该请求「逻辑流」第 `[128j, 128j+128)` 段所在的**物理页号**(cache dim0 下标)。关键差别在「逻辑流」是什么:
- raw KV(swa):逻辑流 = 绝对 token position。
- compressed KV(cmp) 与 indexer-k:逻辑流 = **压缩行号 `r_c = pos // 4`**,不是 token position。证据链:`vllm_ascend/utils.py:1521-1587 get_compressed_pos_and_indices` 对 `compress_ratio>1` 的组直接生成压缩行号 `compressed_pos_ids`(`utils.py:1567-1581`,`compressed_historical_len = num_computed // ratio`),再由 `vllm_ascend/worker/model_runner_v1.py:1072-1077` 的 `block_table.compute_slot_mapping(..., positions_compressed_list, req_indices_compressed_list, ...)` 按同一 128 页分页。顺序是请求主序、位置升序(`np.repeat(arrange_np, num_new)` @ `utils.py:1583`)。

**未分配列的填充值(读码,重要)**
不是 -1,是 0 或陈旧值。`model_runner_v1.py:2793-2807`:compress 模型的正常步只对 **slot_mapping** 填 -1(`:2794-2796`),**block table 不动**;只有 dummy/graph-capture 分支 `:2803-2804` 会 `slot_mapping[:].fill_(0)` + `blk_table_tensor.fill_(0)`,非 compress 分支 `:2806-2807` 才 `blk_table[num_reqs:].fill_(0)`。另有 `dsa_v1.py:971` 对 `num_reqs_actual` 之后的行 `fill_(0)`。vLLM 的 block 0 是 null block(`vllm/v1/core/block_pool.py:176-177` `self.null_block = self.free_block_queue.popleft(); self.null_block.is_null = True`),所以 0 = 空洞。**结论:block table 里没有负数语义**,pto_attn 里所有 `blk >= 0` 的判断(`pto_attn.py:260`、`:311`、`:369`)实际从不生效。

**未知**:两张表的**具体数值**在任何 artifact 里都不存在——`pto_attn.py:230-231` 的 `describe()` 只记录 shape/dtype/contig/stride。见 needs_live #1。

#### 5(b) 一个物理页号能否直接展成四个 32 行子页:能,且零成本

`pto_attn.py:596-599`(repage_kv 的 contiguous 分支):
```python
if cache.is_contiguous() and dtype in (None, cache.dtype):
    view = cache.view(-1, rows, *cache.shape[2:])          # rows = 128//4 = 32
    bt   = _repage_block_table(block_table, want, per, view.shape[0])
    return Paged(view, _pad_cols(bt, kernel_cols), block_table)
```
`view()` 是纯 reshape(`21777×128 → 87108×32` 那一类),不产生任何拷贝,`Paged._cache` 为 None → `compacted=False` → `remap` 恒等(`:540-541`)、`commit` 直接 return(`:546`)。

#### 5(c) 精确公式

`pto_attn.py:225-238`:
```python
per    = COMPRESS_RATIO = 4                       # pto_attn.py:65
rows   = VLLM_PAGE // per = 32                    # pto_attn.py:63, 593
want   = min(kernel_cols, ncols * per)            # pto_attn.py:594  = min(8192, 68*4) = 272
j      = arange(want)                             # :232
src    = clamp(j // per, max = ncols - 1)         # :233
bt_new[i, j] = clamp(bt_vllm[i, src[j]] * per + (j % per), 0, n_blocks - 1)   # :234-238
n_blocks = view.shape[0] = cache.shape[0] * 4
```
再经 `_pad_cols(bt_new, CMP_MAX_BLOCKS=8192)`(`:612-621`):`have == cols` 原样返回(`:615-616`);`have > cols` **截断** `bt[:, :cols]`(`:617-618`);`have < cols` 右侧补 `cols-have` 列 **-1**(`:619-620`)。本配置 272 < 8192 → 第 272..8191 列全是 -1。

**正确性(我推导并验证)**:kernel 行 = `bt_new[i, r//32]*32 + r%32`。令 j = r//32,则 j//4 = r//128、j%4 = (r%128)//32,代入得
`= bt_vllm[i, r//128]*128 + ((r%128)//32)*32 + r%32 = bt_vllm[i, r//128]*128 + r%128`
= vLLM 自己的 flat 行。**恒等,零误差**。前提只有两条:32 整除 128,且 cache 是 row-major contiguous(5(d) 已确认)。

**宽度为何必须是编译期常量**:kernel 签名 `pto_kernels/dspark/decode_csa.py:1218` `cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32]`、`:1221` idx 同理;只有 dim0 `bind_dynamic(0, B_DYN)`(`:1257-1258`),dim1 没有 dynamic 轴。`CMP_MAX_BLOCKS = IDX_MAX_BLOCKS = ceil((MAX_SEQ_LEN//4)/32) = 1048576/128 = 8192`(`decode_csa.py:177-179`,`MAX_SEQ_LEN = FLASH.max_position_embeddings = 1048576`)。host 侧再加一条约束:`pto_attn.py:228-229` 的注释说明,若从表内容推导宽度需要 device→host 读,ACLGraph capture 明令禁止(同文件 `:8-10`、`:475-490`)。

**-1 填充列会不会被解引用**:本配置不会,但 kernel 侧**没有 `>= 0` 保护**。
- indexer 读列上界 = `((cache_len-1)//32)`,`cache_len = kv_seq_lens[b] // COMPRESS_RATIO`(`decode_indexer.py:197`、`:324`);实测 kv_seq_lens 是 token 数(probe1 seqused_kv=4098 @ pos 4097),本工况 ≈1056 → cache_len=264 → 最大列 8。
- sparse attn 读列 = `qk_ridx // BLOCK_SIZE`(`decode_sparse_attn_csa.py:289`),`qk_ridx` 来自 `cmp_sparse_indices`,已被 position 上界裁剪(`:144-152`)。
- 两处都是裸 `pl.read(block_table, [b, col])` 后 `*BLOCK_SIZE`,读到 -1 会得到 -32 行。安全性完全依赖 `max_model_len(8704)/4/32 = 68 << 272`。

#### 5(d) 页内 padding / 交织 / 非常规 stride,以及零搬运的证据

**raw KV 与 compressed KV:完全没有 padding、没有交织、标准 stride(实测)**
`csa_cut_20260920/probe/model_layers_2_self_attn.json` 的 kv_cache 六元组:
```
[0] cmp_kv  [21777,128,1,512] bf16  contig=true  stride [65536,512,512,1]   # 65536 = 128*512
[1] swa_kv  [21777,128,1,512] bf16  contig=true  stride [65536,512,512,1]
[2] state   [21777,  8,1,2048] fp32 contig=false stride [32768,2048,2048,1] # 内容 65536B / 步长 131072B
[3] inner   [21777,  8,1, 512] fp32 contig=false stride [ 4160, 512, 512,1] # 内容 16384B / 步长 16640B
[4] idx_k   [21777,128,1, 128] int8 contig=false stride [16640,128,128,1]   # 内容 16384B / 步长 16640B
[5] idx_sc  [21777,128,1,   1] fp16 contig=false stride [ 8320,  1,  1,1]   # 8320*2 = 16640B
```
[0]/[1] 的 dim0 stride 恰等于 128×512 → 严格 row-major,无页尾填充。这就是 `repage_kv` 走 `cache.is_contiguous()` 分支(`pto_attn.py:596`)的**代码原因**。
反例证实交织确实存在于别处:[4] 与 [5] 的页步长都是 16640 B,而 key 内容只占 16384 B,scale 占 128×2=256 B,16384+256=16640 —— **key 和它的 scale 交织在同一个 16640 B 页里**(`pto_attn.py:584-588` 的 docstring 正是这么写的)。所以这两块走 compact 分支 `:601-609`(`index_select` + `.contiguous()`,scale 还要 `to(float32)`,`:603-604`),必然付出一次拷贝。

**零搬运的实测确认**(`prof_pto/.../ASCEND_PROFILER_OUTPUT/kernel_details.csv`,本 run cache 页数 34730):
- idx 压缩确实发生:`GatherV3 "34730,128,1,128;272;1" -> "272,128,1,128"` INT8 14.25 µs ×8;`GatherV3 "34730,128,1,1;272;1" -> "272,128,1,1"` FP16 10.89 µs ×8。272 = b(4)×ncols(68),与 `:605` 的 `view(b*ncols*per, 32, ...)` 完全吻合。
- idx 回写确实发生:`IndexPutV2 "34730,128,1,128;32,1,128;..."`、`IndexPutV2 "34730,128,1,1;32,1,1;..."`(32 = T = 4×8)。
- **对形状 `34730,128,1,512` 的两块 bf16 cache,全表没有任何 Gather / Index / Slice / Copy 类算子**;该形状只出现在 native 层的 `ScatterNdUpdateV2`(n=16 = 2 个 native 层 × 8 步)和 `SparseAttnSharedkv` 里。→ raw KV 与 compressed KV 的零搬运被 trace 直接证实。

**顺带一条实测(不属于这两块,但 trace 里最贵)**:conversion 层引入了 native profile 里**一条都没有**的巨型拷贝——`ViewCopy "1138032640;...;34730,8,1,2048"` 3600 µs ×8、`ViewCopy "...;34730,8,1,512"` 927 µs ×8、`ViewCopy "...;34730,128,1,128"` 926 µs ×8、`Slice "34730,16,2048"->"34730,8,2048"` 3549 µs ×24、`AsStrided "144476800"->"34730,8,1,512"` 966 µs ×24。来源是 strided state cache 的 `index_copy_` 回写与 strided view 构造。

**一条 lib 侧必须知道的读法约束(读码)**:`decode_sparse_attn_csa.py:271-282` 的 SWA 取数分支**不是逐列读** `window_swa_indices`。它按 32 行对齐切 run(`qk_head = (qk_win_start + qk_s0) % BLOCK_SIZE` @ `:271`,`qk_lo/qk_hi` @ `:274-275`),每个 run 只读**一个**索引 `pl.read(window_swa_indices, [qk_t, qk_s0+qk_lane_kv+qk_lo])`(`:277`),然后 `pl.gather_row(..., valid_shape=[qk_hi-qk_lo, HEAD_DIM])`(`:279-282`) 连读最多 32 行。即它要求「同一个 32 对齐 run 内的物理行必须连续」。因为 32 | 128 且我们只做 view,该假设恒成立;任何把 raw KV 换成非仿射 remap 的改动都会静默破坏它。

**另注**:`pg_swa.table` 在 `pto_attn.py:758` 被算出来但**从未传给 kernel**——entry `decode_csa_attn_tp1_test`(`decode_csa.py:1188-1235`) 根本没有 `ori_block_table` 形参,raw KV 的分页只通过 `ori_slot_mapping` 和 `window_swa_indices` 进入 kernel。这是每层每步一次白白做掉的 `[4,8192]` gather+cat。

---

### ITEM 6 — 五类 slot mapping 的准确语义

#### 共同前提(读码)

1. 五个都是 `[T_DYN] INT64`(`decode_csa.py:1222-1227`),T = n_real × `kcsa.S` = n_real×8。
2. **矩形**:一个 host token 占 slot `r*S`,其余 S-1 个 lane 是 padding,靠写 -1 让它们惰性(`pto_attn.py:692-710 rectangular`,`:784-785 _inert`)。同一请求的 S 个 lane 的 `x_normed` 和 `position_ids` 是**同一行的复制**(`:737`、`:740`),只有 slot mapping 不同。
3. kernel 对五个全部有 `>= 0` 保护,**没有任何别的负值语义**,`< 0` 一律等价于「跳过这一格」:
   - ori → `decode_csa.py:958-961`
   - cmp → `decode_compressor_ratio4.py:354-360`
   - state → `decode_compressor_ratio4.py:299-307`
   - idx → `decode_indexer_compressor.py:393-397` 与 `:410-417`
   - inner_state → `decode_indexer_compressor.py:234-244`
4. **-1 是怎么来的(这条是隐性且 load-bearing 的)**:`model_runner_v1.py:2794-2796` 对 compress 模型 `slot_mapping[total_compressed[gid] : num_tokens_padded].fill_(-1)`;`dsa_v1.py:563-564` 再 `torch.stack([sm // 128, sm % 128], -1)` → -1 变成 **(-1, 127)**(floor 除 + 正余数)。`_flat_slots`(`pto_attn.py:222`)`sm[:,0]*128 + sm[:,1] = -128+127 = -1`,**精确还原**。所以 `_flat_slots` 对真实条目和 padding 哨兵都是 vLLM flat slot 的严格逆。
5. **capture 那一趟不同**:`model_runner_v1.py:2803-2804`(dummy/graph-capture 分支)把 slot_mapping 和 block table **全填 0**。capture 只录算子不录值、replay 读活 buffer,所以数值无害;但录制那一趟确实会往每个 cache 的第 0 行写。

#### (1) ori_slot_mapping

- **值的性质**:`block * page_size + offset`,且因为 kernel 的 32 行页只是 reshape,这个数字同时就是**物理 flat 行号**。三种说法在这里是同一个数:`ori_bt[req, pos//128]*128 + pos%128` = `page32*32 + pos%32`。
- **构造**:`pto_attn.py:787-788` `_inert(_flat_slots(swa_md.slot_mapping, 128).index_select(0, src*seq))` → `:793` `pg_swa.remap(...)`,swa contiguous ⇒ remap 恒等(`:540-541`)。
- **-1**:padding lane(`t % S != 0`)写 -1;vLLM 自己 padding 的 token 也是 -1(经 (-1,127) 还原)。kernel 在 `decode_csa.py:959` 跳过。**不存在 -1 以外的负数。**
- **跨页实例**(bs=4, prompt=1024, 首个 decode, pos=1024):1024//128 = 8、1024%128 = 0 —— 正好是新页第一行。设该请求 `swa_bt = [..., p7, p8, ...]`,则 `ori_slot_mapping[lane0] = p8*128 + 0`,而上一 token(pos=1023)在 `p7*128 + 127`,物理上不相邻。kernel 的 writeback 是单行写(`decode_csa.py:961`),不受影响。
- **地址换算**:`kv_cache_flat = reshape(kv_cache, [ORI_BLOCK_NUM*32, 512])`(`decode_csa.py:951`),写 `kv_cache_flat[ori_slot_mapping[t]]`。vLLM 地址 → 目标地址是 **identity**。

#### (2) cmp_slot_mapping

- **值的性质**:同样是 `block*128 + offset`,但**逻辑流是压缩行号 `r_c = pos // 4`**。值 = `cmp_bt[req, r_c//128]*128 + r_c%128`;传给 kernel 时不变(cmp cache contiguous ⇒ remap 恒等)。
- **打包→按 token 展开**:vLLM 给的是**按压缩行打包**的数组(`utils.py:1567-1586` 生成压缩行号;`dsa_v1.py:963` `slot_mapping = self.slot_mapping[:compressed_tokens_start]`,`compressed_tokens_start` = 本步 boundary token 数,`dsa_v1.py:945-951`)。host 用 `_compressed_rows`(`pto_attn.py:277-280`)的 `row = cumsum(boundary)-1` 把它展回每 token,`_to_token_rows`(`:284-285`)对非 boundary 填 -1。
- **-1**:非 boundary token((pos+1)%4 ≠ 0)→ -1;padding lane → -1。kernel `decode_compressor_ratio4.py:355` 跳过。
- **跨页实例**:pos=1023 是 boundary(1024%4==0),`r_c = 255` → vLLM 逻辑页 1、页内 127 → slot = `cmp_bt[req,1]*128 + 127`。kernel 页 = 255//32 = 7 → `bt_new[7] = cmp_bt[1]*4 + 3` → kernel 行 = `(cmp_bt[1]*4+3)*32 + 31 = cmp_bt[1]*128 + 127` ✔。下一个 boundary pos=1027,`r_c = 256` → vLLM 逻辑页 **2**、页内 0 → `cmp_bt[req,2]*128 + 0`;kernel 页 8 → `bt_new[8] = cmp_bt[2]*4` → `cmp_bt[2]*128` ✔。连续两个压缩行跨了一个 vLLM 物理页,两侧公式各自给出正确且相同的物理行。
- **写入端**:`cmp_kv_cache_flat[cache_row]`(`decode_compressor_ratio4.py:356-360`)。
- ⚠ **本项有一个真实缺陷,见下面「必须上报的缺陷」。**

#### (3) idx_slot_mapping

- **值的性质:既不是 vLLM 的物理行,也不是 vLLM block 表下的地址。** vLLM 原值与 cmp 同构(`idx_bt[req, r_c//128]*128 + r_c%128`),但 idx_k cache 是 strided ⇒ `repage_kv` 走 compact 分支,整块搬进私有 buffer,于是 `Paged.remap`/`_located`(`pto_attn.py:517-532`、`:537-543`)把值改写为私有 buffer 里的行:
  ```
  moved = (req * ncols + col) * 128 + intra          # pto_attn.py:132(方法内) → 文件 :532
  col   = argmax_j ( idx_bt[req, j] == 原 physical block )   # :529-530
  ncols = idx_bt.shape[1] = 68
  找不到 或 原值 < 0 → -1                                     # :531, :543
  ```
  `col` 是**搜出来的**而非由 `r_c//128` 推出来的(`:520-522` 的注释明说),所以它容忍 vLLM 任何 slot 分配规则。
- **具体差异**:req=1、原 physical block = `idx_bt[1][2]`、intra=0 → kernel 看到 `(1*68+2)*128 = 8960`,而 vLLM 行是 `idx_bt[1][2]*128`。**两个数完全不同,lib 侧不能把 idx_slot_mapping 反解成 vLLM 地址。**
- **-1**:padding lane、非 boundary、以及 `_located` 未命中,三种情况都给 -1。
- **scale 没有自己的 mapping**:`pg_ids.track(idx_flat, src)`(`pto_attn.py:797`) 只登记不换算;kernel 用同一个 `idx_slot_mapping` 行写 scale(`decode_indexer_compressor.py:410-417`)。
- **回写**:`Paged.commit`(`pto_attn.py:545-578`) 把私有 buffer 行写回 vLLM cache;未写的行 park 在 cache 第 0 行,靠 ok mask 让 scatter 成恒等(`:552-554` 注释 + `:171-178` 的 parked 逻辑)。实测即 `IndexPutV2`。
- ⚠ **kernel 侧一条必须告诉 lib 的额外约束**:`decode_indexer_compressor.py:388-397` / `:406-417` **不按 token 遍历**,而是按压缩行 `compact_token` 遍历,再反推 token:
  ```
  request = compact_token // (S // COMPRESS_RATIO)
  first_pos = position_ids[request * S]
  token = compact_token*4 + COMPRESS_RATIO - 1 - first_pos % 4      # :392, :409
  ```
  它假设「每请求 S 个 token 里恰有 S/4 个 boundary,位置由 `first_pos%4` 决定」。我们的矩形只有 lane 0 是真 token:当 pos%4==3(唯一会写 idx cache 的情形)时 `first_pos%4=3` → token = `compact_token*4 + 0`,对 `compact_token = r*2` 恰落到 lane 0;`compact_token = r*2+1` 落到 lane 4(-1,被跳过)。**当前恰好自洽**,但依赖 `S % 4 == 0` 且「lane 0 承载真 token」,任一方变动都会静默错位。

#### (4) state_slot_mapping

- **值的性质:既不是 vLLM 行,也不是 vLLM block 表地址。** 它是 host 自建的**私有 per-request ring 的 flat 行号**:
  `state_slot = req * MAIN_STATE_STORAGE_LEN + pos % MAIN_STATE_STORAGE_LEN = req*16 + pos%16`(`pto_attn.py:443-454`)。
- **两端一致性(已逐行核对,结论:恒等)**
  - 写端:`decode_compressor_ratio4.py:299-307` 直接把它当 `compress_state_flat` 的行下标,且**对每个 token 无条件写**(没有 boundary 判断)——所以 state mapping 是 per-token 的,与 cmp/idx 的 per-boundary 不同。
  - 读端:`decode_compressor_ratio4.py:184-207`——`ring_row = logical_pos % STATE_STORAGE_LEN`(`:192`)、`page_off = ring_row // 2`(`:193`)、`blk = compress_state_block_table[c_idx, page_off]`(`:194-195`,带 `blk >= 0` 保护 `:196`)、`state_row = blk*2 + ring_row%2`(`:199`)。
  - host 的 ring 表 `state_block_table`(`pto_attn.py:436-440`)给出 `bt[r,k] = r*8 + k`,代入读端得 `row = r*16 + logical_pos%16` —— 与写端公式完全一致。`pto_attn.py:446-447` docstring 的「`bt[r,(pos//2)%8]*2 + pos%2` reduces to `r*16 + pos%16`」**属实**。
  - 与 lib 自家参考实现对齐:`decode_metadata.py:225-248`(`csa_state_logical = pos // C4A_COMPRESSOR_BLOCK_SIZE`;`bt[req, logical % CSA_STATE_REQUIRED_BLOCKS]`;`slot = blk*2 + pos%2`;`blk<0` 或 count 不足 → -1)。host 把 `% 8` 折进了 ring 表,等价。
  - 几何常数核对:`CSA_STATE_BLOCKS_PER_REQUEST = (8 + DECODE_SEQ + 2 - 1)//2 = 8`(`config.py:272`),`MAIN_STATE_STORAGE_LEN = MAIN_STATE_LEN(8) + S(8) = 16`、`MAIN_STATE_MAX_BLOCKS = 8`(`decode_csa.py:163-164`)。与 host 的 `state_ring_len()`(`pto_attn.py:343-346`)一致。
- **-1**:只有 padding lane(`pto_attn.py:803` 的 `_inert`)。**vLLM 的 state block table 永远不会给负数**(用 0 表示空洞,block 0 是 null block),所以 `state_ring_plan:369` 的 `ok & (blk >= 0)` 实际从不生效。
- **跨页实例**(pos=1027, ks=8):`first = 1027`,plan 位置 = `first - (16-8) + i` = **1019..1034**(`pto_attn.py:362-363`)。
  - ring 行 = `p % 16`:1019→11, 1020→12, 1021→13, 1022→14, 1023→15, 1024→0, …, 1034→10 —— 恰好覆盖 0..15 各一次(无碰撞,因为 16 个位置连续)。
  - vLLM 侧 page = `p // 8`:1019→127(intra 3)、1023→127(intra 7)、1024→**128**(intra 0) —— 正好跨 vLLM state 页。plan 之所以把 (blk, intra) 分开保存就是为此(`:365-368`),并且 `:356-358` 的注释说明 cache dim0 是 strided、不能先展平成行。
  - 写回 `write_state_ring`(`:395-434`)按 `span = 16/8 + 1 = 3` 个连续逻辑页聚合,避免 `index_copy_` 的重复页下标互相覆盖(`:401-405` 注释)。
- **地址换算(vLLM ↔ 目标,两步而非一步)**
  ```
  seed : ring[req*16 + p%16] = vllm_state_cache[ state_bt[req, p//8], p%8 ]     # :349-373, :383-392
  write: vllm_state_cache[ state_bt[req, p//8], p%8 ] = ring[req*16 + p%16]     # :395-434
  p ∈ [first-8, first+7]
  ```
- **隐患(需实测)**:span 的最后一页(本例逻辑页 129 = 位置 1032..1039)在 seq_len=1028 时**尚未分配**,block table 该列给 0 → plan 判 valid → 回写会把「从 page 0 读出来的原样内容」再写回 page 0。目前无害(block 0 是 null block,不属于任何请求),但一旦 vLLM 改 null-block 约定、或该列残留别的请求的旧页号,就会**静默覆盖别人的 state**。见 needs_live #6。

#### (5) inner_state_slot_mapping

- **它和 state_slot_mapping 是同一个 Python 对象——claim 属实。** `pto_attn.py:781` `a["inner_state_slot_mapping"] = a["state_slot_mapping"]`,`:804` 再赋一次同样的语句。同一 device tensor、同一 data_ptr。
- **数值上目前正确**:`INNER_STATE_STORAGE_LEN = INNER_STATE_LEN + S = COFF*COMPRESS_RATIO + S = 8 + 8 = 16` = `MAIN_STATE_STORAGE_LEN`;`INNER_STATE_BLOCK_SIZE = MAIN_STATE_BLOCK_SIZE = 2`;`INNER_STATE_MAX_BLOCKS = MAIN_STATE_MAX_BLOCKS = 8`(`decode_csa.py:160-171`)。两个 ring 几何完全相同,所以共用一张 slot 表成立。
- **对 lib 侧的含义(两条)**
  1. **不能假设这两个指针不同**。目前安全是因为 kernel 只读不写(签名是 `pl.Tensor` 不是 `pl.InOut`,`decode_csa.py:1226-1227`);任何「就地改写 state_slot_mapping」的 kernel 写法会同时改掉 inner 的。
  2. **host 侧的耦合是硬编码且无断言的**:`state_ring_len()`(`pto_attn.py:343-346`)只返回 `MAIN_STATE_STORAGE_LEN`,`make_state_ring`(`:383-392`)和 `state_block_table`(`:436-440`)也都用它去构造 **inner** 的 ring 与表。一旦 lib 让 inner 的 `STATE_LEN` / `BLOCK_SIZE` / `MAX_BLOCKS` 与 main 分叉,**不会报错**(dim0 是 dynamic 轴,形状仍然通得过),会静默错位。如果你们计划分叉,必须显式通知——host 侧没有任何东西能接住。
  - 唯一的形状护栏:`compress_state` 与 `inner_compress_state` 的 dim0 是两个不同的 dynamic 轴(`decode_csa.py:1241-1242`),所以**页数**可以不同,但每页行数和 ring 长度不行。
- **-1 / padding / 跨页语义**与 (4) 完全相同。

#### 两个 review claim 的核实结果

1. **`_state_slots` 是死代码 —— 属实。** `pto_attn.py:288-314` 在整个 vllm-ascend 树里只有定义处一处命中(`decode_csa.py` 里的 `cmp_state_slots`/`inner_state_slots` 是同名子串的局部变量,无关)。活的是 `state_slots`(`:443`),被 `:780` 调用,`tests/pto_attn/test_pto_attn_cpu.py:104` 测的也是活的那个。
   **对 lib 的含义**:死掉的那个实现的是「直接用 vLLM 的 state block table 做普通分页查表、无 ring、无取模」,和现在活的 ring 语义**不兼容**,**不要把它当作 host 侧的意图文档**。`pto_attn.py:317-338` 的注释解释了为什么不能直接喂 vLLM 的 state 表:(a) layout 非 contiguous(65536 B 内容 / 131072 B 步长,16384 B / 16640 B),PyPTO 绑定在三处拒绝非 canonical stride,最内层是 `torch_npu_adapter.cpp` 的 `Require(tensor.is_contiguous())`;(b) vLLM 表装的是绝对页号且空洞为 0,直接喂进去会让每个请求都去寻址 page 0..7(位置 0..63)并在 null block 上互相踩踏。
2. **state / inner 同对象 —— 属实**(见 (5))。

#### ⚠ 必须上报的缺陷:压缩行下标在「矩形」上做 cumsum

**位置**:`pto_attn.py:745` `boundary, row = _compressed_rows(pos)`,其中 `pos` 是 `:737` 的矩形位置(每个请求的位置被复制 `ks=8` 次)。
**机理**:`_compressed_rows`(`:277-280`)的 `row = (cumsum(boundary)-1).clamp_min(0)` 在长度 T 的矩形轴上累加,于是每个 boundary 请求把计数 **+8 而不是 +1**;`_to_token_rows`(`:284`)随后 `row.clamp(max=src.shape[0]-1)`,`src` 长度 = 本步 boundary 数,把溢出的下标一律钳到最后一行。
**实测复算**(我在 host 上用纯 python 跑的算术复算,无 torch、无 NPU、未动任何设备):`n_real=4, ks=8`,四个请求位置都 = 1055(全是 boundary)时,lane 0 取到的打包行是 **[0, 3, 3, 3]**,正确值是 **[0, 1, 2, 3]**。`ks=1` 或 `n_real=1` 时结果正确。
**后果**:请求 1、2、3 的 **cmp_slot_mapping 与 idx_slot_mapping 全部指向请求 3 的压缩行** —— 三个请求把自己的压缩 KV / indexer KV 写进同一物理行并覆盖请求 3 的。**不报错,只算错。** 同一个 `row` 还被 `:754-755` 用来挑 `cmp_freqs_cos/sin`,本工况因四请求位置相同而恰好无害。
**触发条件**:`ks>1`(`kcsa.S=8`)且 `n_real>1` 且本步 ≥2 个 boundary 请求。我们的工况 bs=4、四请求同步推进 → **每 4 个 decode step 命中一次**。
**现有测试为何没抓到**:`tests/pto_attn/test_build_args_cpu.py:96` 的位置是 `arange(4)*7+300` = 300/307/314/321,只有一个 boundary;`:187-189` 只断言 padding lane 是 -1,不校验值。
**修向(供参考)**:`row` 应在 host token 轴上 cumsum 再展到矩形,例如 `row_host = cumsum(boundary_host)-1` 后 `row = row_host[src]`。

#### ⚠ 次要但值得上报:window_swa_indices 的对齐侧与 kernel 假设相反

`_window_indices`(`pto_attn.py:254-255`)用 `offs = arange(WIN) - (WIN-1)`,把有效行放在**右侧**(列 k ↔ `abs_pos = pos-(WIN-1)+k`),docstring(`:245-247`)称「kernel 只对 validity 取 row-max,所以 padding 在哪边无所谓」。**这不成立**:`decode_sparse_attn_csa.py:267-282` 的取数分支按 `qk_win_len = min(pos+1, WIN)`、`qk_win_start = pos - qk_win_len + 1` 从**第 0 列**开始读,即假设有效行**左对齐**(lib 自家参考实现 `decode_metadata.py:70-91` 正是左对齐:`offset < valid_len` 填,其余 -1)。只有 `sparse_bias` / `valid_block_mask`(`:154-173`)用 row-max。
当 `pos >= WIN-1 = 127` 时两种对齐重合,prompt=1024 的工况恒等;**`pos < 127` 会取到错的 KV 行而 mask 却说有效**。现有 CPU 测试用 pos=300..303(`test_pto_attn_cpu.py:42`)且只校验最后一列(`:46-48`),覆盖不到。

#### FACTS
- 【读码】DSA 的 raw/compressed/indexer KV 页大小固定 128,block table 宽度 = ceil(max_model_len/128);本配置 8704 → 68 列。vllm_ascend/attention/dsa_v1.py:349 (block_size=128), dsa_v1.py:371 (self.max_blocks = (max_model_len + block_size - 1)//block_size)
- 【实测】prefill 结构 dump 里 cmp/idx/swa 三组的 block_table 都是 [1,68] int32 contiguous stride [68,1];两组 compressor state 是 [1,1088](页=8, 8704/8)。/data/sunkaixuan/skx_log_output/csa_cut_20260920/probe/model_layers_2_self_attn.json
- 【实测】bs=4 decode 时两张表就是 [4,68] int32:msprof 的 SparseAttnSharedkv Input Shapes = "4,64,512;34730,128,1,512;34730,128,1,512;;4,1,512;4,68;4,68;5;..."。/data/sunkaixuan/skx_log_output/dsv4_vllm/fdoprof/prof_native/*/profile/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv
- 【实测】native decode 探针 ori_block_table=[1,68] int32、cmp_block_table=[1,68] int32(positions=[4097], seqused_kv=[4098])。/data/sunkaixuan/skx_log_output/csa_b_tier/probe1/attn__model_layers_2_self_attn_attn.json
- 【读码】cmp/idx 组的 block table 逻辑流是压缩行号 r_c = pos//4,不是 token position:compressed_historical_len = num_computed // compress_ratio,compressed_pos_ids 按请求主序生成。vllm_ascend/utils.py:1567-1586;由 vllm_ascend/worker/model_runner_v1.py:1072-1077 的 compute_slot_mapping 分页
- 【读码】block table 的未分配列是 0 或陈旧值,不是 -1;compress 模型的正常步只填 slot_mapping 的 -1,不碰 block table。vllm_ascend/worker/model_runner_v1.py:2793-2807;dsa_v1.py:971
- 【读码】vLLM 的 block 0 是 null block,永不分配给请求,所以 0 == 空洞。vllm/v1/core/block_pool.py:176-177
- 【读码】repage_kv 的 contiguous 分支是纯 reshape 零拷贝,且 Paged._cache=None ⇒ remap 恒等、commit 空操作。pto_attn.py:596-599, :540-541, :546
- 【读码】repage 公式:bt_new[i,j] = clamp(bt_vllm[i, clamp(j//4, max=ncols-1)]*4 + j%4, 0, n_blocks-1),j=0..want-1,want=min(kernel_cols, ncols*4)=min(8192,272)=272。pto_attn.py:232-238, :594
- 【读码】_pad_cols:have==cols 原样;have>cols 截断 bt[:, :cols];have<cols 右补 -1。本配置补 8192-272=7920 列 -1。pto_attn.py:612-621
- 【推导+读码】repage 恒等性:kernel 行 = bt_new[i,r//32]*32 + r%32 = bt_vllm[i,r//128]*128 + r%128 = vLLM flat 行,因为 j//4=r//128 且 j%4=(r%128)//32。依赖 32|128 与 row-major contiguous
- 【读码】block table 宽度是编译期常量:cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],只有 dim0 bind_dynamic。decode_csa.py:1218, :1221, :1257-1258;CMP_MAX_BLOCKS=IDX_MAX_BLOCKS=ceil((1048576//4)/32)=8192 @ decode_csa.py:177-179
- 【读码】kernel 读 block table 时没有 >=0 保护,读到 -1 会得到 -32 行。decode_sparse_attn_csa.py:289-290;decode_indexer.py:420-421
- 【读码】indexer 的读列被 kv_seq_lens 限住:cache_len = kv_seq_lens[b] // COMPRESS_RATIO。decode_indexer.py:197, :324;故 kv_seq_lens 传 token 数(pto_attn.py:802)是对的
- 【实测】raw KV 与 compressed KV 无页内 padding、无交织:[21777,128,1,512] bf16 contig=true stride [65536,512,512,1](65536=128*512)。csa_cut_20260920/probe/model_layers_2_self_attn.json 的 kv_cache[0]/[1]
- 【实测】indexer 的 key 与 scale 交织在同一个 16640B 页里:idx_k stride [16640,128,128,1](内容 16384B)、idx_scale stride [8320,1,1,1](=16640B),16384+128*2=16640。同一 probe 的 kv_cache[4]/[5]
- 【实测】两块 state cache 是 padded strided:main [21777,8,1,2048] fp32 stride [32768,...](65536B 内容/131072B 步长)、inner [21777,8,1,512] fp32 stride [4160,...](16384B/16640B)。同一 probe 的 kv_cache[2]/[3]
- 【实测】trace 里对形状 34730,128,1,512 的两块 bf16 cache 没有任何 Gather/Index/Slice/Copy 算子 —— 零搬运被证实;而 idx 的压缩与回写清晰可见:GatherV3 "34730,128,1,128;272;1"→"272,128,1,128" 14.25us×8、GatherV3 "34730,128,1,1;272;1" 10.89us×8、IndexPutV2 "34730,128,1,128;32,1,128"。prof_pto/.../kernel_details.csv
- 【实测】conversion 层引入了 native profile 里完全没有的巨型拷贝:ViewCopy "1138032640;...;34730,8,1,2048" 3600us×8、ViewCopy "...;34730,8,1,512" 927us×8、ViewCopy "...;34730,128,1,128" 926us×8、Slice "34730,16,2048"→"34730,8,2048" 3549us×24、AsStrided "144476800"→"34730,8,1,512" 966us×24;native 侧这些一条都没有
- 【读码】kernel 的 SWA 取数不是逐列读 window_swa_indices,而是按 32 行对齐切 run、每 run 只读一个索引再 gather_row 连读最多 32 行,因此要求 run 内物理行连续。decode_sparse_attn_csa.py:271-282
- 【读码】entry decode_csa_attn_tp1_test 没有 ori_block_table 形参,raw KV 的分页只通过 ori_slot_mapping 与 window_swa_indices 进入 kernel;pg_swa.table 被算出来后从未使用。decode_csa.py:1188-1235;pto_attn.py:758 与 :766-770
- 【读码】五个 slot mapping 都是 [T_DYN] INT64,kernel 对每个都有 >=0 保护且无其它负值语义:ori decode_csa.py:958-961;cmp decode_compressor_ratio4.py:354-360;state decode_compressor_ratio4.py:299-307;idx decode_indexer_compressor.py:393-397 与 :410-417;inner decode_indexer_compressor.py:234-244
- 【读码】vLLM 用 -1 标记 padding token 的 flat slot,经 torch.stack([sm//128, sm%128]) 变成 (-1,127);_flat_slots 的 sm[:,0]*128+sm[:,1] = -128+127 = -1 精确还原。model_runner_v1.py:2794-2796;dsa_v1.py:563-564;pto_attn.py:222
- 【读码】dummy / graph-capture 那一趟 vLLM 把 compress 组的 slot_mapping 和 block table 全填 0(不是 -1)。model_runner_v1.py:2803-2804
- 【读码】ori_slot_mapping = ori_bt[req,pos//128]*128 + pos%128,因页视图是纯 reshape,该数字同时就是 kernel 32 行页下的 flat 物理行;kernel 写 kv_cache_flat[slot]。pto_attn.py:787-788, :793;decode_csa.py:951, :958-961
- 【读码】cmp_slot_mapping 的逻辑流是压缩行号,vLLM 侧按压缩行打包,host 用 row=cumsum(boundary)-1 展回每 token,非 boundary 填 -1。pto_attn.py:277-280, :284-285, :789-790;dsa_v1.py:945-951, :963
- 【推导】跨页实例:r_c=255 → cmp_bt[1]*128+127,kernel 页 7 → (cmp_bt[1]*4+3)*32+31 = 同一数;r_c=256 → cmp_bt[2]*128+0,kernel 页 8 → (cmp_bt[2]*4)*32+0 = 同一数。两个相邻压缩行跨一个 vLLM 物理页,两侧公式一致
- 【读码】idx_slot_mapping 被 Paged.remap 改写成私有 compact buffer 的行 moved=(req*68+col)*128+intra,col 是在该请求的 block table 行里搜出来的;未命中或原值<0 → -1。pto_attn.py:517-532, :537-543, :795
- 【读码】idx_kv_scale 没有自己的 mapping,由 pg_ids.track 登记、kernel 用同一 idx_slot_mapping 行写。pto_attn.py:797;decode_indexer_compressor.py:410-417
- 【读码】indexer-compressor 的 cache 写不按 token 遍历而按压缩行反推 token:token = compact_token*4 + 3 - first_pos%4,假设每请求 S/4 个 boundary 且位置由 first_pos%4 决定。decode_indexer_compressor.py:388-397, :406-417
- 【读码】state_slot_mapping = req*16 + pos%16,是 host 私有 ring 的 flat 行号,不是任何 vLLM 地址。pto_attn.py:443-454
- 【读码】kernel 读端 ring_row = logical_pos % 16、page_off = ring_row//2、blk = bt[c_idx,page_off]、row = blk*2 + ring_row%2,带 blk>=0 保护;代入 host 的 bt[r,k]=r*8+k 得 row = r*16 + logical_pos%16,与写端恒等。decode_compressor_ratio4.py:184-207, :299-307;pto_attn.py:436-440
- 【读码】与 lib 参考实现对齐:csa_state_logical = pos//2、bt[req, logical % CSA_STATE_REQUIRED_BLOCKS]、slot = blk*2 + pos%2、blk<0 或 count 不足 → -1。decode_metadata.py:225-248;CSA_STATE_REQUIRED_BLOCKS = 512//64 = 8 @ decode_metadata.py:42, config.py:272-274
- 【读码】state ring 的 16 个 plan 位置是连续的 [first-8, first+7],所以 pos%16 是 0..15 的双射,无碰撞;vLLM 侧 (blk,intra) 分开保存是因为 cache dim0 strided。pto_attn.py:362-373, :356-358
- 【读码】write_state_ring 按 span = 16/8+1 = 3 个连续逻辑页聚合,避免 index_copy_ 重复页下标互相覆盖。pto_attn.py:395-434, :401-405
- 【读码】claim 核实:_state_slots(pto_attn.py:288-314)全树无调用者,确为死代码;活的是 state_slots(:443),由 :780 调用,tests/pto_attn/test_pto_attn_cpu.py:104 也只测活的那个
- 【读码】claim 核实:inner_state_slot_mapping 与 state_slot_mapping 是同一个 Python 对象,赋值两次。pto_attn.py:781 与 :804
- 【读码】inner 与 main 的 ring 几何当前完全相同(STORAGE_LEN=16, BLOCK_SIZE=2, MAX_BLOCKS=8),所以共用一张 slot 表成立;但 host 用 state_ring_len() 硬编码给 inner 建 ring 与表,分叉时不会报错。decode_csa.py:160-171;pto_attn.py:343-346, :383-392, :436-440
- 【读码】kernel 的两个 state slot mapping 形参都是 pl.Tensor(只读)而非 pl.InOut,所以共用同一对象目前安全。decode_csa.py:1226-1227
- 【缺陷·读码+实测复算】pto_attn.py:745 在矩形位置上做 cumsum(pos 每请求被复制 ks=8 次 @ :737),row=cumsum(boundary)-1(:279)每个 boundary 请求 +8,再被 :284 的 clamp(max=#boundary-1) 钳到最后一行。纯算术复算:n_real=4,ks=8,四请求位置均 1055 时 lane0 取到 [0,3,3,3],正确值 [0,1,2,3] —— 请求 1/2/3 的 cmp 与 idx slot 全部指向请求 3 的压缩行,静默覆盖
- 【读码】上述缺陷未被现有测试覆盖:tests/pto_attn/test_build_args_cpu.py:96 的位置 arange(4)*7+300 = 300/307/314/321 只含一个 boundary,:187-189 只断言 padding lane 是 -1 不校验值
- 【缺陷·读码】_window_indices 把有效行右对齐(offs=arange(WIN)-(WIN-1) @ pto_attn.py:254-255),而 kernel 的 SWA 取数按 qk_win_start = pos-min(pos+1,WIN)+1 从第 0 列开始读,即假设左对齐(lib 参考实现 decode_metadata.py:70-91 也是左对齐)。pos>=127 时两者重合,pos<127 会取错 KV 行而 mask 仍判有效。decode_sparse_attn_csa.py:267-282
- 【读码】docstring 中「kernel 只对 validity 取 row-max」只适用于 sparse_bias / valid_block_mask 那一段,不适用于取数段。decode_sparse_attn_csa.py:154-173 vs :267-282
- 【读码】kernel 侧常数确认:BLOCK_SIZE=32、C4A_COMPRESSOR_BLOCK_SIZE=2、DECODE_BATCH=64、TP=4、DECODE_SEQ=1+7=8 ⇒ B=16, S=8, T=128。pto_kernels/dspark/config.py:243-259, 285;decode_csa.py:132-134
- 【实测】compare 记录(B=1,S=8 的那次实例化)显示 kv_cache/cmp_kv 被 reshape 成 [87104,32,1,512](=21776*4)、idx_kv_cache/idx_kv_scale 被压缩成 [272,32,1,...](=1*68*4)、cmp_block_table/idx_block_table 为 [1,8192]、compress_state [8,2,2048] + 表 [1,8]。/data/sunkaixuan/skx_log_output/csa_cut_20260920/compare/compare__model_layers_2_self_attn_attn.json

#### NEEDS_LIVE
- 【block table 真实内容 —— item 5(a) 的"example values"缺口】在 pto_attn.build_args 里、repage_kv 调用之前(pto_attn.py:757 之前),包在 `if not capture_active():` 里打印:swa_md.block_table[:n_real, :20]、cmp_md.block_table[:n_real, :20]、idx_md.block_table[:n_real, :20]、cst_md.block_table[:n_real, :20]、ist_md.block_table[:n_real, :20],外加每张表 `(bt==0).int().argmax(1)`(第一个 0 出现的列)。用途:确认未分配列确实是 0 而不是上一个占用该行槽的请求留下的陈旧页号 —— 这决定 _repage_block_table 的 clamp(0,…) 会把它映射到物理页 0..3 还是映射到别人的真实页。
- 【slot mapping 原始 (block, intra) 对】在 pto_attn.py:787 之前打印 swa_md.slot_mapping[:8]、cmp_md.slot_mapping[:8]、idx_md.slot_mapping[:8] 以及各自 _flat_slots(...) 的结果。用途:确认 padding 条目确实是 (-1, 127) → -1(model_runner_v1.py:2794-2796 + dsa_v1.py:563-564 的推论),而不是 (0,0) → 0 —— 若是后者,padding token 会写进每个 cache 的第 0 行。
- 【storage_offset 与 data_ptr】probe 的 describe()(pto_attn.py:230-231)不记录这两项。在 vllm_ascend/ops/dsa.py 的 _build_kv_cache return 之前(dsa.py:291 之前)打印 `[(c.shape, c.stride(), c.storage_offset(), c.data_ptr(), c.untyped_storage().data_ptr()) for c in caches]`。用途:验证 idx_k 与 idx_scale 是否真的共享同一块 storage(预期 scale.data_ptr() == key.data_ptr() + 16384,两者 untyped_storage().data_ptr() 相同),这是 item 5(d)"交织"结论目前唯一靠 stride 间接推出的部分;同时确认两块 bf16 cache 的 storage_offset 是 0(否则 cache.view(-1,32,...) 的 flat 行号与 vLLM flat slot 不再相等)。
- 【跨 vLLM 128 页那一步的实证 —— item 5 与 item 6 的 worked example】在 pto_attn.substitute(pto_attn.py:966)里、build_args 之后,当 `(pos[0].item() % 128) == 0` 或 `(pos[0].item() % 4) == 3` 时打印:pos[:8]、args[ARG_ORDER.index('ori_slot_mapping')]、args[...('window_swa_indices')][0, 120:128]、args[...('cmp_slot_mapping')]、args[...('idx_slot_mapping')]。用途:证实 window 行确实在第 127 列换页而前 127 列同页,以及 cmp 行在 r_c 跨 128 时跳到另一物理页。
- 【_compressed_rows 缺陷的现场确认(最高优先级)】在 pto_attn.py:745 之后打印 boundary.view(n_real, ks)[:, 0]、row.view(n_real, ks)[:, 0]、cmp_md.slot_mapping.shape[0]、idx_md.slot_mapping.shape[0]。判据:lane0 的 row 应当等于 arange(本步 boundary 数);若打印出 [0,3,3,3] 这类被 clamp 压平的序列,即确认请求 1..3 的压缩 KV / indexer KV 写到了同一物理行。同时打印 args[...('cmp_slot_mapping')].view(n_real, ks)[:,0],看四个请求的 slot 是否互不相同。
- 【state span 末页是否落到 block 0 / 是否出现重复页号】在 pto_attn.write_state_ring(pto_attn.py:395)内、cache.index_copy_ 之前(:434 之前)打印 phys(形状 [b, span])与 flat_pages,以及 `flat_pages.numel() - flat_pages.unique().numel()`。用途:确认 span 最后一列是否恒为 0(未分配逻辑页 → null block),以及 index_copy_ 是否收到重复下标 —— 重复下标下 last-write-wins,会静默丢掉某个请求的 state 更新。
- 【capture 那一趟的 slot 值】在 capture_active() 为真的那一趟(pto_attn.py:966 的 substitute 里,cap 为 True 的分支,改成用 torch.npu 的 host 打印不可行 —— 改为在 capture 前的 warm-up 趟用 dummy_run 路径打印,或在 model_runner_v1.py:2803 处加一次性打印)输出 cmp_md.slot_mapping[:4] 与 swa_md.block_table[:4,:4]。用途:验证 model_runner_v1.py:2803-2804 的全 0 分支确实命中,从而确定 capture 录制那一趟会不会往每个 cache 的第 0 行写脏数据(replay 无害,但录制趟的写是真的)。
- 【idx 的 _located 命中率】在 pto_attn.py:795 之后打印 `(args[...('idx_slot_mapping')] >= 0).sum()` 与 `(idx_flat >= 0).sum()` 两个数。若前者 < 后者,说明 Paged._located 在该请求的 block table 行里没搜到那个物理块(pto_attn.py:529-531),对应的 indexer KV 会被静默丢弃。

#### UNKNOWNS
- 两张 block table 的具体数值(某一行真实装了哪些物理页号)在所有 artifact 里都不存在 —— describe()(pto_attn.py:230-231)只记 shape/dtype/contig/stride,b_tier probe 只对 cu_seqlens_q / seqused_kv / positions 三个小张量记了 values。本答案里所有"例值"都是按公式推导的符号形式(如 cmp_bt[req,2]*128),不是读到的真实整数。
- block table 未分配列到底是 0 还是上一个请求残留的陈旧页号:代码路径显示 compress 模型的正常步不清理 block table(model_runner_v1.py:2793-2796 只填 slot_mapping),只有 dummy/capture 分支和 dsa_v1.py:971 会 fill_(0)。BlockTable 的行槽跨请求复用,所以陈旧值理论上可能存在。当前安全性依赖 kernel 从不读超出 kv_seq_lens//4/32 的列,但这是外部约束不是内部保证。
- idx_k 与 idx_scale 是否真的共享同一块 storage,目前只从 stride(16640 vs 8320*2 = 16640,内容 16384 + 256 = 16640)推断,没有 data_ptr / untyped_storage 证据。
- 两块 bf16 cache 的 storage_offset 是否为 0 未经确认。若非 0,`cache.view(-1, 32, ...)` 得到的 flat 行号与 vLLM 的 flat slot 就不再是同一个数,5(b)/5(c) 的恒等性会破。
- msprof prof_pto 里的 Slice "34730,16,2048"→"34730,8,2048"(3549us×24)与 AsStrided "144476800"→"34730,8,1,512"(966us×24)出现 24 次,而模型只有 1 个 ratio-4 层 × 8 个 profiled step。24 这个倍数没有对上任何已读到的代码路径,归属未定(可能是 warm-up/capture 重复,也可能是 vLLM 自身每 forward 重建 state cache 视图)。这不影响 item 5/6 的结论,但"这些拷贝全部由 conversion 层引入"这一判断只到"native profile 里一条都没有"为止。
- compressed 组的打包顺序是"请求主序、位置升序"这一点来自 utils.py:1567-1586 的 np.repeat / np.cumsum 构造,是读码推断;没有实测过一条真实的 cmp slot_mapping 数组来验证顺序。若顺序不是这样,_compressed_rows 的 cumsum 逆映射即使修好了 ks 的问题也仍然是错的。
- 本次全程只读文件、只跑了一次纯 python 算术复算,没有启动任何 NPU 作业,也没有插桩重跑。上面 needs_live 里的每一项都需要一次带打印的 eager(非 capture)decode 才能落地。


## 复核补充的细节更正

- WRONG ANCHOR, right conclusion — kv_seq_lens. The answer says '实测 kv_seq_lens 是 token 数(probe1 seqused_kv=4098 @ pos 4097)'. `seqused_kv` in /data/sunkaixuan/skx_log_output/csa_b_tier/probe1/attn__model_layers_2_self_attn_attn.json is an argument to the NATIVE sparse-attn op; it says nothing about `cmp_md.seq_lens`, which is what pto_attn.py:802 actually passes. The correct chain, which I verified and which does confirm the conclusion: dsa_v1.py:540 `self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]` (raw token lengths) → dsa_v1.py:1085 `seq_lens=self.seq_lens[:self.num_decodes]` into AscendDSADecodeMetadata. Contrast dsa_v1.py:668-670, where compressed lengths are derived by an explicit `// compress_ratio` and only for prefill. This matters enough to fix before sending: if the decode metadata had carried the compressed length, decode_indexer.py:197's `cache_len = kv_seq_lens[b] // COMPRESS_RATIO` would see 1/16 of the candidate rows, and the failure would be silent truncation of the indexer's search, not an error.
- MISSING ARTIFACT FACT — the one end-to-end number, and it is bad. /data/sunkaixuan/skx_log_output/csa_cut_20260920/compare/compare__model_layers_2_self_attn_attn.json records `max_abs_diff = 5.7109375`, `cosine = 0.36145833` (with `ok: true`, threshold unknown), at stage=complete, seq=1, B=1/S=8. The answer cites this file for arg shapes and omits the result. Note B=1 means n_real=1, the one configuration in which the confirmed `_compressed_rows` defect cannot fire — so this discrepancy is something else again. Five sections of exact-identity arithmetic (5b, 5c, the 6(4) two-way ring proof) go to the lib team with no statement that the only measured end-to-end comparison on this host does not match. Every 'identity, zero error' conclusion in items 5 and 6 is derived, never corroborated; say so.
- OVERSTATED UNKNOWN — the '24 occurrences' gap is resolvable from code and should move out of `unknowns`. The answer says '24 这个倍数没有对上任何已读到的代码路径,归属未定'. The pto CSV splits it for you: `aclnnIndexSelect_SliceAiCore_Slice` 34730,16,2048→34730,8,2048 appears 16 times and `aclnnInplaceIndexCopy_SliceAiCore_Slice` 8 times; identically for AsStrided (16 + 8). That is exactly 2 `cache.index_select` + 1 `cache.index_copy_` per step over 8 profiled steps: pto_attn.py:378 (`_pick_rows`, called from `make_state_ring`), pto_attn.py:426 (`write_state_ring`'s read-modify-write), pto_attn.py:433 (`cache.index_copy_`). The GatherV3 row counts corroborate it — `34730,8,1,2048;64;1` is b*_RR = 4*16 = 64 (the seed) and `;12;1` is b*span = 4*3 = 12 (the writeback). The conversion layer's ~86 ms of state-cache traffic per 8 steps is fully attributed; leaving it as 'unassigned' invites the lib team to go looking for a phantom path.
- IMPRECISE, and the imprecision weakens the answer's own defect case. The answer writes '只有 sparse_bias / valid_block_mask(:154-173) 用 row-max'. Only `valid_block_mask` is a row-max (decode_sparse_attn_csa.py:161 `raw_block_valid = pl.row_max(v_win_valid)`, written at :163-164). `sparse_bias` is strictly per-column (:173, `sparse_bias[..., 0:WIN] = (v_win_valid - 1) * -NEG_INF`). That is the stronger form of the argument: because the bias is keyed to the same column index as the gather at :277, a misaligned host row produces KV rows for the wrong absolute positions while the bias still marks those columns valid.
- UNDER-SPECIFIED MECHANISM — state the window defect the way someone editing kernel source needs it. For `pos < WIN-1 = 127`, the kernel bounds its gather to columns `< qk_win_len = pos+1` (decode_sparse_attn_csa.py:269-272) while the host writes valid entries only at columns `>= WIN-1-pos` (pto_attn.py:254-255, `abs_pos = pos - 127 + k`). The intersection is columns `127-pos .. pos`, carrying absolute positions `0 .. 2*pos-127` — so the sliding window silently attends to the OLDEST positions and drops the most recent ones. Below pos≈63 the intersection is empty, every column the kernel reads is -1, the `if qk_raw_row >= 0` at :278 skips the gather, and the KV tile stays zero. Also, the answer says the CPU test merely fails to cover this; it is worse than that — test_pto_attn_cpu.py:46-48 asserts `w[:, -1]` is the current token, so the right-aligned convention is pinned by an assertion. Changing only the host reddens that test, so the fix has to name it.
- CITATION DRIFT — four line numbers point at the wrong construct. They all resolve to something real, but a reader who opens them finds unrelated code, which is a doc-consistency defect in a document whose whole premise is 'every claim carries a file:line'. (a) `describe()` is at pto_attn.py:626-630, NOT :230-231 — :228-231 is the docstring/first line of `_repage_block_table`. The answer cites :230-231 three separate times, including in the unknowns and in needs_live #3, where it is the anchor for 'the probe records no storage_offset/data_ptr'. The fact is true at :626-630. (b) `Paged._located`'s `moved` formula is at :531, not :532 (the answer's '`pto_attn.py:132(方法内) → 文件 :532`' is also an artifact of an editing pass and should go). (c) `commit`'s early return for a non-compacted cache is at :555-556, not :546 (:546 is the docstring). (d) the 'parked' logic is at :570-577; the answer cites ':171-178', which is inside `prepare_weights`. Minor: the 'cache dim0 is strided' comment is at :352-355, not :356-358.
- WORTH ADDING, not a correction to anything said — `compressed_tokens_start`, the length of the packed array that the defective `row` indexes into, is computed by `mask.sum().item()` at dsa_v1.py:950 and memoized in `decode_ratio_to_sas_metadata` at :953-961 (a dict handed in at :512, shared across the three layers of a build). The answer describes it correctly as '本步 boundary token 数', and I found no evidence the dict survives a step, so this is not a staleness defect. But it is a device→host `.item()` sitting on the same path, in vLLM's own builder rather than in pto_attn, and the lib team asking about capture-safety will hit it.

---

# 第 7 项 · 运行时 batch/token 维度关系

> 复核状态：已复核通过

#### 第 7 题 —— 运行时 batch / token 维度关系

### 0. 先把三层数字分开（这是混淆的根源）

| 层 | 数字 | 来源 | 性质 |
|---|---|---|---|
| vLLM 服务批 | `max_num_seqs=4` | server.log:27 `'max_num_seqs': 4`（实测） | 一步最多几个请求 |
| 内核编译期容量 | `B=16, S=8, T=T_PAD=128` | decode_csa.py:132-134、decode_sparse_attn_csa.py:71 | 静态工作区上界，**不是**这一次调用的形状 |
| 本次调用的动态维 | `B_DYN=n_real, T_DYN=n_real*S` | pto_attn.py:730-736 | 真正传进去的张量第 0 维 |

**关键结论：B=16 / T=128 从来不是任何一个实参的 shape。**它们只做三件事：(1) 拒绝 `n_real > 16`（pto_attn.py:732-733、981-990）；(2) 决定内核内部静态工作区的行数（`o_packed_heads [O_GROUPS*T_PAD, O_GROUP_IN]`，decode_csa.py:993 / decode_o_proj.py:139,159,164）；(3) 决定编译期常量 `S=8`，而 `S` 被若干子内核硬编码成 token→request 的除数。

---

### (a) CSA 入口真正看到的 B / T / S

**读码链路**（pto_attn.py `build_args`，712-808）：

```
729  host_pos = metadata_list[0].decode.input_positions      # vLLM 的 decode 位置向量
730  n_real   = host_pos.shape[0] // seq                      # seq = PTO_ATTN_SEQ = 1
731  ks       = kcsa.S                                        # = 8，编译期常量
732-733 if n_real > kcsa.B: raise                             # kcsa.B = 16
735  src, real = rectangular(n_real, ks, device)
736  b, t = n_real, n_real * ks
737  pos  = host_pos.index_select(0, src * seq)
```

`rectangular`（692-709）：

```
705  t    = n_real * kernel_seq
707  req  = idx // kernel_seq          # src[t] = t // 8
708  real = (idx - req*kernel_seq)==0  # 只有 t % 8 == 0 是真 token
```

**所以本配置（bs=4, PTO_ATTN_SEQ=1, PTO_ATTN_TP=4）下：**

- `S`（内核编译期）= `DECODE_SEQ` = `1 + DSPARK_SPEC_TOKENS` = `1 + 7` = **8**（config.py:249-250，`PTO_DSPARK_SPEC_TOKENS` 未导出，走默认 7）
- `n_real` = 4 → `B_DYN = 4`，`T_DYN = 4 × 8 = 32`
- 矩形布局：T=32 个槽位，第 0/8/16/24 行是 4 个真实请求的 token，其余 28 行是 padding

**实测（两处互相印证）：**

1. `.../fdoprof/prof_pto/.../kernel_details.csv`（bs=4）：`GatherV3 in:"4,4096;32;1" out:"32,4096"` —— 这是 pto_attn.py:740 的 `hidden_states.index_select(0, src*seq)`，输入 4 行（4 个请求），输出 32 行。`take` 那一步反过来：`in:"32,4096;4;1" out:"4,4096"`（pto_attn.py:1007-1009）。
2. `.../csa_cut_20260920/compare/compare__model_layers_2_self_attn_attn.json`（n_real=1 的一步）：`x_normed [8,4096]`、`position_ids [8]`、`ori_slot_mapping [8]`、`window_swa_indices [8,128]`、`attn_out [8,4096]`，而 `kv_seq_lens [1]`、`cmp_block_table [1,8192]`、`compress_state_block_table [1,8]`。T_DYN=8=1×8，B_DYN=1。

**为什么必须补成 T=n_real×8 而不能直接送 n_real 行？** 这是本题最要紧的一条，也是矩形存在的唯一理由：内核里有**两套**token→request 映射并存。

| 映射 | 位置 | 形式 |
|---|---|---|
| 运行期推导 | decode_compressor_ratio4.py:142 `s_dim = bs // b_dim`；decode_indexer_compressor.py:139 同 | `T_DYN // B_DYN` |
| **编译期常量 S=8** | decode_sparse_attn_csa.py:202 `qk_b = qk_t // S`；同文件:535 `b = t // S`；decode_indexer.py:195 `batch_idx = query // S`、:324 `kv_seq_lens[query // S]`、:380 `for batch in pl.range(query_count // S)`、:388 | `t // 8` 写死 |

只有当 `T_DYN / B_DYN == 8` 时两者才一致。如果直接送 T=4（每请求 1 行），indexer 会把 4 个 token 全部算成 `query//8 == 0`，即**四个请求全部去读请求 0 的 `kv_seq_lens` 和 `idx_block_table` 行**——不报错，结果静默错误。这正是你们担心的那类失效模式。

另外还有两条运行期整除约束，也靠 S=8 兜住：`wb_blocks = t_dim // CSA_WB_TOKEN_TILE(=8)`（decode_csa.py:908，不整除时尾部 token 的 KV 永不落盘）、`rope_cs_blocks = t_dim // ROPE_CS_T_TILE(=8)`（decode_sparse_attn_csa.py:124）、`pl.pipeline(0, t_dim, QUANT_TOKEN_TILE=8)`（decode_o_proj.py:194）。

**为什么不能把 S 调小到 1？** `rectangular` 的 docstring（692-704）说得明白，而且可以在代码里落实：S=1 → T=B×S=16 → `T_PAD=16` → decode_o_proj.py:131-133 `if T_PAD % PROJ_B_MM_T_TILE(=128) != 0: raise ValueError(...)` 在 **import 期**就抛。所以 S=8 目前是被 O 投影的 128 行 tile 钉死的。

---

### (b) TP=4 在我们这里到底意味着什么

**诚实结论：在我们的部署里 `PTO_ATTN_TP=4` 不是并行度，是容量档位。单卡、单进程、TP=1 的 vLLM（run_meta.txt `TP=1`、`ASCEND_RT_VISIBLE_DEVICES=15`），没有第二个 rank。**

- 注入路径：pto_attn.py:35 `_TP = _env_int("PTO_ATTN_TP", 4)`，:38-47 `_import_kernel()` 往 `sys.argv` 塞 `--tp 4`，decode_csa.py:23-35 `_parse_tp_argv()` 读出来设 `config.TP`。vLLM 进程本身的 argv 里没有 `--tp`，所以不注入就会落到模块默认值 2。
- `TP_SIZE` 唯一影响我们用的入口的地方，就是 **`B = DECODE_BATCH // TP_SIZE`**（decode_csa.py:132）。TP=4 → B=16, T=128；TP=1 → B=64, T=512（更大的静态工作区、更多显存）；TP=2 → B=32, T=256。

**内核自己的设计里 TP=4 是什么（供 lib 侧对照）：** 真正的多 rank 入口是 `decode_csa`（decode_csa.py:202-265），它带 `pld.DistributedTensor` 的 `gather_window / attention_window / o_window` 和 `gather_signal / attention_signal / o_signal`，以及 `group_base / tp_rank / local_t` 三个标量。那里的切分是**沿 token 轴的 context-parallel + 输出组切分**：

- 每 rank 拥有 `DECODE_BATCH/TP = 64/4 = 16` 个请求、`LOCAL_T = DECODE_TOKENS/TP = 512/4 = 128` 个 token（decode_o_proj.py:56）；
- 每 rank 只持有 `LOCAL_O_GROUPS = O_GROUPS/TP = 8/4 = 2` 个输出组，权重是 `wo_a [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN]`、`wo_b [D, LOCAL_O_WIDTH=2048]`（decode_csa.py:252-253）；
- 完整 token 流靠 allgather 还原（`decode_cp_csa_main_typed_allgather_step`，decode_csa.py:59-70）。

**我们用的入口不是它。** 我们注册的是 `decode_csa_attn_tp1_test`（decode_csa.py:1187-1274），签名里**没有任何 DistributedTensor、没有 tp_rank/group_base/local_t**，并且权重是**全局**的：`wo_a [O_GROUPS=8, O_LORA, O_GROUP_IN]`、`wo_b [D, O_GROUPS*O_LORA]`（decode_csa.py:1231-1232）。

实测印证：compare artifact 里 `wo_a [8, 1024, 4096]`、`wo_b [4096, 8192]` —— 8 个组，全量，不是 2 组 / 2048 宽。

**一句话给 lib 侧：我们只是借 TP=4 把 token 容量从 512 压到 128（省静态工作区），计算上是单 rank 全组全头的完整 CSA；不要按"4 个 rank 各算 1/4"来理解我们的 kernel 行为。**

---

### (c) 每个参数是按请求索引还是按 token 索引

按 decode_csa.py:1189-1234 的声明维 + build_args 的填法：

**按 token（第 0 维 = `T_DYN` = n_real×8）**

| 参数 | 声明 | 填法 | 值的含义 |
|---|---|---|---|
| `x_normed [T_DYN, D]` | 1189 | pto_attn.py:740 `hidden_states.index_select(0, src*seq)` | 槽 t 取宿主行 `t//8` |
| `freqs_cos/sin [T_DYN, 64]` | 1196-1197 | :749-750 同上 index_select | 同上 |
| `cmp_freqs_cos/sin [T_DYN, 64]` | 1198-1199 | :751-755，经 `_compressed_rows` 的 `row` 展开后 index_select | 压缩表按 boundary 行打包，这里展成 per-token |
| `ori_slot_mapping [T_DYN] INT64` | 1222 | :787-788,793 | 重分页后 kv_cache 视图里的**扁平行号**；-1 = 不写 |
| `cmp_slot_mapping [T_DYN] INT64` | 1224 | :789-790,794 | 同上，且**只有压缩边界 token 非 -1**（`(pos+1)%4==0`，`_to_token_rows` 把非边界填 -1） |
| `idx_slot_mapping [T_DYN] INT64` | 1225 | :791-792,795 | 同 cmp |
| `state_slot_mapping [T_DYN] INT64` | 1226 | :780,803 `state_slots(pos, ks)` | 环行号 `req*16 + pos%16`（pto_attn.py:443-454） |
| `inner_state_slot_mapping` | 1227 | :781/804 —— **与上者是同一个张量对象** | 同上 |
| `window_swa_indices [T_DYN, WIN=128]` | 1223 | :798，`_window_indices`（241-268） | 每 token 128 个滑窗物理行号，-1 = 未映射 |
| `position_ids [T_DYN] INT32` | 1228 | :801 | **不 inert**，padding 行复制本请求的真实 position |
| `attn_out [T_DYN, D]` | 1234 | :741-742 新分配 | 输出；:1007-1009 只取 `r*8` 行回写 |

**按请求（第 0 维 = `B_DYN` = n_real）**

| 参数 | 声明 | 填法 |
|---|---|---|
| `kv_seq_lens [B_DYN]` | 1229 | :802 `cmp_md.seq_lens[:b]`，b=n_real。注意 indexer 自己再除 4：decode_indexer.py:324 `kv_seq_lens[query//S] // COMPRESS_RATIO` |
| `compress_state_block_table [B_DYN, 8]` | 1205 | :778 `state_block_table(b)`（436-440），**是我们自建环的表，不是 vLLM 的表**：`bt[r,j] = r*8 + j` |
| `inner_compress_state_block_table [B_DYN, 8]` | 1215 | :779 同上，另建一份 |
| `cmp_block_table [B_DYN, 8192]` | 1218 | :759,767 `repage_kv(...).table`，列宽被 `_pad_cols` 钉成 `CMP_MAX_BLOCKS=8192` |
| `idx_block_table [B_DYN, 8192]` | 1221 | :760,768 同上 |

**按 cache 块数（各自独立的动态维）**：`compress_state`、`inner_compress_state`、`kv_cache`、`cmp_kv`、`idx_kv_cache`、`idx_kv_scale` —— 第 0 维分别绑到 `MAIN_STATE_BLOCK_NUM_DYN / INNER_STATE_BLOCK_NUM_DYN / ORI_BLOCK_NUM_DYN / CMP_BLOCK_NUM_DYN / IDX_CACHE_BLOCK_NUM_DYN`（decode_csa.py:1241-1246），与 B/T 无关。

**其余 21 个是权重/常量表**，形状全静态，首次调用后缓存在 impl 上（pto_attn.py:163-202）。

**一条需要 lib 侧注意的不变量风险（读码）：** `B_DYN` 被 5 个参数共享，但其中 3 个（`kv_seq_lens`、两张 state 块表）行数 = `n_real`，另 2 个（`cmp_block_table`、`idx_block_table`）行数 = **vLLM 自己 block_table 的行数**（repage_kv:593 `b, ncols = block_table.shape`）。build_args 没有任何地方强制两者相等。本次实测里它们确实都等于 4（kernel_details.csv 里 `in:"4,68;32;1"`、`in:"4,4096;32;1"` 并存），但那是 vLLM 恰好把 decode block_table 切到了请求数。见 (e) 里"没人检查"那一条。

---

### (d) 失效 lane 与投机 token 怎么表示

**PTO_ATTN_SEQ=1 且 MTP 关闭**（run_meta.txt `MODEL_MTP_LAYERS=0`、`SPECULATIVE_CONFIG=(disabled)`；server.log:93 `speculative_config=None`），所以宿主每请求每步只有 1 个 token。而内核 `S` 仍然是 8（`PTO_DSPARK_SPEC_TOKENS` 未设 → config.py:249 默认 7）。**差额 7 全部是 padding lane，不是投机 token。**换句话说：本配置下内核按"每请求 8 个 token"编译，实际只喂了 1 个，7/8 的 token 槽是空转。这是已知的算力浪费，不是错误。

一个 padding lane（`t % 8 != 0`）在各参数里携带什么：

| 参数 | padding lane 的值 | 锚点 |
|---|---|---|
| `x_normed` / `freqs_*` / `cmp_freqs_*` | **真实请求那一行的副本**（`src[t]=t//8`，不区分 real） | pto_attn.py:740,749-755 |
| `position_ids` | **真实 position 的副本**，不是 -1 | :737,801（`_inert` 未作用于它） |
| `ori_slot_mapping` / `cmp_slot_mapping` / `idx_slot_mapping` | **-1** | `_inert`（784-785）：`torch.where(real, x, -1)`，:787-795 |
| `state_slot_mapping` / `inner_state_slot_mapping` | **-1** | :803-804 |
| `window_swa_indices` | **真实窗口行号**（按复制来的 position 算的，非 -1） | :798，`_window_indices` 不接 `real` |
| `attn_out` | 内核会真的算出一行结果，**被丢弃** | :1007-1009 `take = arange(n_real)*ks` |

**内核侧对 -1 的处理（全部读码确认，都是 `>= 0` 门控）：**

- 原始 KV 写回：decode_csa.py:958-961 `write_row_i64 = pl.read(ori_slot_mapping,[write_t]); if write_row_i64 >= 0:`
- 压缩 KV 写回：decode_compressor_ratio4.py:354-355 `cache_row_i64 = pl.read(cmp_slot_mapping,[token]); if cache_row_i64 >= 0:`
- 压缩器 state 提交：decode_compressor_ratio4.py:299-300 `state_row_i64 = pl.read(state_slot_mapping,[token]); if state_row_i64 >= 0:`
- indexer compressor state 提交：decode_indexer_compressor.py:233 起同构

**所以 padding lane 不会写任何一个 cache。**这一点很重要：因为一个请求的 8 个 lane 共享同一个 `position_ids`，`state_slots` 对它们算出的环行号**完全相同**（`req*16 + pos%16`），如果不 inert，8 个 lane 会对同一条 state 行并发写 —— 这正是 pto_attn.py:803 那行 `_inert` 在挡的事。改动这一带时务必保持。

**读侧不受 -1 门控，靠 position 门控：** 压缩器读 state 用的是 `first_pos_b = pl.read(position_ids, [c_idx * s_dim])`（decode_compressor_ratio4.py:160；indexer_compressor:153 同），即**只读该请求第 0 个 lane 的 position**——这正好落在 `rectangular` 放真 token 的槽 `r*S`。这是"真 token 必须放在 `r*S`，不能放在别处"的硬依据。读窗口再被 `if logical_pos >= 0 and logical_pos < first_pos_b` 夹住（ratio4:191）。

**若将来开 MTP / PTO_ATTN_SEQ>1，有一个现成的坑（读码，未实测）：** `pos = host_pos.index_select(0, src*seq)`（:737）里 `src` 取值 0..n_real-1，`src*seq` 只能落在 `0, seq, 2seq, …`。也就是说**每个请求只有第 0 个宿主 token 被读进来，第 1..seq-1 个被静默丢弃**，而 `real` 掩码也只标 lane 0。要支持投机，`rectangular` 和 `src*seq` 都得改。

---

### (e) 图捕获/重放：哪些形状固定、哪些值会变

**固定的（捕获时冻结）**

1. **所有实参的 shape。** `n_real` 是宿主侧 Python 读出来的（`host_pos.shape[0] // seq`，:730），捕获时是多少，这张图就永远是多少。
2. **三张图对应三个 descriptor。** server.log:27/93 `cudagraph_capture_sizes: [1, 2, 4]`、`max_cudagraph_capture_size: 4`、`cudagraph_num_of_warmups: 1`、`cudagraph_mode: FULL_DECODE_ONLY`。捕获顺序由大到小，实测（server.log:146-158）：

```
146 [pto-attn-ran] n=1 tokens=4 capturing=False   ← warm-up
150 [pto-attn-ran] n=2 tokens=4 capturing=True    ← 录 T_DYN=32, B_DYN=4
155 [pto-attn-ran] n=3 tokens=2 capturing=False
156 [pto-attn-ran] n=4 tokens=2 capturing=True    ← 录 T_DYN=16, B_DYN=2
157 [pto-attn-ran] n=5 tokens=1 capturing=False
158 [pto-attn-ran] n=6 tokens=1 capturing=True    ← 录 T_DYN=8,  B_DYN=1
```
（`tokens` 打的是 `n_real*seq`，seq=1，所以 tokens 即 n_real；T_DYN = tokens×8。）
3. **内核内部的静态工作区，与 T_DYN 无关，永远按 T_PAD=128 排。** `o_packed_heads [O_GROUPS*T_PAD, O_GROUP_IN]`（decode_csa.py:993 / decode_o_proj.py:139），组间跨步 `row_base_o = g * T_PAD`（decode_o_proj.py:169），`o_r_pad / o_r_i8_pad / act_scale_dq / partials` 全是 `T_PAD` 行（:159-164）。压缩器的 `BS_PAD = DECODE_BATCH*DECODE_SEQ` 对齐到 64 = **512** 行（decode_compressor_ratio4.py:69-70），比 T_PAD 还大 4 倍。所以 bs=4 时 T_DYN=32/T_PAD=128，只用了 25% 的 O 打包 slab、6% 的压缩器工作区。

**会变的（每步重算）**

`position_ids`、四张 slot mapping、`window_swa_indices`、两张 KV block table、`kv_seq_lens` 的**内容**每步都变。关键机制：`build_args` 全部是**设备端算子**，被录进了图里。kernel_details.csv（prof_pto，实测）里能直接看到它们：`Range 216`、`GatherV3 208`、`FloorDiv 176`、`SelectV2 184`、`ClipByValueV2 256`、`LogicalAnd 112` —— 这些在 prof_native 里一个都没有。重放时这些算子按录下来的地址重新执行，从元数据缓冲区**当前的内容**重新推导出全部派生张量。

**因此哪些张量必须地址稳定：**

- **已审计、结论 none：** 5 个 metadata 对象各 4 个字段 = 20 个张量（`block_table / slot_mapping / seq_lens / input_positions`）。
  - 审计代码：pto_attn.py:847-881（`audit_inputs`），只在 `not capture_active() and _AUDITS[0] < 2` 时跑（:995-996）。
  - 实测：`.../fdoprof/prof_pto/.../server.log:144` `[pto-attn-audit] tensors=20 requests=4`，:154 `[pto-attn-audit] moved_between_steps=none`。另两次独立复现：`.../fdo3/fdo2_pto/.../server.log`、`.../fdo2/fdo2_pto/.../server.log:155`。
  - **诚实边界**：它只比了两次 warm-up（都是 capturing=False，分别在 size-4 和 size-2 descriptor 之前），没有覆盖稳态 decode；而且只比这 20 个，**不含** hidden_states、output、6 个 KV cache、权重。
- **未审计但结构上稳定：** 6 个 KV cache 来自 `_build_kv_cache`（dsa.py:269-303），引擎初始化时一次性分配；权重缓存在 `impl._pto_attn_weights`（pto_attn.py:165-167,201），首调后不再重建。
- **未审计、属 vLLM 自身契约：** `hidden_states` 与 `output` 是 vLLM 每个 descriptor 的常驻图 I/O buffer。
- **不需要外部稳定：** build_args 新分配的一切（`x_normed` 副本、`attn_out`、state ring、compact 后的 idx cache）都落在捕获内存池里，地址随图录下。代价是三张图各留一份：server.log:160 `Graph capturing finished in 20 secs, took 3.67 GiB`。

**"nothing checks them" 的代码依据（重要）：** `bind_dynamic` 在 pypto 里是**纯编译期标注**——tensor.py:226-232 "This is a no-op at runtime. The @pl.jit specializer reads this call statically from the AST"。运行期校验在 interop.py:151-154：

```python
if len(shape) != len(info.shape) or any(
    expected not in (-1, actual) for expected, actual in zip(info.shape, shape)
):
    raise ValueError(...)
```

动态轴是 `-1`，`-1` 与**任何**实际值匹配，而且是**逐参数**比对，**没有任何跨参数一致性检查**。所以 `T_DYN` 在 13 个参数之间、`B_DYN` 在 5 个参数之间是否真的一致，运行期无人验证；不一致的后果是越界读或读错请求，而不是报错。

**捕获之后 Python 不再跑（实测）：** prof_pto 全日志里 `[pto-attn-ran]` 只有 6 行（n=1..6），全在捕获窗口内；打印门槛是 `_RAN[0] <= 5 or _RAN[0] % 10 == 0`（pto_attn.py:1015），之后再没出现 `n=10`，说明 `substitute` 调用次数 ≤ 9，**稳态 decode 步全是图重放**。而内核确实在重放里执行：kernel_details.csv 中只在 prof_pto 出现 `aicore_kernel_mode_0_mix_aic ×8` 与 `simpler_aicpu_kernel_exec_891fbc868201ba61 ×8`；同时 `Compressor ×16`、`QuantLightningIndexer ×8` 在 prof_pto 里消失，`SparseAttnSharedkv` 从 24 降到 16 —— 正好对应 3 层中只有 `model.layers.2` 是 ratio-4 被替换（server.log:141-143 `layers.0/1 ratio=0`、`layers.2 ratio=4`）。

**一个必须点名的重放风险（读码 + 推理，未实测）：** `n_real` 被冻结在图里。当真实请求数是 3 时，vLLM 会 pad 到 size-4 的图去重放，内核仍按 4 个请求 × 8 lane 跑，读的是 vLLM 在 `input_positions / slot_mapping / block_table` **padding 行**里留下的值。这些 padding 行如果 slot 是 0 而不是 -1，`state_slot_mapping` 会算出一个合法的环行号并真的写进去 —— 正是你们说的"静默污染 state"。另有一条历史实测佐证方向：`.../fdo2/fdo2_pto/.../server.log:144-145`（旧版 audit）打过 `block0_in=md0.block_table,...,md4.block_table` 和 `WARNING: block 0 is reachable from a live block table, so commit()'s scratch row is not safe` —— block 0 在活的块表里可达。当前版本的 `audit_inputs` 已经不做这项检查了。

---

### 给 lib 侧的一句话总结

服务批 4 和内核 B=16/S=8 不冲突：**B=16 是容量上界（决定静态工作区和拒绝阈值），S=8 是编译期写死的 token→request 除数**；真正进内核的是 `B_DYN=4, T_DYN=32`，4 个真实 token 摊在 32 个槽的第 0/8/16/24 行，其余 28 个 lane 靠四张 slot mapping 写 -1 保持惰性。TP=4 在我们这边**只**把 token 容量从 512 压到 128，计算上仍是单 rank、8 个输出组的完整 CSA。

#### FACTS
- 【读码】`B = DECODE_BATCH // TP_SIZE` = 64//4 = 16；`S = DECODE_SEQ` = 8；`T = B*S` = 128 —— decode_csa.py:132-134（/data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-ascend-v0.20.2rc1/vllm_ascend/attention/pto_kernels/dspark/decode_csa.py）
- 【读码】`DECODE_BATCH = 64`；`DSPARK_SPEC_TOKENS = int(os.environ.get("PTO_DSPARK_SPEC_TOKENS","") or 7)`；`DECODE_SEQ = 1 + DSPARK_SPEC_TOKENS` = 8；`DECODE_TOKENS = 512` —— dspark/config.py:243, 249, 250, 251
- 【读码】`T_PAD = ((T + 16 - 1)//16)*16` = 128 —— dspark/decode_sparse_attn_csa.py:71；`LOCAL_T = DECODE_TOKENS//TP = 128`、`LOCAL_T_PAD = 128` —— dspark/decode_o_proj.py:56, 72；decode_csa.py:196-199 断言 T==LOCAL_T 且 T_PAD==LOCAL_T_PAD
- 【读码】本次调用的动态维由宿主请求数决定，不是 B/T：`n_real = host_pos.shape[0] // seq`、`ks = kcsa.S`、`b, t = n_real, n_real*ks` —— pto_attn.py:730-736（/data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-ascend-v0.20.2rc1/vllm_ascend/attention/pto_attn.py）
- 【读码】`rectangular()` 把宿主行 r 放在槽 r*S，`real = (idx - req*kernel_seq) == 0`，即只有 t%8==0 是真 token —— pto_attn.py:692-709
- 【读码】容量拒绝阈值用的是编译期 B：`if n_real > kcsa.B: raise`（pto_attn.py:732-733）；捕获期同一判断改成 return False 而非抛错（pto_attn.py:981-990）
- 【实测】bs=4 时 T_DYN=32、B_DYN=4：kernel_details.csv 中 `GatherV3 in:"4,4096;32;1" out:"32,4096"`（= pto_attn.py:740 的 x_normed index_select），以及回取 `in:"32,4096;4;1" out:"4,4096"`（= pto_attn.py:1007-1009）—— /data/sunkaixuan/skx_log_output/dsv4_vllm/fdoprof/prof_pto/dsv4_mtp_vllm_20260920_192612_dp1_bs4/profile/dp0_pp0_tp0_dcp0_ep0_rank0_3965829_20260920192745483_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv
- 【实测】n_real=1 的一步里 T_DYN=8、B_DYN=1：x_normed[8,4096]、position_ids[8]、ori_slot_mapping[8]、window_swa_indices[8,128]、attn_out[8,4096]；kv_seq_lens[1]、cmp_block_table[1,8192]、idx_block_table[1,8192]、compress_state_block_table[1,8] —— /data/sunkaixuan/skx_log_output/csa_cut_20260920/compare/compare__model_layers_2_self_attn_attn.json（同文件 "seq": 1）
- 【读码】内核里两套 token→request 映射并存：运行期推导 `s_dim = bs // b_dim`（decode_compressor_ratio4.py:142、decode_indexer_compressor.py:139）与编译期写死的 S=8（decode_sparse_attn_csa.py:202 `qk_b = qk_t // S`、:535；decode_indexer.py:195、:324 `kv_seq_lens[query//S]`、:380、:388）。只有 T_DYN/B_DYN==8 时两者一致
- 【读码】t_dim 必须是 8 的倍数，否则尾部 token 静默不落盘：`wb_blocks = t_dim // CSA_WB_TOKEN_TILE(=8)` —— decode_csa.py:908；另见 decode_sparse_attn_csa.py:124 `rope_cs_blocks = t_dim // ROPE_CS_T_TILE(=8)`、decode_o_proj.py:194 `pl.pipeline(0, t_dim, QUANT_TOKEN_TILE=8)`
- 【读码】S 不能降到 1：S=1 → T=16 → T_PAD=16，触发 import 期 `if T_PAD % PROJ_B_MM_T_TILE(=128) != 0: raise ValueError` —— decode_o_proj.py:131-133
- 【读码】我们注册的入口 `decode_csa_attn_tp1_test` 不含任何 DistributedTensor / tp_rank / group_base / local_t，权重用全局 O_GROUPS=8：`wo_a [O_GROUPS, O_LORA, O_GROUP_IN]`、`wo_b [D, O_GROUPS*O_LORA]` —— decode_csa.py:1187-1234, 特别是 1231-1232；对比多 rank 入口 `decode_csa` 用 LOCAL_O_GROUPS/LOCAL_O_WIDTH 与 6 个分布式窗口/信号 —— decode_csa.py:252-264
- 【实测】权重确实是全局 8 组：compare artifact 里 wo_a [8,1024,4096]、wo_b [4096,8192]（若按 TP=4 的 rank 本地版应为 2 组 / 宽 2048）
- 【读码】TP=4 的原生含义是 token 轴 CP + 输出组切分：每 rank 拥 DECODE_BATCH/TP=16 请求、LOCAL_T=128 token（decode_o_proj.py:56）、LOCAL_O_GROUPS=O_GROUPS/TP=2（decode_o_proj.py:57-58），全量 token 靠 allgather 还原（decode_csa.py:59-70 引入 decode_cp_csa_main_typed_allgather_step）
- 【实测】我们的部署是单卡单 rank：run_meta.txt 记 `TP=1`、`ASCEND_RT_VISIBLE_DEVICES=15`、`LOCAL_BS=4`、`PROMPT_TOKENS=1024`、`MAX_TOKENS=32`、`MAX_MODEL_LEN=8704`、`MODEL_NUM_HIDDEN_LAYERS=3`、`MODEL_MTP_LAYERS=0`、`SPECULATIVE_CONFIG=(disabled)` —— /data/sunkaixuan/skx_log_output/dsv4_vllm/fdoprof/prof_pto/dsv4_mtp_vllm_20260920_192612_dp1_bs4/run_meta.txt
- 【读码】按 token（T_DYN）索引的参数共 13 个：x_normed、freqs_cos/sin、cmp_freqs_cos/sin、ori/cmp/idx/state/inner_state 四组 slot_mapping（5 个）、window_swa_indices[T,128]、position_ids、attn_out —— decode_csa.py:1189, 1196-1199, 1222-1228, 1234
- 【读码】按请求（B_DYN）索引的参数共 5 个：compress_state_block_table[B,8]、inner_compress_state_block_table[B,8]、cmp_block_table[B,8192]、idx_block_table[B,8192]、kv_seq_lens[B] —— decode_csa.py:1205, 1215, 1218, 1221, 1229
- 【读码】两张 state 块表不是 vLLM 的表，是我们自建环的表 `bt[r,j] = r*8 + j`，并由 `state_slots` 化简成 `req*16 + pos%16` —— pto_attn.py:436-440, 443-454, 778-781
- 【读码】B_DYN 被 5 个参数共享但行数来源不同：kv_seq_lens 与两张 state 块表用 b=n_real（pto_attn.py:778-779, 802），cmp/idx 块表用 vLLM block_table 自己的行数（repage_kv 里 `b, ncols = block_table.shape`，pto_attn.py:593），build_args 不校验二者相等
- 【实测】本次运行里二者确实相等（都是 4）：kernel_details.csv 同时出现 `in:"4,68;32;1" out:"32,68"`（块表按 T=32 个 token 展开）与 `in:"4,4096;32;1"`，说明 vLLM 的 decode block_table 行数 = 4 = n_real；idx cache 压实为 b*ncols=4*68=272 行（`in:"34730,128,1,128;272;1" out:"272,128,1,128"`）
- 【实测】state ring 的规模与 RR=16、VLLM_STATE_PAGE=8 吻合：`in:"34730,8,1,2048;64;1" out:"64,8,1,2048"`（b*RR=4*16=64 次取页）与写回时 `;12;1`（b*span=4*(16//8+1)=12）—— kernel_details.csv，对应 pto_attn.py:349-373, 395-433
- 【读码】padding lane 的取值：x_normed/freqs/cmp_freqs/position_ids 复制本请求真值（pto_attn.py:737, 740, 749-755, 801），四张 slot mapping 被 `_inert` 置 -1（pto_attn.py:784-785, 787-795, 803-804），window_swa_indices 不 inert（pto_attn.py:798）
- 【读码】内核对 -1 一律 `>= 0` 门控，所以 padding lane 不写任何 cache：decode_csa.py:958-961（ori KV）、decode_compressor_ratio4.py:354-355（cmp KV）、decode_compressor_ratio4.py:299-300（main state）、decode_indexer_compressor.py:233 起（inner state）
- 【读码】读 state 时只取本请求第 0 个 lane 的 position：`first_pos_b = pl.read(position_ids, [c_idx * s_dim])` —— decode_compressor_ratio4.py:160、decode_indexer_compressor.py:153；读窗口再被 `if logical_pos >= 0 and logical_pos < first_pos_b` 夹住（ratio4:191）。这是真 token 必须落在槽 r*S 的硬依据
- 【读码】同一请求 8 个 lane 的 position 相同 → state_slots 给出的环行号相同；不 inert 就会 8 路并发写同一行。挡住这件事的是 pto_attn.py:803
- 【读码】PTO_ATTN_SEQ>1 时会静默丢 token：`pos = host_pos.index_select(0, src*seq)` 中 src*seq 只能取 0, seq, 2seq…，每请求第 1..seq-1 个宿主 token 读不到 —— pto_attn.py:737（以及 740, 749-750, 788 同构）
- 【实测】捕获配置与顺序：server.log:27/93 记 `cudagraph_capture_sizes: [1,2,4]`、`max_cudagraph_capture_size: 4`、`cudagraph_num_of_warmups: 1`、`cudagraph_mode: FULL_DECODE_ONLY`、`max_num_seqs: 4`；server.log:146-158 打出 warm-up/capture 交替的 tokens=4→2→1 —— /data/sunkaixuan/skx_log_output/dsv4_vllm/fdoprof/prof_pto/dsv4_mtp_vllm_20260920_192612_dp1_bs4/server.log
- 【实测】地址审计结论 none：server.log:144 `[pto-attn-audit] tensors=20 requests=4`，server.log:154 `[pto-attn-audit] moved_between_steps=none`；另两次独立复现于 .../fdo3/fdo2_pto/.../server.log 与 .../fdo2/fdo2_pto/.../server.log:155。审计实现见 pto_attn.py:847-881，触发条件 pto_attn.py:995-996（仅前两次、且非捕获态）
- 【读码】审计覆盖面有限：只比 5 个 metadata × 4 个字段 = 20 个张量（block_table/slot_mapping/seq_lens/input_positions），不含 hidden_states、output、6 个 KV cache、权重 —— pto_attn.py:859-869
- 【读码】KV cache 与权重结构上稳定：6 元组 cache 在 `_build_kv_cache` 里取自引擎持有的 layer 对象（dsa.py:269-303）；权重首调后缓存在 `impl._pto_attn_weights`（pto_attn.py:165-167, 201）
- 【读码+实测】bind_dynamic 是纯编译期标注（pypto/python/pypto/language/typing/tensor.py:226-232 "This is a no-op at runtime"），运行期校验只做逐参数匹配且 -1 匹配一切：`expected not in (-1, actual)` —— pypto/python/pypto/torch/interop.py:151-154。跨参数的 T_DYN/B_DYN 一致性无人检查
- 【实测】捕获后稳态 decode 全走图重放、Python 不再执行 substitute：prof_pto server.log 中 `[pto-attn-ran]` 只有 n=1..6 六行，且打印门槛是 `_RAN[0] <= 5 or _RAN[0] % 10 == 0`（pto_attn.py:1015），之后再无 n=10
- 【实测】替换后的内核确实在重放里跑：kernel_details.csv 仅在 prof_pto 出现 `aicore_kernel_mode_0_mix_aic ×8` 与 `simpler_aicpu_kernel_exec_891fbc868201ba61 ×8`；同时 `Compressor ×16`、`QuantLightningIndexer ×8` 在 prof_pto 中消失，`SparseAttnSharedkv` 由 24（native，3 层）降为 16（pto，2 层），与仅 model.layers.2 是 ratio-4 一致（server.log:141-143）
- 【实测】build_args 的推导全在设备端并被录进图：prof_pto 独有 `Range 216`、`GatherV3 208`、`FloorDiv 176`、`SelectV2 184`、`ClipByValueV2 256`、`LogicalAnd 112`、`ScatterUpdate 32`、`IndexPutV2 32`，prof_native 一个都没有
- 【读码】内核静态工作区按 T_PAD=128 排，与 T_DYN 无关：`o_packed_heads [O_GROUPS*T_PAD, O_GROUP_IN]`（decode_csa.py:993、decode_o_proj.py:139），组跨步 `row_base_o = g * T_PAD`（decode_o_proj.py:169），`o_r_pad/o_r_i8_pad/act_scale_dq/partials` 均 T_PAD 行（decode_o_proj.py:159-164）；压缩器 `GROUP_BS = DECODE_BATCH*DECODE_SEQ`、`BS_PAD = 512`（decode_compressor_ratio4.py:69-70）
- 【实测】三张图各自占内存池：server.log:160 `Graph capturing finished in 20 secs, took 3.67 GiB`
- 【实测·旧版审计】block 0 在活块表里可达：.../fdo2/fdo2_pto/dsv4_mtp_vllm_20260920_191455_dp1_bs4/server.log:144-145 `block0_in=md0.block_table,md1.block_table,md2.block_table,md3.block_table,md4.block_table` 与 `WARNING: block 0 is reachable from a live block table, so commit()'s scratch row is not safe`。当前版 audit_inputs（pto_attn.py:847-881）已不含该检查
- 【读码】PTO_ATTN_TP / PTO_ATTN_SEQ / PTO_DSPARK_SPEC_TOKENS 由启动器以 `--env NAME="${NAME:-}"` 透传（tools/pto_csa/serving/run_dsv4_mtp_vllm.sh:78-80），未设时为空串，经 `_env_int`（pto_attn.py:29-32）/ config.py:249 落到默认值 TP=4、SEQ=1、SPEC=7

#### NEEDS_LIVE
- 【padding 请求的元数据内容 —— 最高优先级】重放 size-4 图但只有 1..3 个真实请求时，vLLM 在 input_positions / slot_mapping / block_table 的 padding 行里写了什么？若 slot 是 0 而不是 -1，`state_slots` 会算出合法环行号并真的写进 state，静默污染。打印位置：pto_attn.py:735 之后（`src, real = rectangular(...)` 之下）插入一段仅在 `not capture_active()` 时执行的诊断，打印 `host_pos.tolist()`、`swa_md.slot_mapping[:n_real+2].tolist()`、`cmp_md.slot_mapping[:n_real+2].tolist()`、`idx_md.slot_mapping[:n_real+2].tolist()`、`cst_md.block_table[:n_real+2, :4].tolist()`、`cmp_md.seq_lens[:n_real+2].tolist()`，并且要在一个真实请求数 < 4 的稳态 decode 步上抓（可临时设 --enforce-eager 或把 cudagraph_capture_sizes 只留 4 来强制 pad）。
- 【B_DYN 是否真的跨参数一致】cmp_block_table / idx_block_table 的行数来自 vLLM 的 block_table.shape[0]，kv_seq_lens 与两张 state 块表的行数来自 n_real，运行期无人校验（interop.py:151-154 的 -1 匹配一切）。打印位置：pto_attn.py:806 的 return 之前，打印 `[a[n].shape[0] for n in ('kv_seq_lens','compress_state_block_table','inner_compress_state_block_table','cmp_block_table','idx_block_table')]` 与 `a['position_ids'].shape[0]`，确认前 5 个全等且第 6 个是其 8 倍。建议直接升级成一条 assert 留在代码里。
- 【真实 block_table 一行长什么样】cmp/idx/state 块表在真实请求上的实际列内容、有多少列是有效块、null 块用的是 0 还是 -1。这是判定 `_repage_block_table`（pto_attn.py:225-238，把无效列 clamp 到 0..n_blocks-1）是否会把空洞指向块 0 的前提。打印位置：pto_attn.py:758-760 三次 repage_kv 之前，打印 `swa_md.block_table[0,:16].tolist()`、`cmp_md.block_table[0,:16].tolist()`、`idx_md.block_table[0,:16].tolist()`、`cst_md.block_table[0].tolist()`、`ist_md.block_table[0].tolist()`。
- 【storage_offset / data_ptr】现有 probe dump（csa_cut_20260920/probe/*.json、csa_b_tier/probe1/*.json）只有 shape/dtype/contig/stride，没有 storage_offset 也没有 data_ptr，所以无法判定任何一个 cache 视图是否带非零偏移。打印位置：pto_attn.py 的 `describe`（626 起）里补 `storage_offset` 与 `data_ptr`，或在 build_args 末尾对 6 个 cache 打印 `(t.data_ptr(), t.storage_offset(), t.untyped_storage().data_ptr(), t.stride())`。
- 【稳态地址稳定性】现有 moved_between_steps=none 只采样了捕获期两次 warm-up。要覆盖稳态，需把 pto_attn.py:995 的 `_AUDITS[0] < 2` 放宽（例如改成每 50 步采一次），在一次完整的 32-token 生成里跑一遍，确认 20 个 metadata 张量 + 6 个 KV cache + hidden_states/output 的 data_ptr 全程不变。注意这条读 data_ptr 不读张量内容，捕获态下也不会触发 107027，但仍建议门控在 `not capture_active()`。
- 【跨 ring wrap 的行为】state ring 长 16 行（MAIN_STATE_STORAGE_LEN），position 每 16 步绕一圈。需要在一次 position 跨过 16 的倍数的 decode 步上，打印 `state_ring_plan` 返回的 (blk, intra, ring_rows, valid) 与 `state_slots` 的输出，确认 seed 与 write-back 的页集合无重复、且 `write_state_ring` 的 span=3 页覆盖完整。打印位置：pto_attn.py:774-776 之后与 1000-1003 之前。

#### UNKNOWNS
- 本题给的 bs=4 / capture[1,2,4] / prompt1024 / max_tokens32 这组配置，我能拿到的 artifact 是 fdoprof 那一对 profile 和 server.log，以及一份 n_real=1 的 compare json。没有任何一份 artifact 是在 bs=4 的稳态 decode 步上 dump 出 46 个实参完整 shape 的；bs=4 的 T_DYN=32 / B_DYN=4 是从 kernel_details.csv 的算子输入输出形状反推的，不是直接读到的 arg_shapes。
- probe dump（model_layers_{0,1,2}_self_attn.json）全部是 prefill 步（decode=None，attn_state=PrefillNoCache），里面没有任何 decode metadata 的 shape，所以 decode.block_table / slot_mapping / seq_lens / input_positions 的真实第 0 维只能从 kernel_details.csv 的 GatherV3 输入形状间接推（4 行、68 列）。
- 没有任何 artifact 记录 PTO_ATTN_TP / PTO_ATTN_SEQ / PTO_DSPARK_SPEC_TOKENS 的实际取值（启动器只透传、未落日志）。我按代码默认值（4 / 1 / 7）与 compare json 里的 "seq": 1 反推，与题面给的配置一致，但这是推断不是直接观测。
- 真实请求数 < capture size 时 vLLM 往 padding 行写什么，完全未知 —— 这是 (d)(e) 里唯一一个会静默损坏 state 的口子，且现有 artifact 里没有任何线索。
- `cmp_freqs_cos/sin` 走 `_compressed_rows(pos)`（pto_attn.py:271-280, 745, 753-755），而 pos 是矩形化之后的向量，同一请求 8 个 lane 的 position 相同 → boundary 会重复计数 → `row = cumsum(boundary)-1` 在多请求时是否仍指向正确的压缩行，我没有验证。它被 `.clamp(max=cc.shape[0]-1)` 兜住不会越界，但可能取错行。这落在第 5/8 题的范围，此处仅记录。
- 我没有验证 `decode_csa_attn_tp1_test` 在 T_DYN 不是 T_PAD 时，`o_packed_heads` 中 t_dim..T_PAD 之间的行是否被清零。decode_o_proj.py:213-215 有一段把 t_dim..proj_b_padded_rows 填 0 的逻辑（针对 o_r_i8_pad），但 o_packed_heads 本身的尾部行由 sparse_attn_csa_tp1 写，我没读到它是否覆盖全部 T_PAD 行。若未清零且 proj_a 的 pa_rows 用 pl.min 截断（decode_o_proj.py:178），则应该无害，但未确证。


## 复核补充的细节更正

- FALSE as stated: '这些在 prof_native 里一个都没有' (the Range/GatherV3/FloorDiv/SelectV2/ClipByValueV2/LogicalAnd/ScatterUpdate/IndexPutV2 list). Exact counts by the CSV's Type column — pto vs native: Range 216/0, GatherV3 208/0, ClipByValueV2 256/0, LogicalAnd 112/0, ScatterUpdate 32/0, IndexPutV2 32/0, but FloorDiv 176/40 and SelectV2 184/24. Rewrite as 'six of these eight are absent from prof_native; FloorDiv and SelectV2 exist natively (40 and 24) and rise to 176 and 184.' The per-op numbers the answer gave are all correct — only the blanket statement is wrong.
- Off by one: '其余 21 个是权重/常量表' should be 22. 46 args − 13 token-indexed − 5 request-indexed − 6 cache-indexed = 22 (decode_csa.py:1190-1195, 1200-1203, 1206-1213, 1230-1233).
- Mis-attributed log line: 'server.log:27/93 记 … cudagraph_num_of_warmups: 1'. Line 27 (the non-default-args dump) actually shows cudagraph_num_of_warmups': 0 and cudagraph_capture_size': None; only line 93 (the resolved engine config) shows 1 and 4. cudagraph_capture_sizes: [1,2,4], FULL_DECODE_ONLY and max_num_seqs: 4 are where claimed.
- Mislabeled citation: decode_sparse_attn_csa.py:535 ('b = t // S') is inside the torch golden reference, not kernel code — line 531 is `o = torch.zeros(tokens, H, HEAD_DIM)` and 539 calls `.tolist()`. Drop it from the list of device-side compile-time `// S` sites; the claim stands on :202 plus decode_indexer.py:195/324/380/388, which are all genuine kernel code.
- Incomplete quote: the print gate at pto_attn.py:1015 is `if cap or _RAN[0] <= 5 or _RAN[0] % 10 == 0`, not `_RAN[0] <= 5 or _RAN[0] % 10 == 0`. The inference is unaffected (cap is False after capture), but the `cap or` is why n=6 printed at all, so omitting it makes the six-line sequence look unexplained.
- Unanchored: '6 个 KV cache 来自 _build_kv_cache（dsa.py:269-303），引擎初始化时一次性分配'. Lines 269-303 re-read `self.swa_cache_layer.kv_cache` / `self.compressor.state_cache.kv_cache` / etc. on every call and say nothing about when the storage was allocated or whether its address is stable. Keep the 6-tuple and its order (which are correct and match pto_attn.py:725), drop '一次性分配', and move the stability question wholly into the live-read list where the data_ptr item already sits.
- Wrong for one of four: 'o_r_pad / o_r_i8_pad / act_scale_dq / partials 全是 T_PAD 行'. decode_o_proj.py:161 declares `act_scale_dq = pl.create_tensor([O_GROUPS, T_PAD])` — T_PAD is its column count. The other three (159, 160, 164) are T_PAD-row as claimed.
- Name mismatch a reader will grep for: decode_o_proj.py:139 declares the parameter `o_packed`, not `o_packed_heads`. `o_packed_heads` is the caller's local at decode_csa.py:993. Both shapes are [O_GROUPS*T_PAD, O_GROUP_IN] as claimed.
- Over-labeled as 【实测】: '本次实测里它们确实都等于 4' (cmp/idx block-table rows == n_real). What the CSV shows is a [4,68] tensor gathered by a 32-long index (`in:"4,68;32;1"`), i.e. vLLM's decode block_table has 4 rows; the kernel arg's row count is then inferred through repage_kv:593 + _pad_cols. No artifact prints the kernel arg's shape at bs=4. Relabel as inferred, which makes the accompanying 'no one checks this' live-read item read as the open question it is.

---

# 第 8 项 · 可复现 fixture

> 复核状态：**复核未通过，已按更正改写**

## 复核的更正（以这一节为准）

- §d indexer page-crossing, WRONG ARITHMETIC — kills the 'default run misses it' claim. Their own formula gives straddling writes at pos = 512m-1 and 512m+3. m=2 -> pos 1023 and 1027. With pos_k = 1023+k (their own definition), pos 1027 is DECODE STEP 4 of the default P≈1024 / MAX_TOKENS=32 run. The answer reports the nearest pair as '1535/1539 = step 512/516' — wrong m, and a position read as a step index. '差两个数量级' is false. Note they identify step 4 as a compression step (pos ≡ 3 mod 4) two lines earlier. Consequence: the default configuration already exercises an indexer key/scale page change on the decode side; P=1535 is still preferable only because it puts BOTH straddling writes (1535 at step 1, 1539 at step 5) inside decode rather than one of them in prefill. Say that instead.
- §d '1536 同时满足 ... 1536/4 = 384 且 384 % 128 == 0 -> 压缩 KV 128 行页沿' is wrong. boundary = ((pos + 1) % COMPRESS_RATIO) == 0 at pto_attn.py:278, so a boundary token is pos ≡ 3 (mod 4). 1536 % 4 == 0, so at pos 1536 cmp_slot_mapping and idx_slot_mapping are BOTH -1 (pto_attn.py:789-795) and nothing is written to cmp_kv or idx_kv_cache at all. The compressed-KV page crossing actually occurs at pos 1539 (compressed row 384), i.e. at step 5 together with the indexer one — not at step 2. Their claim '四项仅靠 step 1->2 这一对就能覆盖' is true only for the four token-granular boundaries (128-slot KV page, 32-slot kernel page, 16-row ring wrap, 8-row state page).
- §a-3 'overall KV cache excluded by the numel() <= 1<<22 filter' is contradicted by the artifact cited for it. Measured: /data/sunkaixuan/skx_log_output/csa_b_tier/dump1/step0.pt is 18,953,448,427 bytes (18.95 GB), and dump1/analysis.json lists stash.ori_kv = [71894,128,1,512] torch.bfloat16 and stash.cmp_kv = [71894,128,1,512] torch.bfloat16 — 4.71e9 elements each, ~1100x above 1<<22. dump2/3/4 are 103 MB, so the filter at pto_csa.py:956 postdates dump1. Correct statement: the filter exists in today's code and DOES exclude the KV caches from stash, but it is unverified by dump1, it does not apply to the `derived` half of the payload (payload.update has no filter), and the one existing artifact of this facility is a 19 GB file. Also: dump1 has no vendor_attn0.pt — only dump2/3/4 do (the code writes it only if _PENDING_VENDOR_ATTN is not None).
- e-2 'six caches' physical page 0 / row 0 holds capture-period garbage' — BOTH cited anchors say something else. dsa_v1.py:434 is inside AscendDSAMetadataBuilder.__init__ (class at :339, __init__ at :355) and allocates the persistent reusable buffer self.slot_mapping = torch.zeros((max_num_batched_tokens, 2), int32); it is filled from common_attn_metadata.slot_mapping at :562-565 as [slot//block_size, slot%block_size]. It is not 'dummy metadata's slot mapping is zeros'. dsa_v1.py:1506-1514 creates LOCAL THROWAWAY tensors — indexer_k_cache = torch.zeros((1,1,1,head_dim)), indexer_scale_cache likewise — and scatters into THOSE, with the comment 'In profiling stage, create dummy tensors to ensure ACL graph captures scatter operator.' The real indexer cache is never touched. The claim must move from `answer` to `unknowns` in full.
- The mechanism that DOES aim inert rows at page 0 was missed, and it is worse than the one claimed. dsa_v1.py:970-971: `self.start_pos_decode[num_reqs_actual:].fill_(0)` and `self.block_table[num_reqs_actual : self.num_decodes, ...].fill_(0)`. Under graph padding, a padded request's block table row is all zeros, so pto_attn's state_ring_plan (:349-374) gathers blk=0, make_state_ring seeds from physical page 0, and write_state_ring (:395-434) writes back to page 0 — while `real` in rectangular() marks lane t%S==0 of EVERY n_real request (including padded ones) as valid, so _inert does not suppress it. That is a live-tensor question worth putting at the top of the live-read list, phrased against :970-971.
- 'commit() 之前六份 cache 里没有这一步的结果' is false for four of six caches. repage_kv (pto_attn.py:580-596) takes the contiguous branch for swa and cmp and returns a Paged whose `view` is a reinterpretation of vLLM's own cache — the kernel writes vLLM's pages directly and nothing needs copying back. Paged.commit() begins `if not self.compacted or self._slots is None: return` (:554), and `compacted` is true only for the strided indexer key/scale pair. So: kv_cache/cmp_kv are live immediately; idx_kv_cache/idx_kv_scale need commit(); compress_state/inner_compress_state need write_state_ring. Dumping after :1005 is still the right rule — fix the reason, or a reader will believe kv_cache is a private buffer.
- 'Paged.commit 特意把 inert 行停泊在行 0' implies pollution; the cited docstring says the opposite. pto_attn.py:545-553 ends 'Nothing here assumes vLLM keeps row 0 free', and :570-575 computes parked = rows[first] if a real slot owns row 0 else self._cache[0,0] — i.e. the scatter is an IDENTITY for parked rows when row 0 is not really owned. commit() does not contribute to page-0 contamination.
- The live-read recipe for the one item called decision-critical cannot decide it. Printing idx_md.slot_mapping.shape[0] / cmp_md.slot_mapping.shape[0] / host_pos.shape[0] yields identical lengths under both hypotheses, because pto_attn.py:791-793 already gathers idx_md.slot_mapping by the compressed `row` index — so its length is the compressed-row count either way. What separates them is the VALUE. At pto_attn.py:735-740 (after `boundary, row = _compressed_rows(pos)`) print, for one known step: host_pos[:8], _flat_slots(cmp_md.slot_mapping, VLLM_PAGE)[:8], _flat_slots(idx_md.slot_mapping, VLLM_PAGE)[:8], idx_md.block_table[0,:8], cmp_md.block_table[0,:8], and both block_table.shape[1]. Token granularity ⇒ the idx flat row tracks pos; compressed-row granularity ⇒ it tracks pos//4. Nothing short of that settles it.
- The '68 columns' evidence is misattributed. The '4,68' in kernel_details.csv is an input of the vendor SparseAttnSharedkv op (its ATTENTION block table), and stash.ori_block_table [4,68] in dump1/analysis.json is the same table. Neither is the indexer's. The real measurement is in the compare JSON: idx_kv_cache = [272, 32, 1, 128] int8. repage_kv's compacted branch (pto_attn.py:598-610) builds view = packed.view(b*ncols*per, rows, ...), so 272 = 1 x 68 x 4 -> the indexer's vLLM block table has 68 columns = max_model_len/128. That is evidence, but it is CAPACITY, not addressing: 68x128 = 8704 slots would hold every token, or every compressed row with 4x waste. It does not settle the open question either — do not let it look like it does.
- e-4's self-consistency check does not hold for the two state caches. The tensors dumped under the names compress_state / inner_compress_state are the PRIVATE RING rebuilt each step by make_state_ring (:383-393), not vLLM's cache. Because ring_row = pos % 16 is stable per position but the window shifts by one (first-8..first+7 -> first-7..first+8), the ring row holding pos=first-8 at step 0 holds pos=first+8 at step 1 — same index, different content, by construction, one row per request. A whole-tensor bitwise comparison of step0/out vs step1/in will report a false failure. State the check as: bitwise equality applies to kv_cache / cmp_kv / idx_kv_cache / idx_kv_scale on touched rows, and to the two state caches only via their vLLM-side rows, excluding the rotated ring row.
- e-5 overstates the 'future 7 rows' hazard by omitting the kernel-side guard. decode_compressor_ratio4.py:191 reads `if logical_pos >= 0 and logical_pos < first_pos_b:` before it resolves ring_row/state_row — so the 7 rows seeded from positions > first are never read by the compressor pooling and cannot change the golden output. They still must be dumped verbatim for a bit-exact input tensor (the host-side `ok` at :369 genuinely lacks a pos <= first term, which I confirmed), but the answer presents them as a live correctness trap. Fix the framing or the lib team will go hunting for a bug that is not there.
- Anchor corrections (each verified against the file on the host): VLLM_PAGE=128 / VLLM_STATE_PAGE=8 / COMPRESS_RATIO=4 are pto_attn.py:63/64/65, not 64/65/66. KERNEL_STATE_PAGE=2 is :340, not :341. The `_RAN[0] <= 5 or % 10 == 0` print condition is :1015 (:1010 is `_RAN[0] += 1`). MAIN_STATE_LEN / MAIN_STATE_STORAGE_LEN are decode_csa.py:162/163, not 160-161 (the value 16 = COFF(2)*4 + S(8) is right). IDX_MAX_BLOCKS = CMP_MAX_BLOCKS is decode_csa.py:179, not :172. PROFILE_WARMUP is passed through at run_dsv4_mtp_vllm.sh:89, not :81. `--no-enable-prefix-caching` is dsv4_case_inner.sh:179, not :180. prepare_weights returns 22 weight entries, not 23. pypto.torch.init()'s signature is device/platform/runtime/aicpu_thread_num plus the three DFX fields — '签名只有 enable_chip_swimlane / enable_dep_gen / output_dir' is wrong as written; say 'the only DFX fields are those three'. Probe JSON: layer 2 has 5 metadata groups, layers 0 and 1 have 1 each — '5 个 metadata 组' is not true of all three files.
- Size estimates, recomputed from the compare JSON's own arg_shapes rather than estimated: the 22 weights are ~176 MB, not ~140 MB (wo_a [8,1024,4096] bf16 = 67 MB, wq_b [1024,32768] int8 = 34 MB, wo_b [4096,8192] int8 = 34 MB, cmp_wkv + cmp_wgate = 17 MB, wq_a + idx_wq_b = 17 MB, wkv = 4 MB, inner_wkv + inner_wgate = 4 MB). The idx pair estimate of 4.6 MB is right for bs=4 (272 pages at n_real=1 -> 1088 at n_real=4 -> 4.25 MB + 0.14 MB). Two-step fixture ~195 MB, not ~160 MB.

<details>
<summary>原答复（已被上面的更正推翻，保留供对照）</summary>

### 问卷第 8 项 —— 可复现 fixture：现状盘点 + 生产规格

所有路径以主机 myserver 为准。V = /data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-ascend-v0.20.2rc1，P = .../pypto。全程只读，未起任何 NPU 任务。

---

#### a) 现有的 dump 机械：四套，没有一套能直接产出这个 fixture

### 1. `PTO_ATTN_PROBE` + `dump_structure` —— 只有结构，且实测只抓到 prefill

- 入口：`$V/vllm_ascend/ops/dsa.py:209-219`。`_probe = os.environ["PTO_ATTN_PROBE"]`，条件是 `layer_name not in _PROBED`，**每层只触发一次**，且在 `substitute` 之前。
- 实现：`$V/vllm_ascend/attention/pto_attn.py:653-668` `dump_structure` → `describe`（同文件 `:626-651`）。
- **捕获**：`hidden_states` / 6 元组 kv_cache / attn_metadata（深度 4）/ 18 个 impl 权重的 `{shape, dtype, contig, stride}`。
- **不捕获**：任何数值、`storage_offset`、`data_ptr`、block table 的实际行、slot mapping 的实际值。
- **实测致命点**：三份 probe 产物 `/data/sunkaixuan/skx_log_output/csa_cut_20260920/probe/model_layers_{0,1,2}_self_attn.json` 里，**5 个 metadata 组的 `decode` 全是 `null`、`attn_state=PrefillNoCache`、`num_decodes=0`**。因为"每层只落一次"落在了第一次调用，而第一次调用必然是 prefill。所以现有 probe 产物对 decode fixture **零价值**。

### 2. `PTO_ATTN_COMPARE` + `compare_once` —— 一步、只有 46 个 arg 的 shape，输出不可作 golden

- `pto_attn.py:883-948`，由 `dsa.py:234-243` 触发，`_COMPARED` 保证每层一次。
- **捕获**：`arg_shapes`（46 项 `[shape, dtype, contig]`）+ pto/native 的 `absmax/absmean/max_abs_diff/cosine`。
- **不捕获**：数值。
- 现有产物 `/data/sunkaixuan/skx_log_output/csa_cut_20260920/compare/compare__model_layers_2_self_attn_attn.json`：`cosine=0.361`、`max_abs_diff=5.71`。**这个数不是 bug 信号**——`compare_once` 的 docstring (`pto_attn.py:886-891`) 自己写明它在 native 路径已经推进过六份 cache 之后才跑，输出"不具数值可比性，也不打算可比"。**绝不能拿它当 fixture 的 expected。**
- 该产物是 `b=1`（`x_normed [8,4096]` = n_real 1 × S 8），与 bs=4 的目标配置不符。

### 3. `PTO_CSA_PROBE` / `PTO_CSA_DUMP` —— 唯一一套真的落数值的，但属于另一条集成线

- `$V/vllm_ascend/attention/pto_csa.py`：`probe_attention` (`:91-135`)、`probe_oproj` (`:138-163`) 只落 shape（`_describe` 在 `numel()<=16` 且为整型时才附 `values`，`:55-60`）；产物 `/data/sunkaixuan/skx_log_output/csa_b_tier/probe1/*.json`。
- **`_dump_once` (`pto_csa.py:934-964`) 是现有唯一落真实张量值的设施**：`torch.save(payload, "step0.pt")`，配套 `vendor0.pt`（vendor 最终输出）、`vendor_attn0.pt`（vendor stage-1 注意力，逆 RoPE 前）。
- 实测产物在 `/data/sunkaixuan/skx_log_output/csa_b_tier/dump{1,2,3,4}/`。`dump1/step0.pt` 25 个 key，bs=4：`stash.q [4,64,512] bf16`、`stash.ori_kv [71894,128,1,512]`、`stash.ori_block_table [4,68] int32`、`stash.positions [4] int64`、`derived.win_new [8,128]`、`derived.ori_small [8,128,1,512]`、`derived.cmp_small [128,32,1,512]`、`derived.pto_out [4,4096]` 等。
- **局限**：(i) `_DUMPED` 全局量，**只落第 1 步**；(ii) 过滤 `numel() <= 1<<22` (`:956`)，整份 KV cache 被排除；(iii) 走的是 `PTO_CSA` 那条 vendor 级替换，**不是 `pto_attn.py` 的 46-arg 路径**，六份 cache 里的 state / inner state / indexer scale 根本不在里面；(iv) `_debug_allowed` 在 graph 模式/捕获期一律拒绝 (`pto_csa.py:180-195`)。
- 配套离线分析器已存在：`/data/sunkaixuan/codex_sh/csa_b_tier_20260917/analyze_dump.py`。

### 4. pypto / simpler 的 args dump —— **kernel 模式下不存在**

- pypto 的 `enable_dump_args`（0/1/2，产物 `<work_dir>/dfx_outputs/args_dump/args_dump.json` + .bin）只挂在 **program 模式**的 `Runner` / `device_runner` 上：`$P/python/pypto/runtime/runner.py:423,619,1441-1447`、`$P/python/pypto/runtime/device_runner.py:712,757-758,830`。
- vLLM 走的是 **kernel 模式**：`pto_attn.py:836-840` 调 `pypto.torch.init()` + `register(...)`。`init` 的签名只有 `enable_chip_swimlane` / `enable_dep_gen` / `output_dir`（`$P/python/pypto/torch/execution.py:17-25`），`KernelConfig` 也只有这三个 DFX 字段（`$P/python/pypto/runtime/kernel/abi.py:22-31`）；`grep -rn dump_args $P/python/pypto/runtime/kernel/` **零命中**。
- **结论：这条路径上没有现成的 args dump，必须自己写。** swimlane 有（`pypto.torch.begin_dfx/end_dfx`，`execution.py:117-140`；`pto_csa.py:351-493` 有完整用法），但 swimlane 只给任务时序，不给张量。

### 5. pypto 的 fixture **格式**已经定死了，照抄即可

- `$P/python/pypto/runtime/golden_writer.py:18-30`：纯输入 → `data/in/{name}.pt`；纯输出 → `data/out/{name}.pt`；**InOut → `data/in/{name}.pt`（初值）+ `data/out/{name}.pt`（golden 结果）**。落盘函数 `_save_data_files` (`:307-313`) 就是 `torch.save(tensor, data_dir/f"{name}.pt")`。
- 实例：`/data/sunkaixuan/skx_log_output/csa_b_tier/atier_20260917_185731/work/build_output/_jit_sparse_attn_test_v3ta_uq4/data/in/{attn_sink,cmp_block_table,cmp_kv,freqs_cos,freqs_sin,idx_topk,ori_kv,position_ids,q,window_swa_indices,wo_a,wo_b,wo_b_scale}.pt` + `data/out/attn_out.pt`。
- 这个约定和"state / inner state / indexer key+scale 的初值 + 两步后的期望值"**天然一一对应**：六份 cache 在 `decode_csa.py:1204,1214,1216,1217,1219,1220` 全是 `pl.InOut`。
- 同仓已有原则性结论：`/data/sunkaixuan/codex_sh/csa_b_tier_20260917/csa_a_tier_compare.py:13` —"fixture 必须取自 PTO 那一步落盘的 data/in，不能各自现造"。

---

#### b) 最小改动规格

**改一个函数：`pto_attn.substitute`（`$V/vllm_ascend/attention/pto_attn.py:966`）。沿用已有的 `PTO_ATTN_PROBE`，不新增环境变量。**

### 落点（两处，同一步内）

1. `_registered()(*args)`（`:998`）**之前** → `step{n}/in/`
2. `write_state_ring` ×2（`:1000-1003`）和 `for pg in paged: pg.commit()`（`:1004-1005`）**之后** → `step{n}/out/`

第 2 点不可提前：ring 是私有副本，`commit()` 之前六份 cache 里没有这一步的结果。

### 落什么（文件名 = `ARG_ORDER[i]`，`pto_attn.py:672-691`）

| 组 | 内容 | 每步大小（bs=4, T=32） |
|---|---|---|
| 直接存 | `position_ids`、5 个 slot mapping（`ori/cmp/idx/state/inner_state`）、`window_swa_indices [32,128]`、`kv_seq_lens`、`cmp_block_table [4,8192]`、`idx_block_table`、两个 state block table、`x_normed [32,4096]`、`attn_out [32,4096]`、4 张 rope 表 | < 1 MB |
| 直接存 | `compress_state [32,2,2048] fp32`、`inner_compress_state [32,2,512] fp32` | 640 KiB |
| 直接存 | `idx_kv_cache`、`idx_kv_scale`（已被 `repage_kv` 压实，`pto_attn.py:580-610`） | ≈ 4.6 MB |
| **只存被触及的行** | `kv_cache` / `cmp_kv`：按 `Paged._slots` + `window_swa_indices` gather 出来的行，**连同索引向量一起存** | ≈ 4 MB |
| 一次性 | 23 个权重 arg（`prepare_weights`，`:163-206`），只在 `step0/` 存一份 | ≈ 140 MB |
| 每步 | `meta.json`：`n_real`、`ks`、`capture_active()`、`_RAN[0]`、`pos[0]`、`plan_m/plan_i` 的 `(blk,intra,ring_rows,valid)`、`state_c`/`ist_c` 被 plan 点名的原始行 | 小 |

**绝不能整份存 `kv_cache` / `cmp_kv`**：实测 `kernel_details.csv` 里 vendor 的 `ori_kv` 是 `34730,128,1,512`，repage 后 `[138920,32,1,512] bf16` ≈ 4.5 GB/份。`_dump_once` 的 `numel() <= 1<<22` 过滤（`pto_csa.py:956`）正是为这个而设，照抄。

`substitute` 的 locals 里已经全部拿得到：`plan_m, plan_i, state_c, ist_c, main_dim, inner_dim, pos, ks, n_real, paged = plan`（`:993`）。

### 选哪两步

**不要用"前两次调用"。** `_RAN[0]` 把捕获期也算进去（实测 6 次，见 e-2）。规则：

```python
if (not capture_active()) and decode is not None and _FIX[0] < 2:
```

并且 **必须关掉客户端 warmup**：`PROFILE=1` 时是 `--warmup "${PROFILE_WARMUP:-1}"`（`dsv4_case_inner.sh:252`），`PROFILE_WARMUP` 在 `run_dsv4_mtp_vllm.sh:81` 有 `--env` 透传，设 0 即可。`PROFILE=0` 那条走 `--warmup-batches ${WARMUP_BATCHES:-5}`（`dsv4_case_inner.sh:273`），而 `WARMUP_BATCHES` **没有**被 `run_dsv4_mtp_vllm.sh` 透传 → 只能改脚本或走 `PROFILE=1`。**推荐 `PROFILE=1 PROFILE_WARMUP=0`。**

### 落盘格式

每步一个目录，内部 `in/` + `out/`，文件名取 `ARG_ORDER`，即 pypto `golden_writer` 的约定 —— lib 侧现有工具可直接加载。目录结构：

```
$PTO_ATTN_PROBE/model_layers_2_self_attn_attn/
    weights/{wq_a,wq_b,...}.pt      # 一次
    step0/in/*.pt   step0/out/*.pt   step0/meta.json
    step1/in/*.pt   step1/out/*.pt   step1/meta.json
```

### 运行命令

```bash
PROFILE=1 PROFILE_WARMUP=0 PROFILE_TOKENS=16 \
PTO_ATTN_REPLACE=1 PTO_ATTN_TP=4 PTO_ATTN_SEQ=1 \
PTO_ATTN_PROBE=/data/sunkaixuan/skx_log_output/csa_fixture_<ts> \
SERVE_EXTRA="--enforce-eager" \
DEVICE_NUM=1 BATCH_SIZE=4 PROMPT_TOKENS=1535 MAX_TOKENS=16 \
MAX_MODEL_LEN=8704 GPU_UTIL=0.6 \
MODEL=/data/sunkaixuan/skx_log_output/dsv4_vllm/models/official-l3 \
/data/sunkaixuan/codex_sh/own_stack_20260918/dsv4/run_dsv4_mtp_vllm.sh
```

`SERVE_EXTRA` 在 `dsv4_case_inner.sh:81-83` 被拼进 `EXTRA_ARGS` → `vllm serve`（`:160-182`）。`--enforce-eager` **不是可选项**，理由见 e-1。

---

#### c) 时间与占用

**实测**（`fdoprof/prof_pto/dsv4_mtp_vllm_20260920_192612_dp1_bs4/`）：

- `task_submit.log` 提交时刻 `2026-09-20T19:26:13-07:00`（`run_meta.txt` 末行），`end_time.txt` = `19:27:55` → **端到端 102 s**，含取设备锁、起服务（`wait_health.log` 到 READY 是 iteration=30）、ACLGraph 捕获（`server_final.log:160` "Graph capturing finished in 20 secs, took 3.67 GiB"）、客户端跑完、停服。
- native 对照：`prof_native/.../end_time.txt` = `19:26:04`，目录时间戳 19:24:38 → **86 s**。
- 客户端本身：`profile_client.json` `wall_s: 5.907`，`decode_tokens_captured: 8`，TTFT 0.274 / 0.533 s。

**加 `--enforce-eager` 后**：省掉 20 s 捕获和 3.67 GiB，但每个 decode step 要跑一遍 Python `build_args`。按 profile 里实测的换算层算子量（每步约 27 次 `aclnnArange`、26 次 `aclnnIndexSelect` 等），16 步的额外 wall 是秒级。**总计仍在 2 分钟内，预算 3 分钟足够。**

**占用**：`task-submit --device auto --device-num 1`，整轮持有 **1 张 die**（实测 `ASCEND_RT_VISIBLE_DEVICES=15`）；HBM 按 `--gpu-memory-utilization 0.6`；8113–8140 里一个端口（`run_dsv4_mtp_vllm.sh:24-30` 自动挑）；设备日志落 `$OUT/ascend`（`run_dsv4_mtp_vllm.sh:41-43`）。

**产物体积**：每步约 10 MB + 一次性权重 140 MB → 两步 fixture ≈ **160 MB**。

---

#### d) 边界覆盖 —— 默认跑法只能命中 6 个里的 3 个半

### 常量（全部来自代码）

| 量 | 值 | 锚点 |
|---|---|---|
| vLLM KV 页 | 128 slot | `pto_attn.py:64` |
| vLLM state 页 | 8 行 | `pto_attn.py:65` |
| kernel KV 页 | 32 slot | `config.py:258` |
| kernel state 页 | 2 行 | `pto_attn.py:341` / `config.py:259` |
| ring 长度 | 16 = `MAIN_STATE_LEN(8) + S(8)` | `decode_csa.py:160-161` |
| ring 寻址 | `ring_row = logical_pos % 16`；`state_row = bt[req, ring_row//2]*2 + ring_row%2` | `decode_compressor_ratio4.py:192-201` |
| 压缩边界 token | `(pos+1) % 4 == 0` | `pto_attn.py:271-276` |
| padding lane | T 槽 t 有效 iff `t % S == 0`，S=8 | `pto_attn.py:692-711` |

记 prompt 实际长度 P（prefill 覆盖 0..P-1），第 k 个 decode step 的位置 `pos_k = P-1+k`。

### 默认 P≈1024 / MAX_TOKENS=32（pos 1024..1054，约 31 步）

| 边界 | 条件 | 默认跑法 | 说明 |
|---|---|---|---|
| **8 行逻辑页沿**（vLLM state） | `pos ≡ 0 (mod 8)` | ✅ **命中 4 次**：step 1, 9, 17, 25 | 另外 kernel 的 2 行页每 2 步跨一次，必中 |
| **16 行 ring wrap** | `pos ≡ 0 (mod 16)` | ✅ **命中**：step 16→17（pos 1039 ring 15 → 1040 ring 0） | |
| **128 行 KV 页沿** | `pos ≡ 0 (mod 128)` | ❌ **只在 step 1**（pos 1024=8×128），下一个 1152 = step 129 | run 内部不跨。kernel 的 32-slot 页沿同理：1024 之后是 1056 = step 33，**刚好越界一步** |
| **indexer key/scale 换页** | 写入行落新页：`pos = 512m + 3`；跨页的两次**写入**在 `pos = 512m−1` 与 `512m+3`，相隔 4 步 | ❌ 最近一对是 1535/1539 = step 512/516 | 差两个数量级 |
| **inactive token** | 每步 8 个 T 槽里 7 个是 padding | ✅ **任何一步必中** | 另一类"inert 请求"见下 |
| **多请求落不同物理页** | bs=4 + `--no-enable-prefix-caching`（`dsv4_case_inner.sh:180`） | ✅ **必中**，实测 block table 是 `4,68`（`kernel_details.csv` 里 `SparseAttnSharedkv` 的 Input Shapes） | |

附带命中：**压缩节律**（`pos ≡ 3 mod 4`，step 4/8/12/…）—— 四步里只有一步真写 cmp/idx，另外三步 slot mapping 是 −1。两步 fixture 若想同时覆盖"写"和"不写"，必须取 step 3→4。

### 一次跑全命中：**P = 1535**

`P = 1535` 时 `pos_1 = 1535`，`pos_2 = 1536`，而 1536 同时满足：

- `1536 = 12 × 128` → **128-slot KV 页沿**（step 1→2）
- `1536 % 32 = 0` → kernel 32-slot 页沿
- `1536 % 16 = 0` → **ring wrap**
- `1536 % 8 = 0` → **8 行 state 页沿**
- `1536 / 4 = 384 = 3 × 128` → 压缩 KV 128 行页沿
- `1535 % 4 = 3`、`1539 % 4 = 3` → 两次**写入** indexer 的位置分别落在 idx 页 2 的最后一行和页 3 的第一行 → **indexer key/scale 换页**（step 1 与 step 5）

加上 bs=4（多请求不同物理页）和恒有的 padding lane，**`PROMPT_TOKENS=1535, MAX_TOKENS≥10, BATCH_SIZE=4` 一次跑覆盖全部六项**；其中四项仅靠 step 1→2 这一对就能覆盖。

通用式：该边界集合以 512 为周期，`P ∈ {511, 1023, 1535, 2047, ...}` 都行；取 1535 是为了让 prefill 在 `max_model_len=8704` 内且和现有 1024 的配置量级相近。

**第六项"inactive token"的第二种形态（整个请求是 padding 行）**：capture sizes 是 `[1,2,4]`、`max_num_seqs=4`，所以活跃请求数 3 时会重放 size-4 的图，多出一行 inert 请求。实测已自然出现（`client.log` 四个请求拿到 31/31/**30**/31 个 token）。要确定性复现：`BATCH_SIZE=3`。**但注意**：`--enforce-eager` 下没有图 padding，`n_offered` 就是活跃数，这一形态消失（见 e-9）。

### 用 prompt 长度定位边界不可靠

`profile_decode_client.build_prompt`（`:36-41`）是"造一段长度**大致**可控的 prompt"——拼随机词，token 数不精确。非 profiling 客户端 `dsv4_mtp_vllm_client.py:106-116` 走 detokenize 往返，注释明说"往返可能差一两个 token"，长度不符只打印告警。

**所以：不要事前按 `PROMPT_TOKENS` 算哪一步命中哪个边界。每步的 `meta.json` 落 `position_ids`，事后按 pos 挑步。** 把 `MAX_TOKENS` 放宽到 16 给 ±3 的余量。

---

#### e) 天真地产出会错在哪

### 1. 默认配置下 fixture 根本产不出来（最严重）

`cudagraph_mode=FULL_DECODE_ONLY`、capture sizes `[1,2,4]`、`max_num_seqs=4` → **每一个 decode step 都是图重放，Python 不执行**。

实测铁证（`prof_pto/.../server_final.log`）：`[pto-attn-ran]` 只出现 6 次（`:146-158`），**全部在 "Capturing CUDA graphs" 期间**；而 `kernel_details.csv` 显示服务阶段 `simpler_aicpu_kernel_exec_891fbc868201ba61` 与 `aicore_kernel_mode_0_mix_aic` **各执行 8 次**（= 8 个被 profile 的 decode token）。`_RAN[0] <= 5 or % 10 == 0`（`pto_attn.py:1010`）本会打印 n=10/20，没有 → Python 确实没跑。

旁证（替换确实生效）：vendor `SparseAttnSharedkv` 在 native 跑了 **24** 次（8 步 × 3 层），在 pto 跑 **16** 次（8 步 × 2 层），且 ratio-4 那一种入参形状 `"4,64,512;34730,...;4,68;4,68;..."` 在 pto 侧**完全消失**；native-only 的 `Compressor`(16)、`QuantLightningIndexer`(8) 也消失了。

→ **必须 `--enforce-eager`。** `debug_allowed`（`pto_attn.py:475-487`）在捕获期会拒绝并打印一行，但那只覆盖"捕获中"，覆盖不了"整个 decode 都是重放、代码根本不到"。

### 2. 捕获期的 6 次调用已经把六份 cache 写脏了

那 6 次（每个 capture size 一次 warm-up + 一次录制）都真的跑了 `substitute`、真的写了 cache。vLLM 的 dummy metadata 里 `self.slot_mapping = torch.zeros(...)`（`dsa_v1.py:434`），warmup 路径更是显式 `slot_mapping_dummy = torch.zeros(1)` 去写 indexer（`dsa_v1.py:1506-1514`）。

→ **六份 cache 的物理页 0 / 行 0 含捕获期垃圾**。而 `Paged.commit` 又特意把 inert 行"停泊"在行 0（`pto_attn.py:549-560` docstring）。**fixture 里把页 0 当"初始内容"或"两步后期望"就是错的**，要么排除，要么单独标注。

### 3. padding lane 不是 0

`rectangular`（`:692-711`）+ `index_select(0, src*seq)`：padding 槽的 `x_normed`、`freqs_*`、`position_ids` 是**它所属请求那一行的拷贝**，不是零；只有 5 个 slot mapping 被 `_inert` 写成 −1（`:784-786, 803-805`）。消费方若假设 padding 行为零，算出来的 golden 会不一样。

反过来，padding 行的 `attn_out` 是 kernel 写的任意值，宿主侧用 `take = arange(n_real)*ks` 丢掉（`:1007-1009`），**不可参与比对**。

### 4. 六份 cache 原地写 → step0 的 out 必须等于 step1 的 in

`decode_csa.py:1204,1214,1216,1217,1219,1220` 六个 `pl.InOut`。这给了 fixture **自带的一致性校验**：两步都触及的行上，`step0/out/X.pt` 必须逐位等于 `step1/in/X.pt`；不等就是两份 dump 至少有一份取错了时机。前提是 step0 的 out 取在 `write_state_ring` + `commit()` **之后**（`:1000-1005`）。

### 5. state ring 是私有副本，而且**前瞻的 7 行是陈旧数据**

`make_state_ring`（`:383-393`）每步从 vLLM 的 strided cache 新建一份连续 `[b*8,2,dim]`，`write_state_ring`（`:395-434`）再写回。所以"state 的初始内容"是二义的：**必须同时落 vLLM cache 里被 `plan` 点名的原始行，和交给 kernel 的那份 ring**，否则二者之间的映射错了也看不出来。

更要紧的是 `state_ring_plan`（`:349-374`）里 `pos = first - (_RR - seq) + i`，调用处传的 `seq` 是 **`ks`=S=8**（`:774-775`），不是宿主的 seq=1。于是窗口是 `first-8 .. first+7`——**包含 7 个尚不存在的未来位置**，`ok` 判据里没有 `pos <= first` 这一条。这 7 行是从 vLLM state cache 的对应行读出来的**残留值**（上一个占用同一物理页的请求写的）。

→ **fixture 不自洽**，除非把这 7 行原样落盘。任何"从第一性原理重建初始 state"的尝试都会得到不同的数。（这一条同时是 2/3/4 项要确认的内容，此处只说它对 fixture 的后果。）

### 6. 权重是运行期现量化的

`prepare_weights` 缓存在 `impl._pto_attn_weights`（`:163-206`）；`_int8`（`:125-146`）在 checkpoint 是稠密时会**当场量化**（`_quant_int8_per_channel`，`:91-97`）。→ **权重必须随 fixture 落盘**（或至少落哈希），lib 侧从 checkpoint 自行重算可能不一致。

### 7. `kv_seq_lens` 的含义随模式而变

`a["kv_seq_lens"] = cmp_md.seq_lens.to(int32)[:b]`，`b = n_real`（`:802`）。图 padding 下 `n_real` 是**补齐后的**批大小，于是这里会带上不存在的请求；eager 下是真实批。→ `meta.json` 必须同时记 `capture_active()` 和 `n_real`。

### 8. 别拿 `compare_once` 的数当基准

见 a-2。`cosine=0.361` 是设计使然，不是缺陷。

### 9. `--enforce-eager` 自己改变了被测对象

关图之后没有批 padding，"inert 请求"这一边界消失；同时 fixture 描述的也不再是生产实际重放的东西。→ **必须在 fixture 里声明它取自 eager 模式**，要覆盖 inert 请求就另开一轮 `BATCH_SIZE=3`（或错开各请求的 max_tokens）并接受它只能在捕获期取到（那又与 e-1/e-2 冲突）——这是一个真实的取舍，不要糊过去。

### 10. prompt 长度不精确

见 d 末。**按 `position_ids` 事后挑步，不要事前按 `PROMPT_TOKENS` 推断。**


#### FACTS
- 读码：PTO_ATTN_PROBE 只驱动一次性结构 dump —— /data/.../vllm-ascend-v0.20.2rc1/vllm_ascend/ops/dsa.py:209-219 用 `_PROBED` 保证每 layer_name 只触发一次，且调用点在 substitute 之前。
- 读码：dump_structure 只落 shape/dtype/contig/stride —— vllm_ascend/attention/pto_attn.py:653-668 调用 describe（同文件 :626-651），describe 对 Tensor 只返回 {shape,dtype,contig,stride}，没有 values / storage_offset / data_ptr。
- 实测：现有三份 probe 产物全是 prefill —— /data/sunkaixuan/skx_log_output/csa_cut_20260920/probe/model_layers_{0,1,2}_self_attn.json 里所有 metadata 组的 decode 均为 null，attn_state=PrefillNoCache，num_decodes=0。对 decode fixture 零价值。
- 读码：compare_once 的输出按设计不可比 —— pto_attn.py:886-891 docstring 写明它在 native 路径已经推进过六份 cache 之后才跑。
- 实测：/data/sunkaixuan/skx_log_output/csa_cut_20260920/compare/compare__model_layers_2_self_attn_attn.json 记录 cosine=0.361、max_abs_diff=5.71、n_real=1（x_normed [8,4096] = 1×S8），不能作 golden，且批大小与 bs=4 目标不符。
- 读码：现有唯一落真实张量值的设施是 pto_csa.py:934-964 的 _dump_once（torch.save step0.pt + vendor0.pt + vendor_attn0.pt），但 _DUMPED 全局量使其只落第 1 步，且 :956 过滤 numel() <= 1<<22 排除整份 KV cache。
- 实测：该设施的产物存在于 /data/sunkaixuan/skx_log_output/csa_b_tier/dump{1,2,3,4}/step0.pt，dump1 有 25 个 key（stash.q [4,64,512] bf16、stash.ori_block_table [4,68]、derived.ori_small [8,128,1,512]、derived.cmp_small [128,32,1,512]、derived.pto_out [4,4096] 等），但走的是 PTO_CSA vendor 级替换线，不含 state / inner state / indexer scale。
- 读码：pypto 的 enable_dump_args（args_dump/）只存在于 program 模式 —— python/pypto/runtime/runner.py:423,619,1441-1447 与 runtime/device_runner.py:712,757-758,830。
- 读码：kernel 模式没有 args dump —— python/pypto/torch/execution.py:17-25 的 init() 只有 enable_chip_swimlane / enable_dep_gen / output_dir；python/pypto/runtime/kernel/abi.py:22-31 的 KernelConfig 同样只有这三个 DFX 字段；grep dump_args 在 pypto/runtime/kernel/ 下零命中。
- 读码：pypto 的 fixture 格式约定在 python/pypto/runtime/golden_writer.py:18-30 —— 纯输入落 data/in/{name}.pt，纯输出落 data/out/{name}.pt，InOut 两边都落（in 为初值、out 为 golden）；落盘实现 :307-313。
- 实测：该格式的实例在 /data/sunkaixuan/skx_log_output/csa_b_tier/atier_20260917_185731/work/build_output/_jit_sparse_attn_test_v3ta_uq4/data/in/（13 个 .pt）与 data/out/attn_out.pt。
- 读码：六份 cache 全是 pl.InOut —— decode_csa.py 的 decode_csa_attn_tp1_test 签名里 :1204(compress_state) :1214(inner_compress_state) :1216(kv_cache) :1217(cmp_kv) :1219(idx_kv_cache) :1220(idx_kv_scale)。
- 读码：ring 长度 16 = MAIN_STATE_LEN(=COFF*COMPRESS_RATIO=8) + S(=8)，decode_csa.py:160-161；ring 寻址 ring_row = logical_pos % 16、state_row = bt[req, ring_row//2]*2 + ring_row%2，decode_compressor_ratio4.py:192-201。
- 读码：VLLM_PAGE=128、VLLM_STATE_PAGE=8、COMPRESS_RATIO=4 在 pto_attn.py:64-66；KERNEL_STATE_PAGE=2 在 pto_attn.py:341；BLOCK_SIZE=32 与 C4A_COMPRESSOR_BLOCK_SIZE=2 在 pto_kernels/dspark/config.py:258-259。
- 读码：压缩边界 token 判据 boundary = ((pos+1) % 4) == 0，pto_attn.py:271-276；cmp 与 idx 的 slot mapping 都经 _to_token_rows 按该 boundary 填 -1，pto_attn.py:789-793。
- 读码：padding lane —— rectangular（pto_attn.py:692-711）令 T 槽 t 有效 iff t % S == 0（S=8），非有效槽由 _inert（:784-786, :803-805）在 5 个 slot mapping 上写 -1；但 x_normed/freqs/position_ids 是该请求真实行的拷贝，不是零。
- 读码：state_ring_plan（pto_attn.py:349-374）被调用时传的是 ks=S=8（:774-775），于是种子窗口是 first-8 .. first+7，包含 7 个尚不存在的未来位置，ok 判据无 pos<=first 这一条 —— 这 7 行取自 vLLM state cache 的残留值。
- 读码：substitute 的写回顺序 —— _registered()(*args) 在 pto_attn.py:998，write_state_ring ×2 在 :1000-1003，for pg in paged: pg.commit() 在 :1004-1005；step 的 out 必须取在 :1005 之后。
- 读码：kv_seq_lens 被截成 [:b]，b = n_real（pto_attn.py:802），图 padding 下 n_real 是补齐后的批大小。
- 读码：debug_allowed（pto_attn.py:475-487）在 ACLGraph 捕获期拒绝任何读回主机的诊断，并解释捕获期读回会把当时的值烘进图里。
- 实测：默认 FULL_DECODE_ONLY 下 decode step 不跑 Python —— prof_pto/.../server_final.log:141-158 的 [pto-attn-ran] 只出现 6 次且全在 "Capturing CUDA graphs" 期间（n=1..6，capturing 交替 False/True），服务阶段再无；而 kernel_details.csv 显示服务期 simpler_aicpu_kernel_exec_891fbc868201ba61 与 aicore_kernel_mode_0_mix_aic 各执行 8 次。
- 实测：替换在被 profile 的 decode 步确实生效 —— vendor SparseAttnSharedkv 在 prof_native 执行 24 次（含 ratio-4 那一种入参形状），在 prof_pto 只执行 16 次且 ratio-4 形状完全消失；native-only 的 Compressor(16) 与 QuantLightningIndexer(8) 在 pto 侧也消失。
- 实测：msprof 对我们的 kernel 不给形状 —— kernel_details.csv 里 simpler_aicpu_kernel_exec_* 行的 Input Shapes / Output Shapes 都是 N/A（Duration 950.979us，AI_CPU）。
- 实测：KV cache 规模使整份落盘不可行 —— prof 里 vendor 的 ori_kv 形状是 34730,128,1,512 bf16，repage 成 32-slot 页后约 4.5 GB/份。
- 读码：vLLM dummy/warmup 用 0 号 slot —— dsa_v1.py:434 self.slot_mapping = torch.zeros(...)，dsa_v1.py:1506-1514 warmup 显式 slot_mapping_dummy = torch.zeros(1) 写 indexer k/scale cache；配合 Paged.commit 把 inert 行停泊在行 0（pto_attn.py:549-560），页 0/行 0 是被污染的。
- 实测：一轮 PTO profiling 端到端 102 s —— prof_pto/dsv4_mtp_vllm_20260920_192612_dp1_bs4/run_meta.txt 末行 2026-09-20T19:26:13-07:00，end_time.txt 19:27:55；native 对照 prof_native 的 end_time.txt 19:26:04（约 86 s）。
- 实测：其中 ACLGraph 捕获占 20 s —— prof_pto/.../server_final.log:160 "Graph capturing finished in 20 secs, took 3.67 GiB"。
- 实测：客户端本体只跑 5.907 s，抓到 8 个 decode token —— prof_pto/.../profile_client.json 的 profiled.wall_s 与 decode_tokens_captured。
- 实测：bs=4 且各请求 token 数不齐（31/31/30/31）—— prof_pto/.../client.log，说明存在活跃请求数 <4 的 step，会重放 size-4 的图并带一行 inert 请求。
- 实测：block table 是 4 行 68 列 —— kernel_details.csv 里 SparseAttnSharedkv 的 Input Shapes 含 "4,68"，对应 bs=4、max_model_len 8704 / 128。
- 读码：--enforce-eager 可经 SERVE_EXTRA 注入 —— /data/sunkaixuan/codex_sh/own_stack_20260918/dsv4/dsv4_case_inner.sh:61,81-83 把 SERVE_EXTRA 拼进 EXTRA_ARGS，:160-182 是 vllm serve 命令。
- 读码：PTO_ATTN_* 由 /data/sunkaixuan/codex_sh/own_stack_20260918/dsv4/run_dsv4_mtp_vllm.sh:74-79 透传（PROBE/COMPARE/REPLACE/SEQ/TP），PROFILE_WARMUP 由 :81 透传；WARMUP_BATCHES 未被透传，只能在 dsv4_case_inner.sh:273 里取默认 5。
- 读码：prompt 长度不精确 —— profile_decode_client.py:36-41 build_prompt 拼随机词，注释自称"长度大致可控"；dsv4_mtp_vllm_client.py:106-116 走 detokenize 往返，注释写明"往返可能差一两个 token"，不符只打印告警。
- 计算（基于上述常量）：P=1535 时 pos_2=1536 同时满足 1536%128==0（KV 页沿）、%32==0（kernel 页沿）、%16==0（ring wrap）、%8==0（state 页沿）、1536/4=384 且 384%128==0（cmp 页沿）；而 1535%4==3 与 1539%4==3 是跨 idx 页的两次写入位置 —— 故 PROMPT_TOKENS=1535 / MAX_TOKENS>=10 / BATCH_SIZE=4 一次覆盖全部六项边界。
- 计算：默认 P≈1024 / MAX_TOKENS=32（pos 1024..1054）只命中 8 行 state 页沿（step 1/9/17/25）、16 行 ring wrap（step 16→17）、padding lane、多请求不同物理页；128 行 KV 页沿仅在 step 1（下一个 1152=step 129），kernel 32-slot 页沿下一个是 1056=step 33 恰好越界一步，indexer 换页最近一对在 pos 1535/1539 = step 512/516。
- 读码：ARG_ORDER 的 46 个名字定义在 pto_attn.py:672-691，可直接用作 data/in、data/out 的文件名；substitute 的 plan 解包在 :993 已提供 plan_m/plan_i/state_c/ist_c/main_dim/inner_dim/pos/ks/n_real/paged。

#### NEEDS_LIVE
- 第一个 decode 位置的真实值：在 pto_attn.py:966 `substitute` 里、`decode = metadata_list[0].decode` 之后，打印 `decode.input_positions[:8]` 与 `.shape`。决定 d) 里所有 mod 运算落在第几步；prompt 长度不精确使其无法事前推算。
- 捕获期/warmup 的 slot_mapping 到底是 0 还是 -1：在 pto_attn.py:712 `build_args` 开头，当 `capture_active()` 为真时打印 `swa_md.slot_mapping[:4]`、`cmp_md.slot_mapping[:4]`、`idx_md.slot_mapping[:4]` 与 `cst_md.block_table[0]`。决定六份 cache 的物理页 0 是否真被捕获期写脏（e-2 的前提）。
- indexer key 的 slot_mapping 是按压缩行还是按 token 打包：在 pto_attn.py:735-740 打印 `idx_md.slot_mapping.shape[0]`、`cmp_md.slot_mapping.shape[0]`、`host_pos.shape[0]` 三者。若按 token，则 d) 里 indexer 换页的位置从 pos≡0 (mod 512) 塌缩到 pos≡0 (mod 128)，1535 这个 prompt 长度就选错了。代码两侧证据相反：pto_attn.py:791-793 按 boundary/row 走压缩节律，decode_csa.py:172 的 IDX_MAX_BLOCKS==CMP_MAX_BLOCKS 也指向压缩行；但实测 idx block table 是 68 列（= 8704/128，按 token 的容量）。
- compressor state 的行索引是 token 位置还是压缩行：在 pto_attn.py:757-758（unpack 处）与 :774-775 打印 `cst_md.block_table[0]` 与 `state_ring_plan` 返回的 `blk/intra` 前 16 项，并与同一步 native 路径 Compressor 算子的写入位置对照。`state_ring_plan` 用 token 位置直接除 8，若实际是压缩行则整条 seeding 都偏了。
- state ring 的"未来 7 行"到底读到了什么：在 pto_attn.py:383-393 `make_state_ring` 返回前，打印 ring 中 `pos > first` 那 7 行的 absmax / 是否全零。若非零，fixture 必须原样落这 7 行（e-5）。
- 六份 cache 的 storage_offset 与 data_ptr，以及 idx_k_c / idx_s_c 是否同一块分配：`describe`（pto_attn.py:626-651）两者都不记，需在那里补 `storage_offset()` 与 `data_ptr()`，并在 dsa.py:205 `_build_kv_cache` 返回后逐个打印。代码（pto_attn.py:584-588）称二者共用一个 16640 字节的页。
- 四个请求各自拿到的 block table 行：在 pto_attn.py:747-752 每步打印 `swa_md.block_table`、`cmp_md.block_table`、`idx_md.block_table`、`cst_md.block_table`、`ist_md.block_table` 的前若干列。没有这个就无法判断"多请求落不同物理页"是否真的发生，也无法离线重建小 cache。
- `--enforce-eager` 下 `_RAN[0]` 是否从第一个真实 decode step 起算：设 `SERVE_EXTRA=--enforce-eager PROFILE_WARMUP=0` 跑一轮，读 pto_attn.py:1010-1012 已有的 `[pto-attn-ran] n=... capturing=...` 输出。决定 b) 里选哪两步的门限写法。

#### UNKNOWNS
- indexer key/scale 的写入节律究竟是按压缩行还是按 token —— 两处代码证据相反（pto_attn.py:791-793 + decode_csa.py:172 指向压缩行；实测 idx block table 68 列指向 token）。这直接决定 d) 里"indexer 换页"该选哪个 prompt 长度，是本项唯一影响结论的未决点。
- compressor state 的 cache 行索引语义（token 位置 vs 压缩行）。pto_attn.py:349-374 按 token 位置除 VLLM_STATE_PAGE=8 寻址，但这一点属于第 2/3/4 项的职责，本项只依赖"ring_row = pos % 16"（该式在 decode_compressor_ratio4.py:192 有直接锚点）。
- prompt 实际 tokenize 后的长度（因此第一个 decode 位置）。两个客户端都只能近似控制，需事后从 position_ids 读回。
- 捕获期 6 次调用是否真的把物理页 0 写脏 —— dsa_v1.py:434 的零初始化与 :1506-1514 的 slot_mapping_dummy 是强证据，但未读到活体张量确认。
- `--enforce-eager` 下 vLLM 是否仍会做批 padding（若会，e-9 里"inert 请求消失"的结论需修正）。
- fixture 每步体积的估算基于 bs=4 / T=32 与实测 block table 68 列推算，未实际落盘验证；权重一次性约 140 MB 是按 compare 记录里的 arg 形状算出的。

</details>

---

# 待办

七项全部完成复核。第 3、4、8 项复核未通过，已按更正改写，原答复折叠保留。

仍待补的是几处只能从活跃张量上读的：各张量的 `data_ptr` 关系、各块表的实际示例值，
以及一份覆盖边界的 fixture。所需的转储内容各项的「需要活体读取」一节已列明；
第 8 项的边界覆盖算式以复核更正为准。
