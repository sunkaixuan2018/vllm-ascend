"""把 pypto-lib 的 PTO CSA kernel 接到 vLLM 的 decode 路径上。

PTO 的 `sparse_attn_test` 把 注意力 + 逆 RoPE + grouped o_proj 融在一个 kernel 里，
正好覆盖 `AscendDSAImpl.forward` 里 decode 分支的这一段：

    _forward_decode(...)                       -> o_proj_input       # 注意力
    inplace_partial_rotary_mul(o_proj_input, cos, -sin, "interleave")# 逆 RoPE
    npu_transpose_batchmatmul(o_proj_input, wo_a.weight, ...)        # wo_a
    wo_b(o_proj_input)                                               # wo_b

所以替换点在**层级**，不是算子级 1:1 对换 —— vendor 的 `npu_sparse_attn_sharedkv`
只做到第一行。

两个开关，互不依赖：
  `PTO_CSA_PROBE=<dir>`  只读探针，把上面两处调用点的张量长相落盘。
  `PTO_CSA=1`            真正替换 ratio-4 层的 decode 路径。
两个都不设时，本文件对 vLLM 没有任何影响。
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

from vllm_ascend import envs

_LOCK = threading.Lock()
_SEEN: set[str] = set()


# ---------------------------------------------------------------------------
# 探针
# ---------------------------------------------------------------------------


def probe_dir() -> Path | None:
    """`PTO_CSA_PROBE` 指向的落盘目录；没设就返回 None（探针整体关闭）。"""
    d = os.environ.get("PTO_CSA_PROBE", "").strip()
    return Path(d) if d else None


def _describe(obj, depth: int = 0):
    """把张量/模块摊成可 JSON 化的形状描述；标量原样，其余给类型名。"""
    import torch

    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        d = {"shape": list(obj.shape), "dtype": str(obj.dtype), "device": str(obj.device)}
        if obj.numel() and obj.numel() <= 16 and obj.dtype in (torch.int32, torch.int64, torch.int16, torch.int8):
            d["values"] = obj.flatten().tolist()
        return d
    if isinstance(obj, (int, float, bool, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_describe(x, depth + 1) for x in obj[:8]]
    if depth < 2 and hasattr(obj, "__class__"):
        out = {"__class__": type(obj).__name__}
        for attr in ("weight", "weight_scale", "weight_offset", "bias", "input_scale", "deq_scale", "scale"):
            if hasattr(obj, attr):
                out[attr] = _describe(getattr(obj, attr), depth + 1)
        return out
    return {"__class__": type(obj).__name__}


def _emit(tag: str, layer_name: str, payload: dict) -> None:
    """一层一个 tag 只落一次盘，避免每步都写。"""
    d = probe_dir()
    if d is None:
        return
    key = f"{tag}:{layer_name}"
    with _LOCK:
        if key in _SEEN:
            return
        _SEEN.add(key)
    d.mkdir(parents=True, exist_ok=True)
    safe = layer_name.replace(".", "_").replace("/", "_")
    (d / f"{tag}__{safe}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(f"[pto-csa-probe] {tag} {layer_name} -> {d}", flush=True)


def probe_attention(
    impl,
    layer_name,
    *,
    q,
    ori_kv,
    cmp_kv,
    cmp_sparse_indices,
    ori_block_table,
    cmp_block_table,
    cu_seqlens_q,
    seqused_kv,
    sinks,
    softmax_scale,
    positions=None,
) -> None:
    """`_forward_decode` 里 vendor CSA 调用点的入参长相。"""
    if probe_dir() is None:
        return
    _emit(
        "attn",
        layer_name,
        {
            "layer_name": layer_name,
            "compress_ratio": getattr(impl, "compress_ratio", None),
            "n_local_heads": getattr(impl, "n_local_heads", None),
            "head_dim": getattr(impl, "head_dim", None),
            "nope_head_dim": getattr(impl, "nope_head_dim", None),
            "rope_head_dim": getattr(impl, "rope_head_dim", None),
            "window_size": getattr(impl, "window_size", None),
            "index_topk": getattr(impl, "index_topk", None),
            "softmax_scale": float(softmax_scale),
            "q": _describe(q),
            "ori_kv": _describe(ori_kv),
            "cmp_kv": _describe(cmp_kv),
            "cmp_sparse_indices": _describe(cmp_sparse_indices),
            "ori_block_table": _describe(ori_block_table),
            "cmp_block_table": _describe(cmp_block_table),
            "cu_seqlens_q": _describe(cu_seqlens_q),
            "seqused_kv": _describe(seqused_kv),
            "sinks": _describe(sinks),
            "positions": _describe(positions),
        },
    )


def probe_oproj(impl, layer_name, *, o_proj_input, cos, sin, decode_tokens, actual_tokens, num_tokens) -> None:
    """`forward` 里逆 RoPE + o_proj 段的入参长相（= PTO kernel 的后半段）。"""
    if probe_dir() is None:
        return
    _emit(
        "oproj",
        layer_name,
        {
            "layer_name": layer_name,
            "compress_ratio": getattr(impl, "compress_ratio", None),
            "n_local_groups": getattr(impl, "n_local_groups", None),
            "o_lora_rank": getattr(impl, "o_lora_rank", None),
            "n_local_heads": getattr(impl, "n_local_heads", None),
            "head_dim": getattr(impl, "head_dim", None),
            "nope_head_dim": getattr(impl, "nope_head_dim", None),
            "decode_tokens": int(decode_tokens),
            "actual_tokens": int(actual_tokens),
            "forward_context_num_tokens": int(num_tokens),
            "o_proj_input": _describe(o_proj_input),
            "cos": _describe(cos),
            "sin": _describe(sin),
            "wo_a": _describe(getattr(impl, "wo_a", None)),
            "wo_b": _describe(getattr(impl, "wo_b", None)),
            "attn_sink": _describe(getattr(impl, "attn_sink", None)),
        },
    )


# ---------------------------------------------------------------------------
# 替换
# ---------------------------------------------------------------------------

_RUNNER = None
_DUMPED = False
# 逐入参一致性校验。默认关，开了每步都跑（成本是几 MiB 的 gather）。
_VERIFY = os.environ.get("PTO_CSA_VERIFY", "").strip() not in ("", "0", "false", "False")
_DEBUG_REFUSED = False


def _capture_active() -> bool:
    """当前是否正在 ACLGraph 捕获期。

    两套 runner 的标记位置不同：v1/piecewise 由 `acl_graph.py` 置
    `forward_context.capturing`，v2 full-graph 由 `worker/v2/aclgraph_utils.py` 置
    `_EXTRA_CTX.capturing`。本版 `_EXTRA_CTX` 是读写 forward_context 的代理，
    两者落在同一个属性上，读它即可覆盖两条路。
    """
    try:
        from vllm.forward_context import get_forward_context

        return bool(getattr(get_forward_context(), "capturing", False))
    except Exception:
        return False


def _debug_allowed(what: str) -> bool:
    """落盘与逐参对拍这类调试设施只能在 eager 下用，捕获期一律拒绝。

    它们都要把张量读回 CPU。捕获期读回要么报错、要么把当时的值固定进图里，
    之后每步重放都用这个过期值 —— 不报错，结果悄悄错，比直接崩更难查。
    graph 模式的全部意义就是进图，所以整个模式下都不开。
    """
    global _DEBUG_REFUSED
    if os.environ.get("PTO_CSA_MODE", "").strip().lower() == "graph" or _capture_active():
        if not _DEBUG_REFUSED:
            _DEBUG_REFUSED = True
            print(
                f"[pto-csa] 已禁用调试设施({what})：它要把张量读回 CPU，"
                "aclgraph 捕获期不允许；要用请加 --enforce-eager。",
                flush=True,
            )
        return False
    return True


def enabled() -> bool:
    """`PTO_CSA=1` 打开层级替换。"""
    return os.environ.get("PTO_CSA", "").strip() not in ("", "0", "false", "False")


def get_runner():
    """进程内单例：编译一次，之后每步只 dispatch。"""
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = PtoCsaRunner()
    return _RUNNER


@contextmanager
def graph_replay_swimlane(graph_name: str) -> Iterator[None]:
    """Collect a real replay without creating a Worker or re-running the graph."""
    runner = _RUNNER
    if runner is None or not runner._kernel_ready or not runner._swimlane_level:
        yield
        return
    import torch

    if torch.npu.is_current_stream_capturing():
        # A nested wrapper can replay an existing subgraph into an outer capture.
        yield
        return
    with runner._swimlane_lock:
        if runner._swimlane_captures >= runner._swimlane_limit:
            yield
            return
        with runner._collect_swimlane({"mode": "graph_replay", "graph": graph_name}):
            yield


def stash_decode_inputs(impl, **kw) -> None:
    """`_forward_decode` 把 PTO 需要、但 `forward` 里看不到的张量存下来。

    indexer 的 topk 和两份 KV cache 都只在 `_forward_decode` 的作用域里，
    而替换要覆盖到 `forward` 末尾的 o_proj，所以必须跨这两层传递。
    """
    impl._pto_csa_stash = kw


class PtoCsaRunner:
    """pypto-lib `sparse_attn_test` 的 vLLM 适配器。

    两边的数据约定有三处对不上，适配器负责翻译，翻译不了就返回 None 让调用方退回 vendor：

    1. **压缩缓存的页大小**。PTO 的 cmp 页是 `BLOCK_SIZE // COMPRESS_RATIO`（32 槽），
       vLLM 的压缩缓存和原始缓存一样是 128 槽。两者都是 `block * page + intra` 的线性
       编址，所以把 [nblk, 128, 1, D] 重排成 [nblk*4, 32, 1, D] 后物理地址不变，
       块表按 `pblk*4 + j % 4` 重映射即可，没有数据搬运。

    2. **权重 layout 与量化形态**。`wo_a` 在 vLLM 是 [G, O_GROUP_IN, O_LORA]，PTO 要
       [G, O_LORA, O_GROUP_IN]；`wo_b` 在这份官方 checkpoint 里是 **BF16**，而 PTO 的
       kernel 签名写死 INT8 + per-channel scale（flash_mtp 下三个 CSA 变体都是如此，
       没有 BF16 版本）。所以 wo_b 在 setup 时按 pypto-lib 自己的量化器转一次并缓存
       —— 这是**真实的数值差异**，不是适配 bug。

    3. **KV cache 的归属**。PyPTO 的 dispatch 只接受 host 张量（自己做 H2D/D2H）或它自己
       Worker 分配的 `DeviceTensor`；vLLM 的 KV cache 是 torch_npu 分配的，两者都不是。
       整份 cache 十几 GiB，不可能每步上传，所以这里只把**这一步真正会被读到的那几页**
       收拢成一份小 cache 再重写槽号/块表：窗口 128 槽最多跨 2 页，压缩侧的页数由序列
       长度而非 topk 决定，量级是 MiB。代价是每步一轮 D2H+H2D，够验正确性，不谈性能。
    """

    MAX_CMP_PAGES = 4096

    def __init__(self) -> None:
        self._csa = None
        self._compiled = None
        self._cfg = None
        self._weights: dict[tuple, tuple] = {}
        # program: 只吃 host 张量，PyPTO 自己做 H2D/D2H，KV cache 要先把用到的页收拢。
        # kernel : 直接拿 vLLM 的 NPU 张量调 @pl.jit 对象，省掉每步搬运，且**不需要收拢**
        #          —— kernel 模式借用调用方的张量，整份 cache 原样交过去、用原始物理索引。
        self._mode = os.environ.get("PTO_CSA_MODE", "program").strip().lower()
        self._kernel_ready = False
        # 替换有没有在**捕获期**执行，决定它会不会进 aclgraph。靠日志条数推断不可靠，
        # 直接问 torch：捕获期 is_current_stream_capturing() 为真。
        self.stats = {"dispatches": 0, "fallbacks": {}, "capturing": {"yes": 0, "no": 0}}
        # kernel 与 graph 共用同一套无同步推导（整份 cache 原样交、块表定宽）；
        # 二者只差最后怎么下发：kernel 直接调 @pl.jit 对象，graph 写常驻缓冲 + 调注册算子。
        self._nosync = self._mode in ("kernel", "graph")
        self._op = None  # graph 模式：注册后的 torch 算子
        self._bufs = None  # graph 模式：常驻入参缓冲区
        self._swimlane_level = envs.VLLM_ASCEND_PTO_CSA_SWIMLANE_LEVEL
        self._swimlane_root = envs.VLLM_ASCEND_PTO_CSA_SWIMLANE_DIR.strip()
        self._swimlane_limit = envs.VLLM_ASCEND_PTO_CSA_SWIMLANE_MAX_CAPTURES
        if not 0 <= self._swimlane_level <= 4:
            raise ValueError("VLLM_ASCEND_PTO_CSA_SWIMLANE_LEVEL must be in 0..4")
        if self._swimlane_level:
            if not self._nosync:
                raise ValueError("CSA swimlane requires PTO_CSA_MODE=kernel or graph")
            if not self._swimlane_root or self._swimlane_limit < 1:
                raise ValueError("CSA swimlane requires a nonempty SWIMLANE_DIR and positive SWIMLANE_MAX_CAPTURES")
        self._swimlane_dir: Path | None = None
        self._swimlane_warmed: set[tuple] = set()
        self._swimlane_names: dict[str, str] = {}
        self._swimlane_captures = 0
        self._swimlane_lock = threading.Lock()
        self._swimlane_replay_only = False

    def _lazy_import(self):
        if self._csa is None:
            from vllm_ascend.attention.pto_kernels.mtp import decode_sparse_attn_csa as csa

            self._csa = csa
        return self._csa

    def ensure_compiled(self, device_id: int):
        """第一次调用时编译；`@pl.jit` 的特化键由 dummy 张量的形状给出。"""
        if self._compiled is not None:
            return self._compiled
        import torch
        from pypto.runtime import RunConfig

        csa = self._lazy_import()
        self._cfg = RunConfig(
            platform=os.environ.get("PTO_CSA_PLATFORM", "a2a3"),
            device_id=device_id,
            enable_chip_swimlane=0,
            enable_dep_gen=False,
            enable_pmu=0,
            dump_passes=False,
        )
        dummy = [torch.empty(sp.shape, dtype=sp.dtype) for sp in csa.build_tensor_specs()]
        self._compiled = csa.sparse_attn_test.compile(*dummy, config=self._cfg)
        print(
            f"[pto-csa] compiled T={csa.T} B={csa.B} S={csa.S} H={csa.H} HEAD_DIM={csa.HEAD_DIM} device_id={device_id}",
            flush=True,
        )
        return self._compiled

    def ensure_kernel_init(self) -> None:
        """kernel 模式每进程一次，且必须在首次 kernel 调用前、graph capture 之外。"""
        if self._kernel_ready:
            return
        from pypto.torch import init

        if self._swimlane_level:
            import torch

            root = Path(self._swimlane_root).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            self._swimlane_dir = Path(
                tempfile.mkdtemp(prefix=f"worker_{os.getpid()}_device{torch.npu.current_device()}_", dir=root)
            )
            init(enable_chip_swimlane=self._swimlane_level, enable_dep_gen=True, output_dir=self._swimlane_dir)
            print(f"[pto-csa] swimlane enabled -> {self._swimlane_dir}", flush=True)
        else:
            init()
        self._kernel_ready = True
        print("[pto-csa] pypto.torch.init() 完成（kernel 模式）", flush=True)

    def _run_kernel(self, op, args: tuple, layer_name: str) -> None:
        """Submit once, optionally recording a warmed live decode call."""
        if not self._swimlane_level:
            self.ensure_kernel_init()
            op(*args)
            return

        import torch

        # PyPTO must be initialized during warmup, never while recording a graph.
        if torch.npu.is_current_stream_capturing():
            if not self._kernel_ready:
                raise RuntimeError("CSA graph capture requires kernel initialization and warmup before capture")
            op(*args)
            return
        with self._swimlane_lock:
            self.ensure_kernel_init()
            if _capture_active() or self._swimlane_captures >= self._swimlane_limit:
                op(*args)
                return
            signature = (layer_name, tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in args))
            if signature not in self._swimlane_warmed:
                # The real first call compiles/warms this shape. Never replay a
                # live decode just for profiling: it may mutate caller storage.
                op(*args)
                self._remember_swimlane_names(args)
                self._swimlane_warmed.add(signature)
                return
            if self._swimlane_replay_only:
                op(*args)
                return
            metadata = {
                "mode": "eager",
                "layer": layer_name,
                "device": str(args[0].device),
                "dispatch": self.stats["dispatches"] + 1,
                "args": [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in args],
            }
            with self._collect_swimlane(metadata):
                op(*args)

    def _remember_swimlane_names(self, args: tuple) -> None:
        """Read names from the exact warmed specialization, outside capture."""
        from pypto.runtime import CompileOptions
        from pypto.runtime.kernel.context import get_process_kernel_state

        # The integration branch exposes kernel artifacts separately from
        # program-mode compile(). Resolving this warmed artifact never launches
        # an operator or creates another Worker.
        platform = get_process_kernel_state().require_config().platform
        artifact = self._lazy_import().sparse_attn_test._resolve_kernel_artifact(
            args, {"config": CompileOptions(platform=platform)}
        )
        manifest = json.loads((Path(artifact.output_dir) / "binary_manifest.json").read_text())
        names = {str(kernel["func_id"]): kernel["name"] for kernel in manifest["kernels"]}
        for func_id, name in names.items():
            if func_id in self._swimlane_names and self._swimlane_names[func_id] != name:
                raise RuntimeError(f"CSA specializations have conflicting swimlane names for function {func_id}")
        self._swimlane_names.update(names)

    @contextmanager
    def _collect_swimlane(self, metadata: dict) -> Iterator[None]:
        from pypto.torch import begin_dfx, end_dfx

        begin_dfx()
        try:
            yield
        finally:
            end_dfx()
            # The runtime advances its output window even if the launch fails.
            window = self._swimlane_captures
            self._swimlane_captures += 1
        assert self._swimlane_dir is not None
        output = self._swimlane_dir if window == 0 else self._swimlane_dir / f"window_{window}"
        self._export_swimlane(output, metadata)

    def _export_swimlane(self, output: Path, metadata: dict) -> None:
        """Export a live CSA call or graph replay with its execution metadata."""
        records, deps = output / "chip_swimlane_records.json", output / "deps.json"
        if not records.is_file() or not records.stat().st_size:
            raise RuntimeError(f"CSA swimlane did not produce a nonempty artifact: {records}")
        metadata = {
            **metadata,
            "pid": os.getpid(),
            "level": self._swimlane_level,
            "includes_dependency_collection_overhead": True,
        }
        # A model graph may bypass the CSA replacement entirely. Retain the
        # diagnostic evidence, but never advertise an empty graph as a CSA trace.
        if metadata["mode"] == "graph_replay" and not json.loads(records.read_text())["aicore_tasks"]:
            metadata["status"] = "no_csa_tasks"
            (output / "capture.json").write_text(json.dumps(metadata, indent=2))
            print(f"[pto-csa] replay contains no recorded CSA tasks: {metadata['graph']} -> {output}", flush=True)
            return
        if not deps.is_file() or not deps.stat().st_size:
            raise RuntimeError(f"CSA swimlane did not produce a nonempty artifact: {deps}")
        (output / "capture.json").write_text(json.dumps(metadata, indent=2))
        name_map = output / "name_map.json"
        name_map.write_text(json.dumps({"level": 2, "callable_id_to_name": self._swimlane_names}, indent=2))
        merged = output / "merged_swimlane.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "simpler_setup.tools.swimlane_converter",
                str(records),
                "--deps-json",
                str(deps),
                "--func-names",
                str(name_map),
                "-o",
                str(merged),
            ],
            check=True,
            timeout=60,
        )
        trace = json.loads(merged.read_text()) if merged.is_file() else {}
        events = trace.get("traceEvents", [])
        if not any(event.get("ph") == "X" and "taskId" in event.get("args", {}) for event in events):
            raise RuntimeError(f"CSA swimlane conversion did not produce a nonempty trace: {merged}")
        if any(
            event.get("ph") == "X"
            and "taskId" in event.get("args", {})
            and event.get("name", "").startswith(("func_", "task("))
            for event in events
        ):
            raise RuntimeError(f"CSA swimlane is missing kernel function names: {merged}")
        self.stats.setdefault("swimlane", []).append(str(merged))
        print(f"[pto-csa] {metadata['mode']} swimlane -> {merged}", flush=True)

    def ensure_registered(self):
        """把 kernel 注册成正规的 torch 算子，返回 `torch.ops.pypto_csa.sparse_attn`。

        注册本身不编译、不建 Worker、也不需要 `pypto.torch.init`；它只特化前端 IR 去
        推导哪些参数是输出或读写参数。真正下发时和直接调 @pl.jit 对象共用同一份产物与
        进程 Worker，所以仍然要先 init。

        注册不是捕获的必要条件（直接调 @pl.jit 对象一样能被捕获），但注册之后这个
        kernel 在 torch 眼里是一个正规算子节点，才谈得上把它放进 vLLM 构的图里。
        """
        if self._op is not None:
            return self._op
        from pypto.torch import register

        csa = self._lazy_import()
        self._op = register(csa.sparse_attn_test, "pypto_csa::sparse_attn")
        print(f"[pto-csa] 已注册 torch 算子: {self._op}", flush=True)
        return self._op

    def ensure_buffers(self, dev, n_cols: int):
        """一次性分配全部每步会变的入参，之后只原地写。

        重放用的是捕获时记下的**地址**：每步新建张量地址就变了，图里记的还是老地址，
        重放读到的是旧数据。所以这些缓冲区必须常驻、原地更新。
        不变的几项（两份 KV cache、wo_a/wo_b/wo_b_scale）本来地址就固定，不用建缓冲。
        """
        import torch

        if self._bufs is not None and self._bufs["n_cols"] == n_cols:
            return self._bufs
        csa = self._lazy_import()
        T, B, H, HD = csa.T, csa.B, csa.H, csa.HEAD_DIM
        z = lambda shape, dt: torch.zeros(shape, dtype=dt, device=dev)
        self._bufs = {
            "n_cols": n_cols,
            "q": z((T, H, HD), torch.bfloat16),
            "win": z((T, csa.WIN), torch.int32),
            "cmp_bt": z((B, n_cols), torch.int32),
            "idx": z((T, csa.IDX_TOPK), torch.int32),
            "pos": z((T, 1), torch.int32),
            "sink": z((H,), torch.float32),
            "cos": z((T, csa.ROPE_DIM), torch.bfloat16),
            "sin": z((T, csa.ROPE_DIM), torch.bfloat16),
            "out": z((T, csa.D), torch.bfloat16),
        }
        print(f"[pto-csa] 常驻缓冲区已分配 T={T} B={B} n_cols={n_cols}", flush=True)
        return self._bufs

    def weights_for(self, impl, dev=None):
        """把这一层的 wo_a / wo_b 翻译成 PTO 要的 layout 与量化形态，按层缓存。"""
        import torch

        key = (id(impl), self._mode)
        if key in self._weights:
            return self._weights[key]
        csa = self._lazy_import()

        wo_a = impl.wo_a.weight.detach().transpose(1, 2).contiguous().to(torch.bfloat16).cpu()

        wb = impl.wo_b.weight.detach()
        if wb.dtype == torch.int8:
            scale = getattr(impl.wo_b, "weight_scale", None)
            if scale is None:
                raise RuntimeError("wo_b 是 INT8 但没有 weight_scale，无法喂给 PTO kernel")
            wo_b, wo_b_scale = wb.cpu(), scale.detach().float().reshape(-1).cpu()
        else:
            amax = wb.float().abs().amax(dim=-1).clamp_min(csa.INT8_AMAX_EPS)
            scale_q = csa.INT8_SCALE_MAX / amax
            w_i8 = torch.round(wb.float() * scale_q.unsqueeze(-1))
            wo_b = w_i8.to(torch.int32).to(torch.float16).to(torch.int8).cpu()
            wo_b_scale = (1.0 / scale_q).float().cpu()
            print(f"[pto-csa] wo_b 由 {wb.dtype} 量化成 INT8 per-channel (kernel 签名不接受 BF16)", flush=True)

        out = (wo_a, wo_b, wo_b_scale)
        if self._nosync and dev is not None:
            out = tuple(t.to(dev) for t in out)
        self._weights[key] = out
        return out

    def _fallback(self, why: str):
        # 每种原因只打第一次：回退是每步都会发生的，全打会把服务日志淹掉；
        # 但一次都不打就只能看到"输出没变"，定位不到是哪一条前置条件没过。
        n = self.stats["fallbacks"].get(why, 0) + 1
        self.stats["fallbacks"][why] = n
        total = sum(self.stats["fallbacks"].values())
        # 只打第一次会让人把"1 个不同原因"误读成"回退 1 次"，所以再定期报一次总量。
        if total % 100 == 0:
            print(
                f"[pto-csa] 回退累计={total} dispatch累计={self.stats['dispatches']} 明细={self.stats['fallbacks']}",
                flush=True,
            )
        if n == 1:
            print(f"[pto-csa] 回退到 vendor: {why}", flush=True)
        return None

    def run_decode(self, impl, *, cos, sin, layer_name: str = "unknown"):
        """跑一次 PTO CSA，返回 [n_tokens, D] 的 o_proj 输出；不适用时返回 None。"""
        import torch

        if self._swimlane_level:
            # Graph-enabled warmup must not consume the replay capture budget.
            self._swimlane_replay_only = not impl.vllm_config.model_config.enforce_eager
        st = getattr(impl, "_pto_csa_stash", None)
        if not st:
            return self._fallback("没有 decode 暂存（这一步没走 ratio-4 decode）")

        csa = self._lazy_import()
        T, B, S = csa.T, csa.B, csa.S
        BS, WIN = csa.BLOCK_SIZE, csa.WIN
        CMP_PAGE = csa.CMP_STORAGE_BLOCK_SIZE
        PER_ORI = BS // CMP_PAGE

        q, ori_kv, cmp_kv = st["q"], st["ori_kv"], st["cmp_kv"]
        n = int(q.shape[0])
        if n == 0 or n > B:
            return self._fallback(f"decode_tokens={n} 不在 1..{B}")
        if int(q.shape[1]) != csa.H or int(q.shape[2]) != csa.HEAD_DIM:
            return self._fallback(f"q {tuple(q.shape)} 与 kernel 的 [{csa.H},{csa.HEAD_DIM}] 不符")
        if int(cmp_kv.shape[1]) % CMP_PAGE:
            return self._fallback(f"压缩页 {int(cmp_kv.shape[1])} 不是 {CMP_PAGE} 的整数倍")

        dev = q.device
        positions = st.get("positions")
        if positions is None:
            positions = st["seqused_kv"].to(torch.int64) - 1
        pos = positions.to(dev).to(torch.int64).reshape(-1)[:n]

        # vLLM 一步给 n(<=B) 个 decode token、每序列 1 个；kernel 的 (B,S) 是编译期常量。
        # 第 i 个 token 进 batch i 的两个 S 槽（两槽互相独立，取回只用 s=0），
        # 不足 B 的批次用 token 0 填满，填充行的结果丢弃。
        src = torch.arange(T, device=dev) // S
        src = torch.where(src < n, src, torch.zeros_like(src))
        batch_src = src[::S]  # [B] -> vLLM token

        q_t = q.index_select(0, src).contiguous()
        pos_t = pos.index_select(0, src)

        # --- 窗口：算物理槽，再把用到的页收拢成一份小 cache ---
        obt = st["ori_block_table"].to(dev).to(torch.int64)
        if obt.shape[0] < n:
            return self._fallback("ori_block_table 行数少于 decode token 数")
        obt_t = obt.index_select(0, src)
        offs = torch.arange(WIN, device=dev) - (WIN - 1)
        abs_pos = pos_t.unsqueeze(1) + offs.unsqueeze(0)  # [T, WIN]
        lblk = torch.div(abs_pos, BS, rounding_mode="floor")
        intra = abs_pos - lblk * BS
        ok = (abs_pos >= 0) & (lblk >= 0) & (lblk < obt_t.shape[1])
        pblk = torch.gather(obt_t, 1, lblk.clamp(0, obt_t.shape[1] - 1))
        ok = ok & (pblk >= 0)

        # 下面的校验与收拢都要把 NPU 上的值读回 CPU（`bool(...)`、`int(...)`、布尔掩码
        # 索引、torch.unique 的输出长度依赖数据）。捕获期不能读回：那时 CPU 不等 NPU 算完，
        # 读到的是未写入或上一轮的残留，而且读回的值会被当成常量固定进图里，之后每步重放
        # 都用这个过期值 —— 不报错，结果悄悄错。所以 kernel 路径上一处都不能留。
        if self._nosync:
            # 整份 cache 原样交过去，槽号用原始物理值，不收拢、不搬运。越界由
            # lblk < obt_t.shape[1] 与 pblk >= 0 两个逐元素条件挡住，二者都已折进 ok，
            # 无效位置写 -1，kernel 按 -1 跳过。
            ori_small = ori_kv
            win_new = torch.where(ok, pblk.clamp_min(0) * BS + intra, torch.full_like(abs_pos, -1)).to(torch.int32)
        else:
            if not bool(ok.any()):
                return self._fallback("窗口没有任何有效页")
            if int(pblk[ok].max()) >= ori_kv.shape[0]:
                return self._fallback("窗口块表指到了 KV cache 之外")
            pages = torch.unique(pblk[ok])
            ori_small = ori_kv.index_select(0, pages).contiguous()
            premap = torch.zeros(int(pages.max()) + 1, dtype=torch.int64, device=dev)
            premap[pages] = torch.arange(pages.numel(), device=dev)
            win_new = torch.where(ok, premap[pblk.clamp_min(0)] * BS + intra, torch.full_like(abs_pos, -1)).to(
                torch.int32
            )

        # --- 压缩侧：块表从 vLLM 的 128 槽页换算到 PTO 的 32 槽页，再收拢 ---
        idx = st["cmp_sparse_indices"].to(dev).to(torch.int64)
        idx = idx.reshape(-1, idx.shape[-1])
        if idx.shape[0] < n:
            return self._fallback("cmp_sparse_indices 行数少于 decode token 数")
        idx_t = idx.index_select(0, src)[:, : csa.IDX_TOPK]
        if idx_t.shape[1] < csa.IDX_TOPK:
            idx_t = torch.nn.functional.pad(idx_t, (0, csa.IDX_TOPK - idx_t.shape[1]), value=-1)

        # vLLM 的压缩缓存一页装 BLOCK_SIZE 个**压缩槽**（4098 token = 1024 槽，实测正好
        # 写满 8 页），PTO 的一页装 CMP_PAGE=32 个。两边都是 `page * rows + intra` 的线性
        # 编址，所以把 [nblk, 128, 1, D] 重排成 [nblk*4, 32, 1, D] 后物理地址不变，
        # 块表按 `vcbt[j // 4] * 4 + j % 4` 重映射即可，没有数据搬运。
        #
        # 实测块表有效列数不能当依据：分配器给压缩缓存的块数和滑窗缓存一样多（各 33 列），
        # 但其中只有前 8 页真被写过。判断页容量要看写没写，不是看分配了几块。
        vcbt = st["cmp_block_table"].to(dev).to(torch.int64)
        vcbt_t = vcbt.index_select(0, batch_src.clamp(0, vcbt.shape[0] - 1))  # [B, cols]

        if self._nosync:
            # 重排必须是**视图**：整份压缩缓存十几 GiB，触发拷贝就废了。用 view 而不是
            # reshape —— 不连续时直接报错，而不是悄悄搬一份。
            try:
                cmp_view = cmp_kv.view(-1, CMP_PAGE, *cmp_kv.shape[2:])
            except RuntimeError:
                return self._fallback("压缩缓存不连续，重排成 32 槽页会触发整份拷贝")
            # 按固定宽度直接建块表，绕开 `max_slot = int(idx_t.max().item())` 那次读回。
            # 宽度是 host 常量，所以 @pl.jit 的特化键不再随 decode 推进而变（实测宽度每步
            # 变化会导致每次调用重新特化）；越界也不靠读回判断，而是在设备上 clamp / 置 -1。
            n_cols = int(os.environ.get("PTO_CSA_CMP_COLS", "128"))
            j = torch.arange(n_cols, device=dev)
            v_col = torch.div(j, PER_ORI, rounding_mode="floor").clamp(max=vcbt_t.shape[1] - 1)
            cmp_phys = torch.gather(vcbt_t, 1, v_col.unsqueeze(0).expand(B, -1)) * PER_ORI + (j % PER_ORI).unsqueeze(
                0
            )  # [B, n_cols]
            cmp_small = cmp_view
            cmp_bt = cmp_phys.clamp(0, cmp_view.shape[0] - 1).to(torch.int32)
            # 超出这张固定宽度块表能表达范围的槽号直接作废：kernel 对 -1 的处理是跳过，
            # 与 golden 一致。这样不必把最大槽号读回来校验。
            idx_t = torch.where(idx_t < n_cols * CMP_PAGE, idx_t, torch.full_like(idx_t, -1))
        else:
            max_slot = int(idx_t.max().item())
            if max_slot < 0:
                return self._fallback("这一步没有任何有效压缩槽")
            n_logical = max_slot // CMP_PAGE + 1
            if n_logical > self.MAX_CMP_PAGES:
                return self._fallback(f"压缩逻辑块 {n_logical} 超过上限 {self.MAX_CMP_PAGES}")
            j = torch.arange(n_logical, device=dev)
            v_col = torch.div(j, PER_ORI, rounding_mode="floor")
            if int(v_col.max()) >= vcbt_t.shape[1]:
                return self._fallback("压缩块表列数不够覆盖这一步用到的槽")
            cmp_phys = torch.gather(vcbt_t, 1, v_col.unsqueeze(0).expand(B, -1)) * PER_ORI + (j % PER_ORI).unsqueeze(
                0
            )  # [B, n_logical]
            cmp_view = cmp_kv.reshape(-1, CMP_PAGE, *cmp_kv.shape[2:])
            used = torch.unique(cmp_phys[cmp_phys >= 0])
            if used.numel() == 0 or int(used.max()) >= cmp_view.shape[0]:
                return self._fallback("压缩块表指到了缓存之外")
            cmp_small = cmp_view.index_select(0, used).contiguous()
            cremap = torch.zeros(int(used.max()) + 1, dtype=torch.int64, device=dev)
            cremap[used] = torch.arange(used.numel(), device=dev)
            cmp_bt = cremap[cmp_phys.clamp_min(0)].to(torch.int32)

        # --- 其余入参 ---
        sink = impl.attn_sink.detach().float().reshape(-1)
        cos_t = _rope_table(cos, src, csa)
        sin_t = _rope_table(sin, src, csa)
        wo_a, wo_b, wo_b_scale = self.weights_for(impl, dev)

        if _VERIFY and _debug_allowed("verify") and "inputs" not in self.stats:
            self.stats["inputs"] = verify_inputs(
                impl,
                csa,
                st,
                dict(
                    n=n,
                    src=src,
                    batch_src=batch_src,
                    q_t=q_t,
                    win_new=win_new,
                    ori_small=ori_small,
                    cmp_bt=cmp_bt,
                    cmp_small=cmp_small,
                    idx_t=idx_t,
                    pos_t=pos_t,
                    cos=cos,
                    sin=sin,
                    cos_t=cos_t,
                    sin_t=sin_t,
                    wo_a=wo_a,
                    wo_b=wo_b,
                    wo_b_scale=wo_b_scale,
                    sink=sink,
                ),
            )
            print("[pto-csa] verify " + json.dumps(self.stats["inputs"], ensure_ascii=False), flush=True)

        if self._mode == "graph":
            # 可捕获形态：入参全部是常驻缓冲区，每步只原地写内容，不新建张量。
            # 重放用的是捕获时记下的地址，新建张量地址会变，图里记的还是老地址。
            # 下发走注册后的 torch 算子，它在 torch 眼里是一个正规算子节点。
            op = self.ensure_registered()
            b = self.ensure_buffers(dev, int(cmp_bt.shape[1]))
            b["q"].copy_(q_t.to(torch.bfloat16))
            b["win"].copy_(win_new)
            b["cmp_bt"].copy_(cmp_bt)
            b["idx"].copy_(idx_t.to(torch.int32))
            b["pos"].copy_(pos_t.to(torch.int32).reshape(T, 1))
            b["sink"].copy_(sink)
            b["cos"].copy_(cos_t.to(torch.bfloat16))
            b["sin"].copy_(sin_t.to(torch.bfloat16))
            self._run_kernel(
                op,
                (
                    b["q"],
                    ori_small.to(torch.bfloat16),
                    b["win"],
                    cmp_small.to(torch.bfloat16),
                    b["cmp_bt"],
                    b["idx"],
                    b["pos"],
                    b["sink"],
                    b["cos"],
                    b["sin"],
                    wo_a,
                    wo_b,
                    wo_b_scale,
                    b["out"],
                ),
                layer_name,
            )
            attn_out = b["out"]
            take = torch.arange(n, dtype=torch.int64, device=dev) * S
            out = attn_out.index_select(0, take)
        elif self._mode == "kernel":
            # 全程 NPU 张量，直接调 @pl.jit 对象（不是 op.compile(...)(...)）。张量由调用方
            # 持有、kernel 借用，所以没有 H2D/D2H，也挂在当前 torch NPU 流上。
            attn_out = torch.zeros((T, csa.D), dtype=torch.bfloat16, device=dev)
            self._run_kernel(
                csa.sparse_attn_test,
                (
                    q_t.to(torch.bfloat16),
                    ori_small.to(torch.bfloat16),
                    win_new,
                    cmp_small.to(torch.bfloat16),
                    cmp_bt,
                    idx_t.to(torch.int32),
                    pos_t.to(torch.int32).reshape(T, 1),
                    sink,
                    cos_t.to(torch.bfloat16),
                    sin_t.to(torch.bfloat16),
                    wo_a,
                    wo_b,
                    wo_b_scale,
                    attn_out,
                ),
                layer_name,
            )
            take = torch.arange(n, dtype=torch.int64, device=dev) * S
            out = attn_out.index_select(0, take)
        else:
            compiled = self.ensure_compiled(_device_index(dev))
            attn_out = torch.zeros((T, csa.D), dtype=torch.bfloat16)
            compiled(
                q_t.to(torch.bfloat16).cpu(),
                ori_small.to(torch.bfloat16).cpu(),
                win_new.cpu(),
                cmp_small.to(torch.bfloat16).cpu(),
                cmp_bt.cpu(),
                idx_t.to(torch.int32).cpu(),
                pos_t.to(torch.int32).reshape(T, 1).cpu(),
                sink.cpu(),
                cos_t.to(torch.bfloat16).cpu(),
                sin_t.to(torch.bfloat16).cpu(),
                wo_a,
                wo_b,
                wo_b_scale,
                attn_out,
                config=self._cfg,
            )
            take = torch.arange(n, dtype=torch.int64) * S
            out = attn_out.index_select(0, take).to(dev)
        self.stats["dispatches"] += 1
        try:
            cap = bool(torch.npu.is_current_stream_capturing())
        except Exception:
            cap = None
        if cap is not None:
            self.stats["capturing"]["yes" if cap else "no"] += 1
            if self.stats["capturing"]["yes"] + self.stats["capturing"]["no"] <= 8:
                print(f"[pto-csa] dispatch #{self.stats['dispatches']} 捕获期={cap}", flush=True)
        _dump_once(
            st,
            dict(
                q_t=q_t,
                win_new=win_new,
                ori_small=ori_small,
                cmp_small=cmp_small,
                cmp_bt=cmp_bt,
                idx_t=idx_t,
                pos_t=pos_t,
                sink=sink,
                cos_t=cos_t,
                sin_t=sin_t,
                wo_a=wo_a,
                wo_b=wo_b,
                wo_b_scale=wo_b_scale,
                attn_out_full=attn_out,
                pto_out=out,
                src=src,
                n=n,
            ),
        )
        return out


def _device_index(dev) -> int:
    """torch 设备号；PyPTO 的 device_id 与 ASCEND_RT_VISIBLE_DEVICES 之后的序号一致。"""
    return int(dev.index) if getattr(dev, "index", None) is not None else 0


_REPORT: dict[str, dict] = {}


def report(layer_name: str, vendor_out, pto_out) -> None:
    """记录 vendor 与 PTO 两条路径在同一步、同一批 token 上的差距。

    每层只累计统计量，不落盘每步的张量：decode 会跑上百步，逐步落盘既慢又没用。
    `PTO_CSA_REPORT=<file>` 指定最终写到哪里，不设就只在进程退出时打印。
    """
    # 逐步对拍要把两边的 max/mean 读回 CPU，捕获期不允许：实测会以
    #   "Not allow to synchronize captured-stream" / LocalScalarDenseNpu.cpp:23 / 107027
    # 直接把引擎初始化打挂。它是调试设施，捕获期跳过即可，eager 轮照常统计。
    if _capture_active():
        return

    d = (vendor_out.float() - pto_out.float()).abs()
    cur = _REPORT.setdefault(layer_name, {"steps": 0, "max_abs": 0.0, "sum_mean": 0.0})
    cur["steps"] += 1
    cur["max_abs"] = max(cur["max_abs"], d.max().item())
    cur["sum_mean"] += d.mean().item()
    cur["mean_abs"] = cur["sum_mean"] / cur["steps"]
    cur["vendor_abs_mean"] = vendor_out.float().abs().mean().item()
    if cur["steps"] in (1, 2, 5, 10, 50, 100):
        print(
            f"[pto-csa] {layer_name} step={cur['steps']} "
            f"max_abs={cur['max_abs']:.6g} mean_abs={cur['mean_abs']:.6g} "
            f"(vendor |mean|={cur['vendor_abs_mean']:.6g})",
            flush=True,
        )


def dump_report() -> None:
    """把累计的 vendor-vs-PTO 对比与 fallback 统计写盘。"""
    path = os.environ.get("PTO_CSA_REPORT", "").strip()
    payload = {"per_layer": _REPORT, "forward_shapes": dict(_FORWARD_SEEN)}
    if _RUNNER is not None:
        payload["runner"] = _RUNNER.stats
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print("[pto-csa] report " + text, flush=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)


def _dump_once(stash, derived) -> None:
    """`PTO_CSA_DUMP=<dir>` 时，把第一次替换用到的全部张量落盘一次。

    live 路径和离线 fixture 的差别只能靠真实入参定位：翻译层哪一步走偏了，
    在服务日志里只表现为"输出不对"。落一份盘就能离线重放、逐段对拍。
    """
    if not _debug_allowed("dump"):
        return
    global _DUMPED
    d = os.environ.get("PTO_CSA_DUMP", "").strip()
    if not d or _DUMPED:
        return
    import torch

    _DUMPED = True
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    # 整份 KV cache 有十几 GiB，落盘没有意义 —— 翻译层真正用到的是收拢后的小 cache，
    # 那份在 derived 里。这里只留小张量。
    payload = {
        f"stash.{k}": (v.detach().cpu() if hasattr(v, "detach") else v)
        for k, v in stash.items()
        if v is not None and (not hasattr(v, "numel") or v.numel() <= 1 << 22)
    }
    payload.update({f"derived.{k}": (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in derived.items()})
    torch.save(payload, p / "step0.pt")
    if _PENDING_VENDOR_ATTN is not None:
        torch.save(_PENDING_VENDOR_ATTN, p / "vendor_attn0.pt")
    print(
        f"[pto-csa] dumped step0 -> {p / 'step0.pt'} (vendor stage-1: {_PENDING_VENDOR_ATTN is not None})", flush=True
    )


def dump_vendor(vendor_out) -> None:
    """把同一步 vendor 的输出补进 dump，作为对拍基准。"""
    if not _debug_allowed("dump_vendor"):
        return
    d = os.environ.get("PTO_CSA_DUMP", "").strip()
    if not d:
        return
    import torch

    f = Path(d) / "vendor0.pt"
    if f.exists():
        return
    torch.save(vendor_out.detach().cpu(), f)
    print(f"[pto-csa] dumped vendor0 -> {f}", flush=True)


_PENDING_VENDOR_ATTN = None


def dump_vendor_attn(attn_output) -> None:
    """记下 vendor 的 stage-1 注意力（逆 RoPE 之前），由 `_dump_once` 与 step0 一起落盘。

    直接在这里写盘会取到**另一步**的结果：这个钩子每一步都过，而 step0 只在第一次真正
    发生替换时落。两者不同步就没法对拍（表现为 batch 维对不上）。
    """
    if not _debug_allowed("dump_vendor_attn"):
        return
    global _PENDING_VENDOR_ATTN
    if not os.environ.get("PTO_CSA_DUMP", "").strip() or _DUMPED:
        return
    _PENDING_VENDOR_ATTN = attn_output.detach().cpu()


def _rope_table(t, src, csa):
    """把 vLLM 的 RoPE 表翻成 pypto-lib 的排布。

    vLLM 走 `rotary_mode="interleave"`，表是 `[c0, c0, c1, c1, ...]` —— 64 个位置上
    只有 32 个不同的频率，成对重复。pypto-lib 的表是"前半 32 个为真值、后半复制"
    （见 `precompute_freqs_cos_sin` 的 docstring），golden 只读 `[:, :HALF_ROPE]`。
    直接截前 32 个会只拿到前 16 个频率各两份，所以这里按步长 2 去重。
    """
    import torch

    flat = t.reshape(t.shape[0], -1)
    half = csa.ROPE_DIM // 2
    uniq = flat[:, 0::2][:, :half]
    if uniq.shape[1] < half:
        uniq = torch.nn.functional.pad(uniq, (0, half - uniq.shape[1]))
    return torch.cat([uniq, uniq], dim=1).index_select(0, src).to(torch.bfloat16)


def verify_inputs(impl, csa, st, d) -> dict:
    """逐个入参核对 PTO 侧与 vLLM 侧是否指的是同一个东西。

    只比形状/dtype 没有意义 —— 真正会错的是**语义**：同一个压缩槽号，两边解析到的是不是
    同一行 KV；seqlen 在 PTO 侧根本不是入参，而是被烘进 `window_swa_indices` 和
    `position_ids`，那条推导对不对。所以这里做的是**按内容对拍**：
    用 vLLM 自己的约定把 KV 取一遍，再用喂给 kernel 的那套张量取一遍，比两次取到的行。

    窗口的参考实现取 pypto-lib 自己的 `swa_indices_and_lens`（纯 python 双层循环），
    与适配器里那套向量化推导是两份独立实现，对上了才算数。
    """
    import torch
    from utils import swa_indices_and_lens

    v: dict = {}
    S, BS, WIN = csa.S, csa.BLOCK_SIZE, csa.WIN
    CMP_PAGE = csa.CMP_STORAGE_BLOCK_SIZE
    n = int(d["n"])
    dev = d["q_t"].device
    take = torch.arange(n, device=dev, dtype=torch.int64) * S

    # --- 1. 编译期常量 vs vLLM 运行期配置 ---
    pairs = {
        "H / n_local_heads": (csa.H, impl.n_local_heads),
        "HEAD_DIM / head_dim": (csa.HEAD_DIM, impl.head_dim),
        "NOPE_DIM / nope_head_dim": (csa.NOPE_DIM, impl.nope_head_dim),
        "ROPE_DIM / rope_head_dim": (csa.ROPE_DIM, impl.rope_head_dim),
        "WIN / window_size": (csa.WIN, impl.window_size),
        "IDX_TOPK / index_topk": (csa.IDX_TOPK, impl.index_topk),
        "COMPRESS_RATIO / compress_ratio": (csa.COMPRESS_RATIO, impl.compress_ratio),
        "SOFTMAX_SCALE / softmax_scale": (round(float(csa.SOFTMAX_SCALE), 12), round(float(impl.softmax_scale), 12)),
        "BLOCK_SIZE / ori_kv.shape[1]": (csa.BLOCK_SIZE, int(st["ori_kv"].shape[1])),
        "O_GROUPS / n_local_groups": (csa.O_GROUPS, impl.n_local_groups),
        "O_LORA / o_lora_rank": (csa.O_LORA, impl.o_lora_rank),
        "D / wo_b.weight.shape[0]": (csa.D, int(impl.wo_b.weight.shape[0])),
    }
    bad = {k: {"pto": a, "vllm": b} for k, (a, b) in pairs.items() if a != b}
    v["consts"] = {"checked": len(pairs), "mismatched": bad}

    # --- 2. seqlen：PTO 不收 seqused_kv，靠 position_ids 与窗口表达 ---
    seq = st["seqused_kv"].to(dev).to(torch.int64).reshape(-1)[:n]
    pos = st["positions"].to(dev).to(torch.int64).reshape(-1)[:n]
    pos_fed = d["pos_t"].to(dev).to(torch.int64).index_select(0, take)
    v["seqlen"] = {
        "seqused_kv": seq.tolist(),
        "positions": pos.tolist(),
        "pos == seqused_kv - 1 的反例数": int((pos != seq - 1).sum()),
        "喂给 kernel 的 position 与 vLLM 的反例数": int((pos_fed != pos).sum()),
    }

    # --- 3. 窗口：内容对拍（参考实现来自 pypto-lib 自己的 utils） ---
    obt = st["ori_block_table"][:n].detach().cpu()
    ref_slots, ref_lens = swa_indices_and_lens(pos.reshape(n, 1).cpu(), obt, block_size=BS, window=WIN)
    ref = ref_slots.to(dev).to(torch.int64)  # vLLM 缓存里的物理槽
    mine = d["win_new"].to(dev).to(torch.int64).index_select(0, take)
    ori_flat = st["ori_kv"].reshape(-1, st["ori_kv"].shape[-1])
    small_flat = d["ori_small"].reshape(-1, d["ori_small"].shape[-1])
    both = (ref >= 0) & (mine >= 0)
    a = small_flat.index_select(0, mine.clamp_min(0).reshape(-1))
    b = ori_flat.index_select(0, ref.clamp_min(0).reshape(-1))
    rowdiff = (a.float() - b.float()).abs().amax(dim=-1).reshape(n, -1)
    v["window"] = {
        "有效位不一致的格数": int((ref >= 0).ne(mine >= 0).sum()),
        "每 token 有效槽数(参考)": ref_lens.reshape(-1).tolist(),
        "取到的 KV 行最大差": float(rowdiff[both].max()) if bool(both.any()) else None,
        "对拍格数": int(both.sum()),
    }

    # --- 4. 压缩槽：同一个 slot，两边解析到的是不是同一行 ---
    idx = d["idx_t"].to(dev).to(torch.int64).index_select(0, take)
    valid = idx >= 0
    sl = idx.clamp_min(0)
    vcbt = st["cmp_block_table"].to(dev).to(torch.int64)[:n]
    vcol = torch.div(sl, BS, rounding_mode="floor").clamp(0, vcbt.shape[1] - 1)
    vslot = torch.gather(vcbt, 1, vcol) * BS + (sl - vcol * BS)
    pcbt = d["cmp_bt"].to(dev).to(torch.int64)[:n]
    pcol = torch.div(sl, CMP_PAGE, rounding_mode="floor").clamp(0, pcbt.shape[1] - 1)
    pslot = torch.gather(pcbt, 1, pcol) * CMP_PAGE + (sl - pcol * CMP_PAGE)
    cmp_flat = st["cmp_kv"].reshape(-1, st["cmp_kv"].shape[-1])
    cs_flat = d["cmp_small"].reshape(-1, d["cmp_small"].shape[-1])
    a = cs_flat.index_select(0, pslot.reshape(-1))
    b = cmp_flat.index_select(0, vslot.reshape(-1))
    cdiff = (a.float() - b.float()).abs().amax(dim=-1).reshape(n, -1)
    bound = ((pos + 1) // csa.COMPRESS_RATIO).unsqueeze(1)
    v["cmp"] = {
        "每 token 有效槽数": valid.sum(dim=1).tolist(),
        "取到的 KV 行最大差": float(cdiff[valid].max()) if bool(valid.any()) else None,
        "对拍格数": int(valid.sum()),
        "被 PTO 的 bound 掩掉的槽数": int(((idx >= bound) & valid).sum()),
    }

    # --- 5. RoPE 表：vLLM 成对复制 vs pypto-lib 前半真值 ---
    cflat = d["cos"].reshape(d["cos"].shape[0], -1).float()
    sflat = d["sin"].reshape(d["sin"].shape[0], -1).float()
    half = csa.ROPE_DIM // 2
    src = d["src"]
    # kernel 签名要 BF16，vLLM 的表是 FP32，所以参考值要先按 BF16 round 再比 —— 直接比
    # FP32 会看到约 2^-9 的常数差，那是表示精度不是对应关系错。
    cos_ref = cflat[:, 0::2][:, :half].index_select(0, src).to(torch.bfloat16)
    sin_ref = sflat[:, 0::2][:, :half].index_select(0, src).to(torch.bfloat16)
    v["rope"] = {
        "vLLM cos 相邻成对相等的最大差": float((cflat[:, 0::2] - cflat[:, 1::2]).abs().max()),
        "vLLM sin 相邻成对相等的最大差": float((sflat[:, 0::2] - sflat[:, 1::2]).abs().max()),
        "喂给 kernel 的 cos[:HALF] 与去重参考(BF16)的差": float(
            (d["cos_t"][:, :half].to(torch.bfloat16).float() - cos_ref.float()).abs().max()
        ),
        "喂给 kernel 的 sin[:HALF] 与去重参考(BF16)的差": float(
            (d["sin_t"][:, :half].to(torch.bfloat16).float() - sin_ref.float()).abs().max()
        ),
        "BF16 化相对 vLLM FP32 表的最大差": float(
            (d["cos_t"].float()[:, :half] - cflat[:, 0::2][:, :half].index_select(0, src)).abs().max()
        ),
    }

    # --- 6. 权重与 sink ---
    wa_v = impl.wo_a.weight.detach().to(torch.bfloat16)
    wb_v = impl.wo_b.weight.detach().float()
    deq = d["wo_b"].to(dev).float() * d["wo_b_scale"].to(dev).float().unsqueeze(-1)
    v["weights"] = {
        "wo_a 转置回去是否逐位相同": bool(torch.equal(d["wo_a"].to(dev).transpose(1, 2).contiguous(), wa_v)),
        "wo_b 反量化的相对最大误差": float((deq - wb_v).abs().max() / wb_v.abs().max()),
        "attn_sink 是否逐位相同": bool(
            torch.equal(d["sink"].to(dev).float(), impl.attn_sink.detach().float().reshape(-1))
        ),
        "q 行是否逐位相同": bool(torch.equal(d["q_t"].index_select(0, take), st["q"][:n])),
    }

    # --- 7. kv_cache 的读写方向 ---
    # `sparse_attn_test` 里只有 attn_out 是 pl.Out，两份 KV cache 都是纯输入；
    # cache 的写入仍由 vLLM 的 prolog（wkv + slot_mapping）完成，不在替换范围内。
    # 窗口末尾就是当前 token，所以 gather 必须发生在 prolog 写完之后 —— 下面这项就是在验它。
    cur = mine[:, -1]
    v["当前 token 的 KV 已在窗口内且非零"] = {
        "零行数": int((small_flat.index_select(0, cur.clamp_min(0)).float().abs().amax(dim=-1) == 0).sum()),
        "token 数": n,
    }

    # 两处差异是 kernel 签名与 checkpoint 的格式不同造成的，**不是对应关系错**，
    # 所以单独列出来而不计入判据 —— 把它们混进 pass/fail 会让这个校验永远红着，
    # 红着的检查等于没有检查。
    v["declared_gaps"] = {
        "wo_b: checkpoint 是 BF16，kernel 签名只有 INT8+per-channel scale": v["weights"]["wo_b 反量化的相对最大误差"],
        "RoPE 表: vLLM 用 FP32，kernel 签名要 BF16": v["rope"]["BF16 化相对 vLLM FP32 表的最大差"],
    }
    v["全部通过"] = (
        not bad
        and v["seqlen"]["pos == seqused_kv - 1 的反例数"] == 0
        and v["seqlen"]["喂给 kernel 的 position 与 vLLM 的反例数"] == 0
        and v["window"]["有效位不一致的格数"] == 0
        and (v["window"]["取到的 KV 行最大差"] or 0) == 0
        and (v["cmp"]["取到的 KV 行最大差"] or 0) == 0
        and v["rope"]["vLLM cos 相邻成对相等的最大差"] == 0
        and v["rope"]["喂给 kernel 的 cos[:HALF] 与去重参考(BF16)的差"] == 0
        and v["rope"]["喂给 kernel 的 sin[:HALF] 与去重参考(BF16)的差"] == 0
        and v["weights"]["wo_a 转置回去是否逐位相同"]
        and v["weights"]["attn_sink 是否逐位相同"]
        and v["weights"]["q 行是否逐位相同"]
        and v["当前 token 的 KV 已在窗口内且非零"]["零行数"] == 0
    )
    return v


_FORWARD_SEEN: dict[str, int] = {}


def note_forward(impl, layer_name, *, has_decode, has_prefill, md=None) -> None:
    """记录 `forward` 每次进来时的批次形态。

    替换只在"纯 decode 且这一层是 ratio-4"时发生。形态不符时既不会替换也不会留下痕迹，
    于是"一行 [pto-csa] 都没有"既可能是钩子没进来，也可能是每步都被前置条件挡掉——
    这一条把两者分开。
    """
    key = f"ratio={getattr(impl, 'compress_ratio', None)} decode={bool(has_decode)} prefill={bool(has_prefill)}"
    if md is not None:
        key += (
            f" num_decodes={getattr(md, 'num_decodes', '?')}"
            f" decode_tokens={getattr(md, 'num_decode_tokens', '?')}"
            f" num_prefills={getattr(md, 'num_prefills', '?')}"
            f" actual={getattr(md, 'num_actual_tokens', '?')}"
        )
    n = _FORWARD_SEEN.get(key, 0) + 1
    _FORWARD_SEEN[key] = n
    if n == 1:
        print(f"[pto-csa] forward 形态首次出现: {key} ({layer_name})", flush=True)


def _atexit_report():
    with suppress(Exception):
        dump_report()


atexit.register(_atexit_report)
