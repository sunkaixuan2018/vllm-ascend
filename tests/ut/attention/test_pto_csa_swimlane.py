# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU coverage for opt-in profiling of live CSA kernel dispatches."""

import importlib.util
import json
import subprocess
import sys
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention import pto_csa


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(pto_csa, "_RUNNER", None)
    monkeypatch.setenv("PTO_CSA_MODE", "kernel")
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_SWIMLANE_LEVEL", "4")
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_SWIMLANE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_SWIMLANE_MAX_CAPTURES", "2")
    monkeypatch.setattr(pto_csa, "_capture_active", lambda: False)
    npu = SimpleNamespace(current_device=lambda: 0, is_current_stream_capturing=Mock(return_value=False))
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True))
    artifact_dir = tmp_path / "kernel"
    artifact_dir.mkdir()
    (artifact_dir / "binary_manifest.json").write_text(json.dumps({"kernels": [{"func_id": 0, "name": "CSA"}]}))
    resolve = Mock(return_value=SimpleNamespace(output_dir=artifact_dir))
    csa = SimpleNamespace(sparse_attn_test=SimpleNamespace(_resolve_kernel_artifact=resolve))
    monkeypatch.setattr(pto_csa.PtoCsaRunner, "_lazy_import", lambda self: self._csa or csa)

    events = []
    state = {}

    def init(**kwargs):
        state.update(kwargs)
        state["window"] = 0
        events.append("init")

    def end():
        events.append("end")
        root = state["output_dir"]
        output = root if state["window"] == 0 else root / f"window_{state['window']}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "chip_swimlane_records.json").write_text(json.dumps({"aicore_tasks": [[0, 0, 0, 1, 2]]}))
        (output / "deps.json").write_text("{}")
        state["window"] += 1

    adapter = ModuleType("pypto.torch")
    adapter.init = Mock(side_effect=init)
    adapter.begin_dfx = Mock(side_effect=lambda: events.append("begin"))
    adapter.end_dfx = Mock(side_effect=end)
    monkeypatch.setitem(sys.modules, "pypto.torch", adapter)
    compiler = ModuleType("pypto.runtime")
    compiler.CompileOptions = SimpleNamespace
    monkeypatch.setitem(sys.modules, "pypto.runtime", compiler)
    context = ModuleType("pypto.runtime.kernel.context")
    context.get_process_kernel_state = lambda: SimpleNamespace(require_config=lambda: SimpleNamespace(platform="a2a3"))
    monkeypatch.setitem(sys.modules, "pypto.runtime.kernel.context", context)

    def convert(command, **kwargs):
        events.append("convert")
        assert command[:3] == [sys.executable, "-m", "simpler_setup.tools.swimlane_converter"]
        assert Path(command[3]).is_file()
        assert command[4] == "--deps-json" and Path(command[5]).is_file()
        assert command[6] == "--func-names" and Path(command[7]).is_file()
        assert command[8] == "-o"
        assert kwargs == {"check": True, "timeout": 60}
        Path(command[-1]).write_text(json.dumps({"traceEvents": [{"name": "CSA", "ph": "X", "args": {"taskId": 0}}]}))

    converter = Mock(side_effect=convert)
    monkeypatch.setattr(pto_csa.subprocess, "run", converter)
    return SimpleNamespace(
        adapter=adapter, events=events, state=state, converter=converter, npu=npu, config=config, resolve=resolve
    )


def test_warmup_capture_limit_and_window_paths(runtime):
    runner = pto_csa.PtoCsaRunner()
    op = Mock(side_effect=lambda *_: runtime.events.append("launch"))
    args = (torch.empty(2, 4),)
    for _ in range(4):
        runner._run_kernel(op, args, "model.layers.1")

    assert runtime.events == [
        "init",
        "launch",
        "begin",
        "launch",
        "end",
        "convert",
        "begin",
        "launch",
        "end",
        "convert",
        "launch",
    ]
    assert op.call_count == 4  # No extra warmup or profiling re-execution.
    runtime.adapter.init.assert_called_once()
    assert runtime.state["enable_chip_swimlane"] == 4
    assert runtime.state["enable_dep_gen"] is True
    root = runtime.state["output_dir"]
    assert runner.stats["swimlane"] == [
        str(root / "merged_swimlane.json"),
        str(root / "window_1" / "merged_swimlane.json"),
    ]
    metadata = json.loads((root / "capture.json").read_text())
    assert metadata["layer"] == "model.layers.1"
    assert metadata["args"][0]["shape"] == [2, 4]
    assert json.loads((root / "name_map.json").read_text())["callable_id_to_name"] == {"0": "CSA"}
    runtime.resolve.assert_called_once_with(args, {"config": SimpleNamespace(platform="a2a3")})


def test_conflicting_specialization_names_are_rejected(runtime):
    runner = pto_csa.PtoCsaRunner()
    runner._swimlane_names = {"0": "different_kernel"}
    with pytest.raises(RuntimeError, match="conflicting swimlane names"):
        runner._remember_swimlane_names(())


def test_each_layer_and_shape_warms_outside_window(runtime):
    runner = pto_csa.PtoCsaRunner()
    op = Mock()
    for layer, shape in [("layer1", (2, 4)), ("layer2", (2, 4)), ("layer1", (3, 4))]:
        runner._run_kernel(op, (torch.empty(shape),), layer)
    assert op.call_count == 3
    runtime.adapter.begin_dfx.assert_not_called()
    assert runtime.resolve.call_count == 3
    assert runner._swimlane_names == {"0": "CSA"}
    runner._run_kernel(op, (torch.empty(3, 4),), "layer1")
    runtime.adapter.begin_dfx.assert_called_once_with()


def test_disabled_does_not_query_capture_or_export(runtime, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_SWIMLANE_LEVEL", "0")
    runner = pto_csa.PtoCsaRunner()
    op = Mock()
    runner._run_kernel(op, (), "layer1")
    runner._run_kernel(op, (), "layer1")
    runtime.adapter.init.assert_called_once_with()
    runtime.npu.is_current_stream_capturing.assert_not_called()
    runtime.adapter.begin_dfx.assert_not_called()
    runtime.converter.assert_not_called()
    assert op.call_count == 2


def test_graph_capture_requires_prior_init(runtime):
    runner = pto_csa.PtoCsaRunner()
    runtime.npu.is_current_stream_capturing.return_value = True
    op = Mock()
    with pytest.raises(RuntimeError, match="warmup before capture"):
        runner._run_kernel(op, (), "layer1")
    runtime.adapter.init.assert_not_called()
    runtime.adapter.begin_dfx.assert_not_called()
    op.assert_not_called()


@pytest.mark.parametrize(
    "variable,value",
    [
        ("VLLM_ASCEND_PTO_CSA_SWIMLANE_LEVEL", "5"),
        ("VLLM_ASCEND_PTO_CSA_SWIMLANE_LEVEL", "-1"),
        ("VLLM_ASCEND_PTO_CSA_SWIMLANE_DIR", ""),
        ("VLLM_ASCEND_PTO_CSA_SWIMLANE_MAX_CAPTURES", "0"),
        ("PTO_CSA_MODE", "program"),
    ],
)
def test_invalid_configuration_refused(runtime, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError):
        pto_csa.PtoCsaRunner()
    runtime.adapter.init.assert_not_called()


def test_graph_enabled_model_reserves_budget_for_replay(runtime):
    runtime.config.model_config.enforce_eager = False
    runner = pto_csa.PtoCsaRunner()
    runner.run_decode(SimpleNamespace(vllm_config=runtime.config), cos=None, sin=None)
    op = Mock()
    for _ in range(3):
        runner._run_kernel(op, (torch.empty(2, 4),), "layer1")
    assert op.call_count == 3
    runtime.adapter.begin_dfx.assert_not_called()
    runtime.resolve.assert_called_once()
    assert runner._swimlane_names == {"0": "CSA"}


def test_graph_capture_records_kernel_without_dfx_boundaries(runtime):
    runner = pto_csa.PtoCsaRunner()
    runner.ensure_kernel_init()
    runtime.npu.is_current_stream_capturing.return_value = True
    op = Mock()
    runner._run_kernel(op, (torch.empty(2, 4),), "layer1")
    op.assert_called_once()
    runtime.adapter.init.assert_called_once()
    runtime.adapter.begin_dfx.assert_not_called()
    runtime.adapter.end_dfx.assert_not_called()


def test_replay_collects_without_reentering_python_kernel(runtime, monkeypatch):
    runner = pto_csa.PtoCsaRunner()
    monkeypatch.setattr(pto_csa, "_RUNNER", runner)
    runner.ensure_kernel_init()
    for _ in range(3):
        with pto_csa.graph_replay_swimlane("FULL:batch=4"):
            runtime.events.append("replay")
    assert runtime.events == [
        "init",
        "begin",
        "replay",
        "end",
        "convert",
        "begin",
        "replay",
        "end",
        "convert",
        "replay",
    ]
    metadata = json.loads((runner._swimlane_dir / "capture.json").read_text())
    assert metadata["mode"] == "graph_replay"
    assert metadata["graph"] == "FULL:batch=4"


def test_replay_without_worker_does_not_initialize_pypto(runtime):
    with pto_csa.graph_replay_swimlane("FULL:batch=4"):
        runtime.events.append("replay")
    assert runtime.events == ["replay"]


def test_replay_inside_outer_capture_never_opens_dfx(runtime, monkeypatch):
    runner = pto_csa.PtoCsaRunner()
    monkeypatch.setattr(pto_csa, "_RUNNER", runner)
    runner.ensure_kernel_init()
    runtime.npu.is_current_stream_capturing.return_value = True
    with pto_csa.graph_replay_swimlane("nested"):
        runtime.events.append("replay")
    runtime.adapter.begin_dfx.assert_not_called()
    assert runtime.events == ["init", "replay"]


def test_replay_failure_closes_window_and_advances_output(runtime, monkeypatch):
    runner = pto_csa.PtoCsaRunner()
    monkeypatch.setattr(pto_csa, "_RUNNER", runner)
    runner.ensure_kernel_init()
    with pytest.raises(RuntimeError, match="replay failed"), pto_csa.graph_replay_swimlane("broken"):
        raise RuntimeError("replay failed")
    runtime.adapter.end_dfx.assert_called_once()
    runtime.converter.assert_not_called()
    with pto_csa.graph_replay_swimlane("working"):
        pass
    assert runner.stats["swimlane"] == [str(runner._swimlane_dir / "window_1" / "merged_swimlane.json")]


def test_replay_with_no_csa_records_is_not_reported_as_trace(runtime, monkeypatch, capsys):
    runner = pto_csa.PtoCsaRunner()
    monkeypatch.setattr(pto_csa, "_RUNNER", runner)
    runner.ensure_kernel_init()
    end = runtime.adapter.end_dfx.side_effect

    def empty_end():
        end()
        (runner._swimlane_dir / "chip_swimlane_records.json").write_text(json.dumps({"aicore_tasks": []}))
        (runner._swimlane_dir / "deps.json").unlink()

    runtime.adapter.end_dfx.side_effect = empty_end
    with pto_csa.graph_replay_swimlane("without_csa"):
        pass
    assert "no recorded CSA tasks" in capsys.readouterr().out
    runtime.converter.assert_not_called()
    assert "swimlane" not in runner.stats
    assert json.loads((runner._swimlane_dir / "capture.json").read_text())["status"] == "no_csa_tasks"


def test_failed_launch_closes_window(runtime):
    runner = pto_csa.PtoCsaRunner()
    args = (torch.empty(2, 4),)
    runner._run_kernel(Mock(), args, "layer1")
    with pytest.raises(RuntimeError, match="launch failed"):
        runner._run_kernel(Mock(side_effect=RuntimeError("launch failed")), args, "layer1")
    runtime.adapter.end_dfx.assert_called_once_with()
    runtime.converter.assert_not_called()


def test_failed_begin_does_not_close_another_window(runtime):
    runner = pto_csa.PtoCsaRunner()
    args = (torch.empty(2, 4),)
    runner._run_kernel(Mock(), args, "layer1")
    runtime.adapter.begin_dfx.side_effect = RuntimeError("already open")
    op = Mock()
    with pytest.raises(RuntimeError, match="already open"):
        runner._run_kernel(op, args, "layer1")
    op.assert_not_called()
    runtime.adapter.end_dfx.assert_not_called()


@pytest.mark.parametrize("failure", ["missing_records", "converter", "empty_trace", "metadata_only", "missing_names"])
def test_export_failure_is_visible(runtime, failure):
    runner = pto_csa.PtoCsaRunner()
    args = (torch.empty(2, 4),)
    runner._run_kernel(Mock(), args, "layer1")
    if failure == "missing_records":
        runtime.adapter.end_dfx.side_effect = None
    elif failure == "converter":
        runtime.converter.side_effect = subprocess.CalledProcessError(1, "converter")
    else:
        runtime.converter.side_effect = None
        if failure == "metadata_only":
            (runner._swimlane_dir / "merged_swimlane.json").write_text(
                json.dumps({"traceEvents": [{"ph": "M", "name": "process_name"}]})
            )
        elif failure == "missing_names":
            (runner._swimlane_dir / "merged_swimlane.json").write_text(
                json.dumps({"traceEvents": [{"ph": "X", "name": "func_0_a(t0)", "args": {"taskId": 0}}]})
            )
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        runner._run_kernel(Mock(), args, "layer1")
    assert "swimlane" not in runner.stats


def test_worker_output_directories_are_unique(runtime):
    first, second = pto_csa.PtoCsaRunner(), pto_csa.PtoCsaRunner()
    first.ensure_kernel_init()
    second.ensure_kernel_init()
    assert first._swimlane_dir != second._swimlane_dir
    assert first._swimlane_dir.parent == second._swimlane_dir.parent


@pytest.mark.parametrize("mode", ["kernel", "graph"])
def test_live_run_decode_profiles_actual_kernel_without_extra_invocations(runtime, monkeypatch, mode):
    monkeypatch.setenv("PTO_CSA_MODE", mode)
    runner = pto_csa.PtoCsaRunner()
    op = Mock(side_effect=lambda *args: args[-1].fill_(7))
    op._resolve_kernel_artifact = runtime.resolve
    runner._csa = SimpleNamespace(
        T=2,
        B=1,
        S=2,
        H=1,
        HEAD_DIM=2,
        D=2,
        WIN=2,
        BLOCK_SIZE=4,
        CMP_STORAGE_BLOCK_SIZE=1,
        IDX_TOPK=1,
        ROPE_DIM=2,
        sparse_attn_test=op,
    )
    monkeypatch.setattr(runner, "weights_for", lambda *_: (torch.ones(1),) * 3)
    monkeypatch.setattr(runner, "ensure_registered", lambda: op)
    monkeypatch.setattr(pto_csa, "_dump_once", lambda *_: None)
    monkeypatch.setattr(pto_csa, "_VERIFY", False)
    impl = SimpleNamespace(
        vllm_config=runtime.config,
        attn_sink=torch.zeros(1),
        _pto_csa_stash={
            "q": torch.ones(1, 1, 2),
            "ori_kv": torch.ones(2, 4, 1, 2),
            "cmp_kv": torch.ones(2, 4, 1, 2),
            "ori_block_table": torch.zeros(1, 1, dtype=torch.int32),
            "cmp_block_table": torch.zeros(1, 1, dtype=torch.int32),
            "cmp_sparse_indices": torch.zeros(1, 1, dtype=torch.int32),
            "positions": torch.ones(1, dtype=torch.int32),
        },
    )
    for _ in range(2):
        output = runner.run_decode(impl, cos=torch.ones(1, 2), sin=torch.zeros(1, 2), layer_name="layer1")
        torch.testing.assert_close(output, torch.full((1, 2), 7, dtype=torch.bfloat16))
    assert op.call_count == 2
    runtime.adapter.begin_dfx.assert_called_once_with()
    runtime.adapter.end_dfx.assert_called_once_with()
    metadata = json.loads((runner._swimlane_dir / "capture.json").read_text())
    assert metadata["layer"] == "layer1"
    assert metadata["dispatch"] == 2


@pytest.fixture
def graph_modules(runtime, monkeypatch):
    """Load the real graph wrappers with CPU substitutes for vLLM/NPU dependencies."""

    class Mode(Enum):
        NONE = 0
        FULL = 1
        PIECEWISE = 2

    context = SimpleNamespace(batch_descriptor="batch4", cudagraph_runtime_mode=Mode.FULL)

    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    module("vllm", __path__=[])
    module("vllm.envs", VLLM_LOGGING_LEVEL="INFO")
    module("vllm.compilation.counter", compilation_counter=SimpleNamespace(num_cudagraph_captured=0))
    module("vllm.compilation.cuda_graph", CUDAGraphOptions=Mock)
    module("vllm.compilation.monitor", validate_cudagraph_capturing_enabled=Mock())
    module("vllm.config", CUDAGraphMode=Mode, VllmConfig=object)
    module("vllm.config.compilation", CUDAGraphMode=Mode)
    module(
        "vllm.forward_context",
        BatchDescriptor=object,
        get_forward_context=lambda: context,
        set_forward_context=lambda *a, **kw: nullcontext(),
    )
    module("vllm.logger", logger=Mock())
    module("vllm.platforms", current_platform=Mock())
    module("vllm_ascend.ascend_forward_context", _EXTRA_CTX=SimpleNamespace(is_draft_model=False))
    module("vllm_ascend.utils", weak_ref_tensors=lambda x: x)
    module("torch_npu", _C=SimpleNamespace(_NPUTaskGroupHandle=object))
    runtime.npu.NPUGraph = object
    runtime.npu.ExternalEvent = object
    runtime.npu.current_stream = lambda: SimpleNamespace(synchronize=Mock())

    root = Path(pto_csa.__file__).parents[1]

    def load(name, relative_path):
        spec = importlib.util.spec_from_file_location(name, root / relative_path)
        value = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, value)
        spec.loader.exec_module(value)
        return value

    acl = load("vllm_ascend.compilation.acl_graph", "compilation/acl_graph.py")

    class BaseManager:
        def run_fullgraph(self, desc):
            return self.base_replay()

    module("vllm.sequence", IntermediateTensors=dict)
    module("vllm.v1.kv_cache_interface", KVCacheConfig=object)
    module("vllm.v1.worker.gpu.block_table", BlockTables=object)
    module("vllm.v1.worker.gpu.cudagraph_utils", BatchExecutionDescriptor=object, ModelCudaGraphManager=BaseManager)
    module("vllm.v1.worker.gpu.input_batch", InputBuffers=object)
    module("vllm.v1.worker.gpu.model_states.interface", ModelState=object)
    module("vllm.v1.worker.utils", AttentionGroup=object)
    module("vllm_ascend.worker.v2.utils", communicator_switch=nullcontext)
    v2 = load("vllm_ascend.worker.v2.aclgraph_utils", "worker/v2/aclgraph_utils.py")
    return acl, v2, Mode, context


@pytest.mark.parametrize("mode", ["FULL", "PIECEWISE"])
@pytest.mark.parametrize("enabled", [True, False])
def test_aclgraph_wrapper_collects_around_real_replay_branch(runtime, monkeypatch, graph_modules, mode, enabled):
    acl, _, modes, context = graph_modules
    runner = pto_csa.PtoCsaRunner()
    monkeypatch.setattr(pto_csa, "_RUNNER", runner)
    runner.ensure_kernel_init()
    if not enabled:
        monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_SWIMLANE_LEVEL", "0")
    mode = getattr(modes, mode)
    context.cudagraph_runtime_mode = mode
    wrapper = acl.ACLGraphWrapper(Mock(), SimpleNamespace(compilation_config=None), mode)
    output = object()
    replay = Mock(side_effect=lambda: runtime.events.append("replay"))
    wrapper.concrete_aclgraph_entries[context.batch_descriptor] = acl.ACLGraphEntry(
        context.batch_descriptor, aclgraph=SimpleNamespace(replay=replay), output=output
    )
    assert wrapper() is output
    replay.assert_called_once_with()
    wrapper.runnable.assert_not_called()
    assert runtime.events == (["init", "begin", "replay", "end", "convert"] if enabled else ["init", "replay"])


@pytest.mark.parametrize("enabled", [True, False])
def test_v2_fullgraph_preserves_return_and_parameter_update_order(runtime, monkeypatch, graph_modules, enabled):
    _, v2, modes, _ = graph_modules
    runner = pto_csa.PtoCsaRunner()
    monkeypatch.setattr(pto_csa, "_RUNNER", runner)
    runner.ensure_kernel_init()
    if not enabled:
        monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_SWIMLANE_LEVEL", "0")
    manager = object.__new__(v2.ModelAclGraphManager)
    output = object()

    def replay():
        runtime.events.append("replay")
        return output

    manager.base_replay = Mock(side_effect=replay)
    manager.device = torch.device("cpu")
    manager.vllm_config = SimpleNamespace()
    manager.model_runner = SimpleNamespace(
        input_buffers=SimpleNamespace(positions=torch.zeros(4)),
        dp_size=1,
        model_state=SimpleNamespace(attn_metadata=None),
        attn_backends={"test": object()},
        update_stream=None,
        speculative_config=None,
    )
    monkeypatch.setattr(v2, "update_full_graph_params", lambda *a: runtime.events.append("update"))
    assert manager.run_fullgraph(SimpleNamespace(num_tokens=4, cg_mode=modes.FULL)) is output
    manager.base_replay.assert_called_once_with()
    assert runtime.events == (
        ["init", "begin", "replay", "update", "end", "convert"] if enabled else ["init", "replay", "update"]
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--confcutdir", str(Path(__file__).parent)])
