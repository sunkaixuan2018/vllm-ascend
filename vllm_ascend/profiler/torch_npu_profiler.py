#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import os
from contextlib import suppress
from typing import Any

import torch_npu
from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import WorkerProfiler

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config


class TorchNPUProfilerWrapper(WorkerProfiler):
    """Subclass of vLLM ``WorkerProfiler`` that wires in ``torch_npu.profiler``."""

    def __init__(self, profiler_config: ProfilerConfig, trace_name: str) -> None:
        super().__init__(profiler_config)
        self.profiler: Any = self._create_profiler(profiler_config, trace_name)

    @staticmethod
    def _create_profiler(profiler_config: ProfilerConfig, trace_name: str) -> Any:
        if profiler_config.profiler != "torch":
            raise RuntimeError(f"Unrecognized profiler: {profiler_config.profiler}")
        if not profiler_config.torch_profiler_dir:
            raise RuntimeError("torch_profiler_dir cannot be empty.")
        msmonitor_use_daemon = envs_ascend.MSMONITOR_USE_DAEMON
        with suppress(RuntimeError):
            msmonitor_use_daemon = get_ascend_config().msmonitor_use_daemon
        if msmonitor_use_daemon:
            raise RuntimeError("MSMONITOR_USE_DAEMON and torch profiler cannot be both enabled at the same time.")

        # Level2 额外带上 AICPU 与通信算子，device 侧的时间线才算完整。
        level = {
            "Level0": torch_npu.profiler.ProfilerLevel.Level0,
            "Level1": torch_npu.profiler.ProfilerLevel.Level1,
            "Level2": torch_npu.profiler.ProfilerLevel.Level2,
        }[(os.environ.get("VLLM_ASCEND_PROFILER_LEVEL") or "Level1").strip()]
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=level,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        return torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            # with_modules 只给到模块层级，看不到 Python 帧。要在 trace 里读出
            # 某个 kernel 是从哪段 Python 下发的（例如 CSA 替换走的是 vendor 还是
            # PTO），必须开 with_stack —— 代价是明显的采集开销，所以仍然由
            # --profiler-config.torch_profiler_with_stack 控制，默认行为不变。
            with_stack=profiler_config.torch_profiler_with_stack,
            profile_memory=profiler_config.torch_profiler_with_memory,
            with_modules=profiler_config.torch_profiler_with_stack,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                profiler_config.torch_profiler_dir,
                worker_name=trace_name,
            ),
        )

    def _start(self) -> None:
        self.profiler.start()

    def _stop(self) -> None:
        self.profiler.stop()

    def _profiler_step(self) -> bool:
        return True
