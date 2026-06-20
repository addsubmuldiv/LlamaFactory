# Copyright 2025 the LlamaFactory team.
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

import inspect
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch


_SUPPORTED_ACTIVITIES = {"auto", "all", "cpu", "device"}
_SUPPORTED_RANK_MODES = {"all", "rank0"}


@dataclass
class ProfilerConfig:
    enabled: bool = False
    output_dir: Optional[str] = None
    start_step: Optional[int] = None
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 1
    repeat: int = 1
    interval_steps: Optional[int] = None
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = False
    with_modules: bool = False
    activities: str = "auto"
    rank_mode: str = "all"

    @classmethod
    def from_args(cls, args: Any) -> "ProfilerConfig":
        return cls(
            enabled=bool(getattr(args, "enable_profiler", False) or getattr(args, "enable_torch_profiler", False)),
            output_dir=getattr(args, "profiler_output_dir", None),
            start_step=getattr(args, "profiler_start_step", None),
            wait_steps=getattr(args, "profiler_wait_steps", 1),
            warmup_steps=getattr(args, "profiler_warmup_steps", 1),
            active_steps=getattr(args, "profiler_active_steps", 1),
            repeat=getattr(args, "profiler_repeat", 1),
            interval_steps=getattr(args, "profiler_interval_steps", None),
            record_shapes=getattr(args, "profiler_record_shapes", False),
            profile_memory=getattr(args, "profiler_profile_memory", False),
            with_stack=getattr(args, "profiler_with_stack", False),
            with_flops=getattr(args, "profiler_with_flops", False),
            with_modules=getattr(args, "profiler_with_modules", False),
            activities=getattr(args, "profiler_activities", "auto"),
            rank_mode=getattr(args, "profiler_rank_mode", "all"),
        )

    def validate(self) -> None:
        if not self.enabled:
            return

        if self.start_step is not None and self.start_step <= 0:
            raise ValueError("`profiler_start_step` must be a positive integer.")
        if self.wait_steps < 0:
            raise ValueError("`profiler_wait_steps` must be greater than or equal to 0.")
        if self.warmup_steps < 0:
            raise ValueError("`profiler_warmup_steps` must be greater than or equal to 0.")
        if self.active_steps <= 0:
            raise ValueError("`profiler_active_steps` must be a positive integer.")
        if self.repeat < 0:
            raise ValueError("`profiler_repeat` must be greater than or equal to 0.")
        if self.interval_steps is not None and self.interval_steps <= 0:
            raise ValueError("`profiler_interval_steps` must be a positive integer.")
        if self.activities not in _SUPPORTED_ACTIVITIES:
            raise ValueError(f"`profiler_activities` must be one of {sorted(_SUPPORTED_ACTIVITIES)}.")
        if self.rank_mode not in _SUPPORTED_RANK_MODES:
            raise ValueError(f"`profiler_rank_mode` must be one of {sorted(_SUPPORTED_RANK_MODES)}.")

        self.schedule_kwargs()

    def schedule_kwargs(self) -> dict[str, int]:
        if self.start_step is None:
            return dict(
                wait=self.wait_steps,
                warmup=self.warmup_steps,
                active=self.active_steps,
                repeat=self.repeat,
                skip_first=0,
            )

        wait_steps = 0
        if self.interval_steps is not None:
            wait_steps = self.interval_steps - self.warmup_steps - self.active_steps
            if wait_steps < 0:
                raise ValueError(
                    "`profiler_interval_steps` must be greater than or equal to "
                    "`profiler_warmup_steps + profiler_active_steps`."
                )

        skip_first = self.start_step - wait_steps - self.warmup_steps - 1
        if skip_first < 0:
            raise ValueError(
                "`profiler_start_step` is too early for the requested warmup/interval schedule. "
                "Increase `profiler_start_step`, reduce `profiler_warmup_steps`, or unset `profiler_interval_steps`."
            )

        return dict(
            wait=wait_steps,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=self.repeat,
            skip_first=skip_first,
        )


class _ProfilerBackend:
    def __init__(self, name: str, profiler_module: Any, device_activity_name: str) -> None:
        self.name = name
        self.profiler_module = profiler_module
        self.device_activity_name = device_activity_name

    def build_schedule(self, config: ProfilerConfig) -> Any:
        kwargs = config.schedule_kwargs()
        if self.name == "npu":
            return self.profiler_module.schedule(
                wait=kwargs["wait"],
                active=kwargs["active"],
                warmup=kwargs["warmup"],
                repeat=kwargs["repeat"],
                skip_first=kwargs["skip_first"],
            )

        return self.profiler_module.schedule(**kwargs)

    def build_activities(self, config: ProfilerConfig) -> list[Any]:
        activity_cls = self.profiler_module.ProfilerActivity
        activities = []
        if config.activities in ("auto", "all", "cpu"):
            activities.append(activity_cls.CPU)
        if config.activities in ("auto", "all", "device"):
            activities.append(getattr(activity_cls, self.device_activity_name))
        return activities

    def build_experimental_config(self, config: ProfilerConfig) -> Optional[Any]:
        if self.name != "npu" or config.activities == "cpu":
            return None

        profiler = self.profiler_module
        config_kwargs = dict(
            export_type=[profiler.ExportType.Text],
            profiler_level=profiler.ProfilerLevel.Level0,
            aic_metrics=profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
            host_sys=[],
            sys_io=False,
            sys_interconnection=False,
            mstx=False,
            mstx_domain_include=[],
            mstx_domain_exclude=[],
        )
        return profiler._ExperimentalConfig(
            **_filter_supported_kwargs(profiler._ExperimentalConfig, config_kwargs)
        )


def _get_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()

    return int(os.getenv("RANK", "0"))


def _get_current_accelerator_type() -> str:
    if hasattr(torch, "accelerator"):
        try:
            accelerator = torch.accelerator.current_accelerator(check_available=True)
            if accelerator is not None:
                return accelerator.type
        except Exception:
            pass

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                return "npu"
        except Exception:
            pass

    return "cpu"


def _get_backend() -> _ProfilerBackend:
    accelerator_type = _get_current_accelerator_type()
    if accelerator_type == "cuda":
        return _ProfilerBackend("cuda", torch.profiler, "CUDA")

    if accelerator_type == "npu":
        try:
            import torch_npu  # type: ignore
        except ImportError as exc:
            raise RuntimeError("NPU profiler requires `torch_npu` to be installed.") from exc

        return _ProfilerBackend("npu", torch_npu.profiler, "NPU")

    raise RuntimeError(f"Profiler only supports CUDA and NPU devices, got current accelerator: {accelerator_type}.")


def _filter_supported_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    supported = inspect.signature(fn).parameters
    return {key: value for key, value in kwargs.items() if key in supported}


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return

    rank_method = f"{level}_rank0"
    if hasattr(logger, rank_method):
        getattr(logger, rank_method)(message)
    else:
        getattr(logger, level)(message)


class ProfilerController:
    def __init__(self, args: Any, logger: Any = None) -> None:
        self.config = ProfilerConfig.from_args(args)
        self.logger = logger
        self.profiler = None
        self.backend_name: Optional[str] = None
        self.trace_dir: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start(self, output_dir: str) -> None:
        self.config.validate()
        if not self.config.enabled:
            return

        rank = _get_rank()
        if self.config.rank_mode == "rank0" and rank != 0:
            return

        backend = _get_backend()
        self.backend_name = backend.name

        trace_root = self.config.output_dir or os.path.join(output_dir, "profiler")
        self.trace_dir = os.path.realpath(os.path.join(trace_root, f"rank_{rank}"))
        os.makedirs(self.trace_dir, exist_ok=True)

        profile_kwargs = dict(
            activities=backend.build_activities(self.config),
            schedule=backend.build_schedule(self.config),
            on_trace_ready=backend.profiler_module.tensorboard_trace_handler(self.trace_dir),
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            with_modules=self.config.with_modules,
            experimental_config=backend.build_experimental_config(self.config),
        )
        self.profiler = backend.profiler_module.profile(
            **_filter_supported_kwargs(backend.profiler_module.profile, profile_kwargs)
        )
        self.profiler.start()

        schedule = self.config.schedule_kwargs()
        _log(
            self.logger,
            "info",
            (
                f"Profiler started on {backend.name}: skip_first={schedule['skip_first']}, "
                f"wait={schedule['wait']}, warmup={schedule['warmup']}, active={schedule['active']}, "
                f"repeat={schedule['repeat']}. Traces -> {trace_root}"
            ),
        )

    def step(self) -> None:
        if self.profiler is not None:
            self.profiler.step()

    def stop(self) -> None:
        if self.profiler is None:
            return

        self.profiler.stop()
        self.profiler = None
        _log(self.logger, "info", "Profiler stopped.")
