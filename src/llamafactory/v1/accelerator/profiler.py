from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llamafactory.extras.profiler import ProfilerController

from ..utils import logging
from ..utils.callbacks import TrainerCallback, TrainerState


if TYPE_CHECKING:
    from ..config import TrainingArguments


logger = logging.get_logger(__name__)


class ProfilerCallback(TrainerCallback):
    def __init__(self, args: TrainingArguments) -> None:
        self.profiler = ProfilerController(args, logger=logger)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        self.profiler.start(args.output_dir)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        self.profiler.step()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        self.profiler.stop()
