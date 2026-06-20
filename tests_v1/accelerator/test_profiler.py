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

import pytest

from llamafactory.extras.profiler import ProfilerConfig


def test_profiler_start_step_schedule():
    config = ProfilerConfig(enabled=True, start_step=10, warmup_steps=1, active_steps=3, repeat=1)

    assert config.schedule_kwargs() == {
        "wait": 0,
        "warmup": 1,
        "active": 3,
        "repeat": 1,
        "skip_first": 8,
    }


def test_profiler_interval_schedule():
    config = ProfilerConfig(
        enabled=True,
        start_step=20,
        warmup_steps=2,
        active_steps=3,
        repeat=2,
        interval_steps=10,
    )

    assert config.schedule_kwargs() == {
        "wait": 5,
        "warmup": 2,
        "active": 3,
        "repeat": 2,
        "skip_first": 12,
    }


def test_profiler_rejects_invalid_interval():
    config = ProfilerConfig(enabled=True, start_step=10, warmup_steps=2, active_steps=3, interval_steps=4)

    with pytest.raises(ValueError, match="profiler_interval_steps"):
        config.validate()
