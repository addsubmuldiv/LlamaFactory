# Training Argument

## Profiler

LLaMA Factory can collect PyTorch profiler traces during training on CPU, CUDA GPU, and Ascend NPU devices. Enable it in the training YAML:

```yaml
enable_torch_profiler: true
profiler_output_dir: ./saves/profile
profiler_skip_first: 8
profiler_wait_steps: 0
profiler_warmup_steps: 1
profiler_active_steps: 3
profiler_repeat: 1
profiler_rank_mode: rank0
```

The schedule follows the official `torch.profiler.schedule` / `torch_npu.profiler.schedule` semantics: first skip `profiler_skip_first` steps, then each cycle runs `wait -> warmup -> active`, repeated by `profiler_repeat`. The callback calls `prof.step()` once after each optimizer step.

For Ascend NPU, use `profiler_level` and `profiler_aic_metrics` to control collection depth:

```yaml
profiler_level: level1
profiler_aic_metrics: pipe_utilization
profiler_backend_options:
  npu:
    data_simplification: true
    host_sys: [cpu, mem]
    sys_io: false
    sys_interconnection: false
```

String enum values are fixed short values. `profiler_activities` supports `auto`, `all`, `cpu`, `device`; `profiler_rank_mode` supports `all`, `rank0`; `profiler_level` supports `none`, `level0`, `level1`, `level2`; `profiler_aic_metrics` supports `auto`, `none`, `pipe_utilization`, `arithmetic_utilization`, `memory`, `memory_l0`, `memory_ub`, `l2_cache`, `memory_access`, `resource_conflict_ratio`; `profiler_backend_options.npu.host_sys` supports `cpu`, `mem`, `disk`, `network`, `osrt`. Values outside these lists fail validation. `profiler_backend_options` must be a YAML mapping, not a JSON string.

Do not enable Ascend `dynamic_profile` through `PROF_CONFIG_PATH` at the same time as `enable_torch_profiler`.

Official references:

- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [Ascend CANN 9.0 PyTorch Profiler](https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/Profiling/atlasprofiling_16_0033.html)
