# 训练参数

## Profiler

LLaMA Factory 支持在训练过程中采集 PyTorch profiler trace，覆盖 CPU、CUDA GPU 和昇腾 NPU。可以在训练 YAML 中开启：

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

调度语义对齐官方 `torch.profiler.schedule` / `torch_npu.profiler.schedule`：先跳过 `profiler_skip_first` 个 step，之后每个周期按 `wait -> warmup -> active` 执行，并由 `profiler_repeat` 控制重复次数。callback 在每个 optimizer step 结束后调用一次 `prof.step()`。

昇腾 NPU 上可以通过 `profiler_level` 和 `profiler_aic_metrics` 控制采集深度：

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

字符串枚举参数固定为短值：`profiler_activities` 支持 `auto`、`all`、`cpu`、`device`；`profiler_rank_mode` 支持 `all`、`rank0`；`profiler_level` 支持 `none`、`level0`、`level1`、`level2`；`profiler_aic_metrics` 支持 `auto`、`none`、`pipe_utilization`、`arithmetic_utilization`、`memory`、`memory_l0`、`memory_ub`、`l2_cache`、`memory_access`、`resource_conflict_ratio`；`profiler_backend_options.npu.host_sys` 支持 `cpu`、`mem`、`disk`、`network`、`osrt`。传入不在列表内的值会直接报错。`profiler_backend_options` 必须是 YAML mapping，不支持 JSON 字符串。

不要同时通过 `PROF_CONFIG_PATH` 开启昇腾 `dynamic_profile` 和 LLaMA Factory 的 `enable_torch_profiler`。

官方参考：

- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [昇腾 CANN 9.0 PyTorch Profiler](https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/Profiling/atlasprofiling_16_0033.html)
