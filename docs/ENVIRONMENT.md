# Tested environment

The released experiment scripts were validated on Linux/aarch64 with four NVIDIA GH200 GPUs.
These compiled packages are deliberately not placed in `requirements.txt`, because their wheels
and build flags depend on the CUDA platform.

| Component | Tested version or revision |
|---|---|
| Python | 3.12 |
| PyTorch | 2.11.0+cu129 |
| CUDA toolkit | 12.9.1 |
| cuDNN | 9.17.1.4 |
| NCCL | 2.27.7 |
| SGLang | 0.5.15.post1, source revision `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` |
| sglang-kernel | 0.4.4+cu129 |
| Megatron-LM | source revision `1dcf0dafa884ad52ffb243625717a3471643e087` |
| Transformer Engine | 2.16.1+c9877be |
| sglang_router | 0.3.2+slime |

The tested SGLang tree applies Slime's `v0.5.15.post1` patches: `sglang.patch`,
`sglang-top_p.patch`, `sglang-release_hicache.patch`, `sglang-pull_weights.patch`, and
`sglang-deterministic.patch`. The tested Megatron-LM tree applies `megatron.patch` and
`megatron-sglang-aligned.patch` from this Slime version.

On the tested GH200/aarch64 stack, SGLang FA3 was unavailable, so all included launchers select
`--sglang-attention-backend triton`. Megatron attention remains configured independently with
`--attention-backend flash`.

The public scripts accept the following portable environment variables:

```bash
export CUDA_HOME=/path/to/cuda
export NCCL_HOME=/path/to/nccl          # optional when already on the loader path
export GCC_RUNTIME_LIB=/path/to/lib64   # optional
export CUDNN_ROOT=/path/to/cudnn        # defaults to the active environment
export MEGATRON_DIR=/path/to/Megatron-LM
export SGLANG_DIR=/path/to/sglang
```

Do not interpret this version table as a generic `pip install` recipe. PyTorch, Apex,
Transformer Engine, SGLang kernels, and the patched source trees must be installed for the target
GPU architecture.
