# Environment compatibility

The release depends on the following major interfaces. These versions form the tested reference
stack; exact CUDA wheels and compiler flags should be selected for the user's GPU architecture.

| Component | Reference version or revision |
|---|---|
| Bernoulli OPD / Slime fork | 1.0.0, based on the Slime 0.3.2-era tree |
| Python | 3.12 |
| Ray | 2.58.0 |
| PyTorch | 2.11.0 |
| SGLang | 0.5.15.post1, `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087` |

The authors validated the paper runs on Linux/aarch64 with four NVIDIA GH200 GPUs, CUDA 12.9,
and cuDNN 9.17.1.4. The previous cuDNN 9.16 pin is incompatible with the tested PyTorch build and
must not be used for that stack. PyTorch, CUDA, cuDNN, NCCL, Apex, Transformer Engine, attention
kernels, and router wheels otherwise need to match the target hardware and do not have to copy
the GH200 build exactly.

The tested SGLang and Megatron trees use the patches under `docker/patch/v0.5.15.post1/`. On the
tested GH200 stack, SGLang rollout uses `--sglang-attention-backend triton`; users may select a
different supported backend after validating it on their hardware.

`build_conda.sh` and `docker/Dockerfile` are reference environment recipes, not universal binary
installers. The runtime entry points only require the external source paths and artifact roots:

```bash
export MEGATRON_DIR=/path/to/Megatron-LM
export SGLANG_DIR=/path/to/sglang
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
```
