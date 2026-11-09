# Paper results

These are the reported results from Table 2 of
[Bernoulli On-Policy Distillation](https://arxiv.org/abs/2611.02093v1). The first four benchmarks
use Avg@32 and MATH500 uses Avg@4. Checkpoints were selected on AIME24 Avg@8 as described in the
paper. GC-OPD is a diagnostic objective; Bernoulli OPD is the main method.

| Teacher / method | AIME24 | AIME25 | HMMT25 | AMC23 | MATH500 | Avg. | Delta vs. Vanilla |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-4B-Instruct-2507** | | | | | | | |
| Vanilla OPD | 33.96 | 21.77 | 15.00 | 68.36 | 82.80 | 44.38 | - |
| SuRe | 35.10 | 23.02 | 14.48 | 69.84 | 82.30 | 44.95 | +0.57 |
| PowerOPD | 38.65 | 26.46 | 18.23 | 71.09 | 83.90 | 47.67 | +3.29 |
| GC-OPD | 38.02 | 27.60 | 18.96 | 75.39 | 85.60 | 49.11 | +4.73 |
| Bernoulli OPD | **39.58** | **33.02** | **19.90** | **78.67** | **87.65** | **51.76** | **+7.38** |
| **Qwen3-8B-DAPO-RL** | | | | | | | |
| Vanilla OPD | 50.21 | 39.79 | 25.42 | 82.19 | 89.20 | 57.36 | - |
| SuRe | 48.75 | 39.58 | 25.42 | 82.66 | 88.80 | 57.04 | -0.32 |
| PowerOPD | 52.50 | 39.17 | 24.38 | 85.31 | 89.45 | 58.16 | +0.80 |
| GC-OPD | 52.50 | 40.00 | 24.38 | 85.94 | 86.60 | 57.88 | +0.52 |
| Bernoulli OPD | 51.04 | **41.25** | **25.42** | 82.97 | **91.20** | **58.38** | **+1.02** |
| **DeepSeek-R1-Distill-Qwen-7B** | | | | | | | |
| Vanilla OPD | 28.44 | 23.02 | 15.10 | 70.31 | 82.75 | 43.92 | - |
| SuRe | 29.06 | 23.02 | 14.17 | 70.31 | 82.90 | 43.89 | -0.03 |
| PowerOPD | 30.63 | 23.96 | 13.75 | 70.16 | 82.60 | 44.22 | +0.30 |
| GC-OPD | 31.15 | 23.33 | 14.27 | 70.16 | 82.75 | 44.33 | +0.41 |
| Bernoulli OPD | 30.10 | 23.85 | 14.38 | **72.66** | **83.50** | **44.90** | **+0.98** |
| **Skywork-OR1-Math-7B** | | | | | | | |
| Vanilla OPD | 35.00 | 27.60 | 16.46 | 75.78 | 85.40 | 48.05 | - |
| SuRe | 33.33 | 27.19 | 16.77 | 73.59 | 83.85 | 46.95 | -1.10 |
| PowerOPD | 34.90 | 27.40 | 16.46 | 75.94 | 84.80 | 47.90 | -0.15 |
| GC-OPD | 38.13 | 27.19 | 17.50 | 73.91 | 85.45 | 48.44 | +0.39 |
| Bernoulli OPD | **40.42** | 26.88 | **17.71** | **79.22** | **86.40** | **50.13** | **+2.08** |

PowerOPD is reported using the better result from the unified recipe and the authors' faithful
reference recipe for each teacher setting. See `REPRODUCING.md` for both recipes; comparing only
the unified commands will not exactly reproduce every PowerOPD row above.

## Balanced Bernoulli ablation

| Teacher | Bernoulli | Balanced Bernoulli | Delta |
|---|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | 51.76 | 51.87 | +0.11 |
| Qwen3-8B-DAPO-RL | 58.38 | 58.65 | +0.27 |
| DeepSeek-R1-Distill-Qwen-7B | 44.90 | 44.96 | +0.06 |
| Skywork-OR1-Math-7B | 50.13 | 49.21 | -0.92 |

Reported numbers are research results, not CI acceptance thresholds. Hardware nondeterminism,
training seed, and checkpoint selection can change a reproduction run; retain the full run
metadata when comparing results.
