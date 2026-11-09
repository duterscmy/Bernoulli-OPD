# Reproducing the paper

This checklist maps the public entry points to the experiments in
[arXiv:2611.02093v1](https://arxiv.org/abs/2611.02093v1). Run Python, model conversion, and
training commands on a compatible GPU compute node. The tested environment is recorded in
`docs/ENVIRONMENT.md`.

## 1. Prepare the environment

Use either the platform-specific Conda build or the CUDA image. Both commands must be run from
the repository root.

```bash
# Conda/micromamba environment; defaults to bopd-slime.
CONDA_ENV_NAME=bopd-slime bash build_conda.sh

# Or build the CUDA image from this source tree.
docker build -f docker/Dockerfile -t bernoulli-opd:1.0.0 .
```

Set the external source and artifact roots:

```bash
export MEGATRON_DIR=/path/to/Megatron-LM
export SGLANG_DIR=/path/to/sglang
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
```

## 2. Download immutable model snapshots

The setting files under `configs/settings/` are the machine-readable source of these revisions.

```bash
hf download Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --local-dir "$MODEL_ROOT/Qwen3-1.7B"
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --local-dir "$MODEL_ROOT/Qwen3-4B-Instruct-2507"
hf download pb09204048/Qwen3-8B-DAPO-iter449-disable-thinking \
  --revision 18a5b7dad35596cea0756c10330e453e95533ffa \
  --local-dir "$MODEL_ROOT/Qwen3-8B-DAPO-iter449-disable-thinking"
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --local-dir "$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B"
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --revision 916b56a44061fd5cd7d6a8fb632557ed4f724f60 \
  --local-dir "$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-7B"
hf download Skywork/Skywork-OR1-Math-7B \
  --revision fcef970f1d10682f13c182b280146a194ed57d1e \
  --local-dir "$MODEL_ROOT/Skywork-OR1-Math-7B"
```

Convert the two student checkpoints to Megatron distributed format:

```bash
source scripts/models/qwen3-1.7B.sh
PYTHONPATH="$MEGATRON_DIR:${PYTHONPATH:-}" python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$MODEL_ROOT/Qwen3-1.7B" \
  --save "$MODEL_ROOT/Qwen3-1.7B_torch_dist"

source scripts/models/deepseek-r1-distill-qwen-1.5B.sh
PYTHONPATH="$MEGATRON_DIR:${PYTHONPATH:-}" python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B" \
  --save "$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B_torch_dist"
```

## 3. Download pinned datasets

```bash
./scripts/download_data.sh "$DATA_ROOT"
```

The downloader checks row counts and writes the dataset repository, revision, and split to a
`source.json` file beside each JSONL file.

## 4. Run the unified training matrix

The paper's unified recipe uses 200 optimizer updates, global batch size 64, one response per
prompt, maximum rollout length 12,000, sampling `(temperature, top-p, top-k) = (1.0, 1.0, -1)`,
Adam betas `(0.9, 0.98)`, weight decay 0.1, constant learning rate `1e-6`, and gradient clipping
at 1.0. These are the defaults of `scripts/train.sh`.

Run an individual experiment:

```bash
./scripts/train.sh qwen3_4b bernoulli
```

Run every method and teacher setting sequentially on an allocated node:

```bash
for setting in qwen3_4b qwen3_8b_rl deepseek_7b skywork_7b; do
  ./scripts/train_all_methods.sh "$setting"
done
```

Or submit the full 24-run matrix through Slurm:

```bash
SLURM_ACCOUNT=<account> SLURM_PARTITION=<partition> ./scripts/submit_all.sh
```

The primary robustness experiment repeats Vanilla and Bernoulli OPD with three training seeds:

```bash
for seed in 1234 5678 13579; do
  SEED="$seed" RUN_NAME="qwen3_4b-vanilla-s${seed}" \
    ./scripts/train.sh qwen3_4b vanilla
  SEED="$seed" RUN_NAME="qwen3_4b-bernoulli-s${seed}" \
    ./scripts/train.sh qwen3_4b bernoulli
done
```

## 5. Run the PowerOPD reference recipe

Table 2 reports the better of the unified and reference PowerOPD recipes for each teacher. The
reference recipe changes the settings listed in Appendix A.4 while retaining `alpha=5`:

```bash
for setting in qwen3_4b qwen3_8b_rl deepseek_7b skywork_7b; do
  NUM_ROLLOUT=1500 \
  ROLLOUT_BATCH_SIZE=32 \
  GLOBAL_BATCH_SIZE=32 \
  MAX_RESPONSE_LEN=1024 \
  ROLLOUT_TEMPERATURE=1.0 \
  ROLLOUT_TOP_P=0.95 \
  ROLLOUT_TOP_K=20 \
  LEARNING_RATE=5e-7 \
  LR_WARMUP_FRACTION=0.05 \
  ADAM_BETA1=0.9 \
  ADAM_BETA2=0.999 \
  WEIGHT_DECAY=0 \
  RUN_NAME="${setting}-poweropd-reference-s1234" \
    ./scripts/train.sh "$setting" poweropd
done
```

## 6. Select and evaluate checkpoints

Training evaluates AIME24 Avg@8 every 20 updates with temperature 0.7, top-p 0.8, top-k 20, and
the setting-specific maximum response length. Select the checkpoint with the highest AIME24
Avg@8, then run:

```bash
./scripts/evaluate.sh <setting> <checkpoint-root> <step> validation
./scripts/evaluate.sh <setting> <checkpoint-root> <best-step> final
```

Final mode evaluates AIME24, AIME25, HMMT25, and AMC23 with Avg@32, then MATH500 with Avg@4.
The Qwen3-8B-DAPO-RL setting uses maximum response length 38,912; the other settings use 32,168.

## 7. Record the run

Retain the Git commit, setting/method, seed, model and dataset revisions, full environment,
launcher output, checkpoint-selection metric, selected step, and final benchmark outputs. Compare
against `RESULTS.md`; do not treat exact score equality as a deterministic software test.
