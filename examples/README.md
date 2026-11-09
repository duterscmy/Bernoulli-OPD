# Paper examples

This release keeps only the Bernoulli OPD / OPSA launchers used by the paper reproduction flow:

- `opsa/run-qwen3-1.7B-opd-opsa-study.sh` — shared training launcher for all released objectives;
- `opsa/eval-qwen3-1.7B-best-checkpoint.sh` — checkpoint validation and final evaluation.

Use the stable wrappers under `scripts/` rather than invoking these implementation-level files
directly. See [`../REPRODUCING.md`](../REPRODUCING.md) for complete commands.
