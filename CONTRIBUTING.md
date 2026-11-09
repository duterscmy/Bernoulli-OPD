# Contributing to Bernoulli OPD

Thank you for helping improve the Bernoulli OPD reference implementation. This repository is a
maintained source fork of Slime with a paper-reproduction layer; see [PROVENANCE.md](PROVENANCE.md)
for the boundary between upstream infrastructure and project-specific code.

## Scope

We welcome:

- correctness fixes for the OPD objectives, diagnostics, launchers, and evaluation pipeline;
- reproducibility improvements such as pinned inputs, clearer commands, and small tests;
- documentation fixes and portability improvements for supported GPU environments;
- narrowly scoped fixes to inherited Slime code when they are required by this project.

General Slime features and unrelated infrastructure changes should normally be proposed to
[THUDM/slime](https://github.com/THUDM/slime) first. Please do not combine an upstream sync,
algorithm change, and result update in one pull request.

## Before opening a pull request

1. Open an issue for changes that alter objective semantics, data, checkpoint selection, or
   reported numbers.
2. Branch from the current `main` and keep commits reviewable.
3. Add or update a test for mathematical and behavioral changes.
4. Record the exact model/data revisions and the command used for any reproduced result.
5. Do not commit checkpoints, datasets, generated results, logs, credentials, or compiled
   artifacts.

## Validation

Run the checks that are appropriate for the change:

```bash
git diff --check
bash -n build_conda.sh
find scripts examples/bopd -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
python -m compileall -q setup.py scripts/download_data.py \
  slime/backends/megatron_utils/opd.py slime/rollout/on_policy_distillation.py
```

Tests that import PyTorch, SGLang, Megatron-LM, Transformer Engine, or CUDA extensions must run
in a compatible GPU environment. In the Arrhenius development environment, run them on a GH200
compute node, never on the login node. Include relevant unit-test output and, for runtime changes,
the smallest successful smoke-test metrics in the pull request.

## Reproducibility expectations

- Preserve Vanilla OPD behavior unless the change explicitly targets it.
- Keep loss geometry separate from token selection and other ablations.
- Report seeds, model and dataset revisions, hardware, and all non-default overrides.
- Never replace a reported paper number silently. Explain the old and new protocol and retain an
  audit trail in the pull request.
- Keep platform-specific binary dependencies out of `requirements.txt`; document them in
  `docs/ENVIRONMENT.md`.

By contributing, you agree that your contribution is licensed under Apache License 2.0.
