# Provenance and fork policy

## Repository model

Bernoulli OPD is maintained as a **Slime source fork plus a focused paper-reproduction layer**.
The core framework source is retained so that the exact training stack remains inspectable, while
generic upstream launchers and model presets unrelated to the paper are intentionally pruned from
`scripts/`.

The trade-off is explicit: this repository does not claim to be the canonical home for general
Slime development. Project maintainers will selectively synchronize security, correctness, and
compatibility fixes from upstream. Unrelated Slime features should be contributed upstream first.
Large upstream synchronizations must be isolated from paper-specific changes and record the old
and new upstream revisions.

## Upstream baseline

The fork was developed from the Slime 0.3.2-era source tree. The operational baseline used during
development referenced upstream Slime commit
`8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b` (2026-09-23); the upstream `v0.3.2` tag points to
`3778dbf6d1a533ab478ecf5ddaa11449a47752b2`.

The public repository was imported as a squashed source snapshot, so its Git history does not
encode byte-for-byte ancestry from a single upstream commit. The revisions above are therefore
reference baselines, not a claim that every inherited file is identical to either commit. Future
syncs must preserve this distinction in their commit messages.

Upstream project: <https://github.com/THUDM/slime>

Upstream documentation snapshots are retained in `docs/UPSTREAM_SLIME_README.md`,
`docs/UPSTREAM_SLIME_README_ZH.md`, and `docs/UPSTREAM_SLIME_CONTRIBUTING.md`.

## Paper-specific surface

The main project-specific implementation and reproduction surface includes:

- `slime/backends/megatron_utils/opd.py` and its integration in actor/model/loss code;
- `slime/rollout/on_policy_distillation.py`;
- OPD arguments and diagnostic logging;
- `configs/methods/`, `configs/settings/`, and the launch/evaluation scripts under `scripts/`;
- `examples/opsa/` and OPD/OPSA objective tests;
- release metadata, pinned model/data inputs, and reproduction documentation.

The paper implementation includes Vanilla OPD, SuRe, PowerOPD, GC-OPD, Bernoulli OPD, and
Balanced Bernoulli. The top-level `scripts/` entry points, two student-model presets,
`examples/opsa/`, and focused OPD/OPSA tests are maintained for the paper. Generic upstream
launchers, examples, documentation, conversion utilities, and hardware/model-specific tests are
excluded from the public release; they remain available from the upstream Slime project.

## Patched runtime dependencies

The validated runtime used:

| Dependency | Revision/version | Local changes |
|---|---|---|
| SGLang | `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` / 0.5.15.post1 | Slime patches from `docker/patch/v0.5.15.post1/` |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087` | `megatron.patch` and `megatron-sglang-aligned.patch` |
| Transformer Engine | 2.16.1+c9877be | platform build |

See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the complete tested stack. These dependencies
are not vendored into this repository.

## Licensing and attribution

The code remains under Apache License 2.0. Copyright and attribution from inherited files are
preserved. Paper figures are included for scholarly documentation and remain subject to the
authors' paper copyright.
