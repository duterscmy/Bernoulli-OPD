#!/usr/bin/env python3
"""Retain the best-eval checkpoint and, by default, the final checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


CHECKPOINT_RE = re.compile(r"iter_(\d{7})$")
EVAL_RE = re.compile(r"eval (\d+): \{(.*)\}")


def checkpoint_steps(save_dir: Path) -> dict[int, Path]:
    checkpoints = {}
    for path in save_dir.glob("iter_*"):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if path.is_dir() and match:
            checkpoints[int(match.group(1))] = path
    return checkpoints


def metric_by_step(log_file: Path, metric: str) -> dict[int, float]:
    metric_re = re.compile(rf"['\"]{re.escape(metric)}['\"]:\s*([-+0-9.eE]+)")
    scores = {}
    with log_file.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            eval_match = EVAL_RE.search(line)
            if not eval_match:
                continue
            metric_match = metric_re.search(eval_match.group(2))
            if metric_match:
                scores[int(eval_match.group(1))] = float(metric_match.group(1))
    return scores


def previous_best(save_dir: Path) -> tuple[int, float] | None:
    metadata_path = save_dir / "checkpoint_retention.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return int(metadata["best_step"]), float(metadata["best_score"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def latest_complete_step(save_dir: Path) -> int:
    tracker_path = save_dir / "latest_checkpointed_iteration.txt"
    if not tracker_path.is_file():
        raise RuntimeError(f"checkpoint tracker not found: {tracker_path}")
    value = tracker_path.read_text(encoding="utf-8").strip()
    if not value.isdigit():
        raise RuntimeError(f"invalid checkpoint tracker value {value!r} in {tracker_path}")
    return int(value)


def retention_plan(
    save_dir: Path,
    log_file: Path,
    metric: str,
    keep_steps: tuple[int, ...] = (),
    keep_last: bool = True,
) -> dict:
    checkpoints = checkpoint_steps(save_dir)
    if not checkpoints:
        raise RuntimeError(f"no iter_XXXXXXX checkpoints found in {save_dir}")

    last_step = latest_complete_step(save_dir)
    if last_step not in checkpoints:
        raise RuntimeError(f"latest complete checkpoint iter_{last_step:07d} is missing from {save_dir}")

    scores = metric_by_step(log_file, metric)
    scored_checkpoints = {step: scores[step] for step in checkpoints.keys() & scores.keys()}
    old_best = previous_best(save_dir)
    if old_best is not None and old_best[0] in checkpoints:
        scored_checkpoints.setdefault(old_best[0], old_best[1])
    if not scored_checkpoints:
        raise RuntimeError(f"no checkpoint-aligned {metric!r} metrics found in {log_file}")

    # Prefer the later checkpoint when several evaluations share the best score.
    best_step = max(scored_checkpoints, key=lambda step: (scored_checkpoints[step], step))
    missing_keep_steps = sorted(set(keep_steps) - checkpoints.keys())
    if missing_keep_steps:
        raise RuntimeError(f"requested retained checkpoints are missing: {missing_keep_steps}")

    retained_steps = sorted({best_step, *keep_steps} | ({last_step} if keep_last else set()))
    # A directory newer than the tracker can only be an in-progress or interrupted
    # save. Never delete it here; the next successful save can supersede it safely.
    removed_steps = sorted(step for step in checkpoints if step <= last_step and step not in retained_steps)
    return {
        "metric": metric,
        "best_step": best_step,
        "best_score": scored_checkpoints[best_step],
        "last_step": last_step,
        "protected_steps": sorted(set(keep_steps)),
        "keep_last": keep_last,
        "retained_steps": retained_steps,
        "removed_steps": removed_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--metric", default="eval/aime")
    parser.add_argument(
        "--keep-step",
        type=int,
        action="append",
        default=[],
        help="additionally retain this checkpoint step; may be specified more than once",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete non-retained checkpoint directories; otherwise print the plan only",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="after a completed run, retain the best checkpoint but not an additional final checkpoint",
    )
    args = parser.parse_args()

    plan = retention_plan(
        args.save_dir,
        args.log_file,
        args.metric,
        tuple(args.keep_step),
        keep_last=not args.best_only,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.apply:
        return

    checkpoints = checkpoint_steps(args.save_dir)
    for step in plan["removed_steps"]:
        shutil.rmtree(checkpoints[step])

    metadata_path = args.save_dir / "checkpoint_retention.json"
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(metadata_path)


if __name__ == "__main__":
    main()
