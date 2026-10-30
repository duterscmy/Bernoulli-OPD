import json
from pathlib import Path

from scripts.retain_best_checkpoint import retention_plan


def _checkpoint(save_dir: Path, step: int) -> None:
    (save_dir / f"iter_{step:07d}").mkdir()


def test_retention_plan_keeps_best_and_latest_complete(tmp_path: Path) -> None:
    for step in (19, 39, 59):
        _checkpoint(tmp_path, step)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("59\n")
    log_file = tmp_path / "run.log"
    log_file.write_text(
        "eval 19: {'eval/aime': 0.10}\n"
        "eval 39: {'eval/aime': 0.30}\n"
        "eval 59: {'eval/aime': 0.20}\n"
    )

    plan = retention_plan(tmp_path, log_file, "eval/aime")

    assert plan["best_step"] == 39
    assert plan["last_step"] == 59
    assert plan["retained_steps"] == [39, 59]
    assert plan["removed_steps"] == [19]


def test_retention_plan_carries_best_across_resumed_job(tmp_path: Path) -> None:
    for step in (39, 59, 79):
        _checkpoint(tmp_path, step)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("79\n")
    (tmp_path / "checkpoint_retention.json").write_text(
        json.dumps({"best_step": 39, "best_score": 0.30})
    )
    log_file = tmp_path / "resumed.log"
    log_file.write_text("eval 79: {'eval/aime': 0.20}\n")

    plan = retention_plan(tmp_path, log_file, "eval/aime")

    assert plan["best_step"] == 39
    assert plan["last_step"] == 79
    assert plan["retained_steps"] == [39, 79]
    assert plan["removed_steps"] == [59]


def test_retention_plan_does_not_delete_unfinished_newer_directory(tmp_path: Path) -> None:
    for step in (19, 39, 59):
        _checkpoint(tmp_path, step)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("39\n")
    log_file = tmp_path / "run.log"
    log_file.write_text("eval 19: {'eval/aime': 0.10}\neval 39: {'eval/aime': 0.20}\n")

    plan = retention_plan(tmp_path, log_file, "eval/aime")

    assert plan["retained_steps"] == [39]
    assert plan["removed_steps"] == [19]
    assert 59 not in plan["removed_steps"]


def test_retention_plan_keeps_explicitly_protected_step(tmp_path: Path) -> None:
    for step in (39, 59, 79, 99, 119):
        _checkpoint(tmp_path, step)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("119\n")
    log_file = tmp_path / "run.log"
    log_file.write_text(
        "eval 39: {'eval/aime': 0.40}\n"
        "eval 79: {'eval/aime': 0.50}\n"
        "eval 99: {'eval/aime': 0.55}\n"
        "eval 119: {'eval/aime': 0.52}\n"
    )

    plan = retention_plan(tmp_path, log_file, "eval/aime", keep_steps=(79,))

    assert plan["best_step"] == 99
    assert plan["last_step"] == 119
    assert plan["protected_steps"] == [79]
    assert plan["retained_steps"] == [79, 99, 119]
    assert plan["removed_steps"] == [39, 59]


def test_retention_plan_can_keep_only_best_after_completed_run(tmp_path: Path) -> None:
    for step in (19, 39, 59):
        _checkpoint(tmp_path, step)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("59\n")
    log_file = tmp_path / "run.log"
    log_file.write_text(
        "eval 19: {'eval/aime': 0.10}\n"
        "eval 39: {'eval/aime': 0.30}\n"
        "eval 59: {'eval/aime': 0.20}\n"
    )

    plan = retention_plan(tmp_path, log_file, "eval/aime", keep_last=False)

    assert plan["best_step"] == 39
    assert plan["last_step"] == 59
    assert plan["keep_last"] is False
    assert plan["retained_steps"] == [39]
    assert plan["removed_steps"] == [19, 59]
