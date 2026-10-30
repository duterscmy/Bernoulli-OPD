#!/usr/bin/env python3
"""Download and normalize every dataset used by the released experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset


# Revisions are pinned to the versions used to prepare the release data.
DATASETS = {
    "dapo-math-17k": {
        "repo_id": "zhuzilin/dapo-math-17k",
        "revision": "2e65612930298bde4c5d58fd97b3f23a483aaff9",
        "split": "train",
        "output": "dapo-math-17k/dapo-math-17k.jsonl",
        "expected_rows": 17_398,
    },
    "aime-2024": {
        "repo_id": "zhuzilin/aime-2024",
        "revision": "1c625e328db94ec7ef7ff169016b097c468d60b9",
        "split": "train",
        "output": "aime-2024/aime-2024.jsonl",
        "expected_rows": 30,
    },
    "aime-2025": {
        "repo_id": "zhuzilin/aime-2025",
        "revision": "3d2ffb246285920641112d03a9c23218db4de9b7",
        "split": "train",
        "output": "aime-2025/aime-2025.jsonl",
        "expected_rows": 30,
    },
    "hmmt-feb-2025": {
        "repo_id": "MathArena/hmmt_feb_2025",
        "revision": "6fdc4277120810ff75aa22d2d5489b91f7a262a1",
        "split": "train",
        "output": "hmmt-feb-2025/hmmt-feb-2025.jsonl",
        "expected_rows": 30,
    },
    "amc-2023": {
        "repo_id": "math-ai/amc23",
        "revision": "80815d37005feb82cd7f8fbc6901d5d3eff43057",
        "split": "test",
        "output": "amc-2023/amc-2023.jsonl",
        "expected_rows": 40,
    },
    "math-500": {
        "repo_id": "HuggingFaceH4/MATH-500",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "split": "test",
        "output": "math-500/math-500.jsonl",
        "expected_rows": 500,
    },
}


def first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError(f"none of {keys!r} is present; available keys: {sorted(row)}")


def normalize_prompt(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, list) and value:
        messages = []
        for message in value:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise TypeError(f"invalid chat message: {message!r}")
            messages.append({"role": str(message["role"]), "content": str(message["content"])})
        return messages
    raise TypeError(f"unsupported prompt value: {value!r}")


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = first_present(row, "prompt", "problem", "question")
    label = first_present(row, "label", "answer", "solution")
    return {"prompt": normalize_prompt(prompt), "label": str(label)}


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    for name, spec in DATASETS.items():
        print(f"Downloading {name} from {spec['repo_id']}@{spec['revision']} ...", flush=True)
        dataset = load_dataset(
            spec["repo_id"],
            split=spec["split"],
            revision=spec["revision"],
        )
        rows = [normalize_row(dict(row)) for row in dataset]
        if len(rows) != spec["expected_rows"]:
            raise RuntimeError(f"{name}: expected {spec['expected_rows']} rows, found {len(rows)}")

        output = args.data_root / spec["output"]
        atomic_write_jsonl(output, rows)
        atomic_write_json(
            output.parent / "source.json",
            {
                "dataset": spec["repo_id"],
                "revision": spec["revision"],
                "split": spec["split"],
                "rows": len(rows),
                "output": output.name,
            },
        )
        print(f"  wrote {len(rows)} rows to {output}", flush=True)

    print("All datasets are ready.", flush=True)


if __name__ == "__main__":
    main()
