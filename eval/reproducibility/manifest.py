"""Phase 14 reproducibility manifest extraction (RESEARCH_REDESIGN_PLAN §Phase 14)."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any


def collect_git_sha(*, cwd: Path | str | None = None) -> str | None:
    """Best-effort commit pin: env stamp (Modal) then ``git rev-parse HEAD`` (local)."""
    for key in ("KV_GIT_SHA", "GIT_SHA", "GIT_COMMIT"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=None if cwd is None else str(cwd),
        )
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return None
    sha = proc.stdout.strip()
    return sha or None

def _git(args: list[str], cwd: Path | str | None) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            cwd=None if cwd is None else str(cwd),
        )
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return None
    return proc.stdout


def collect_git_state(*, cwd: Path | str | None = None) -> dict[str, Any]:
    """Commit + working-tree state, so a recorded SHA cannot silently hide local edits.

    ``git_sha`` alone is HEAD at submission time; Modal mounts the *working tree*, so
    uncommitted fixes ran under the wrong SHA in the Gemma3 bundle (audit P1).
    Returns ``{"sha", "dirty", "dirty_file_count", "diff_sha256"}`` (``dirty``/``diff_sha256``
    are ``None`` when git is unavailable, e.g. inside a container without ``.git``; the
    submitting process stamps them through ``KV_GIT_DIRTY`` / ``KV_GIT_DIFF_SHA256``).
    """
    sha = collect_git_sha(cwd=cwd)
    env_dirty = os.environ.get("KV_GIT_DIRTY", "").strip()
    env_diff = os.environ.get("KV_GIT_DIFF_SHA256", "").strip()
    if env_dirty:
        return {
            "sha": sha,
            "dirty": env_dirty == "1",
            "dirty_file_count": None,
            "diff_sha256": env_diff or None,
        }
    status = _git(["status", "--porcelain"], cwd)
    if status is None:
        return {"sha": sha, "dirty": None, "dirty_file_count": None, "diff_sha256": None}
    lines = [line for line in status.splitlines() if line.strip()]
    if not lines:
        return {"sha": sha, "dirty": False, "dirty_file_count": 0, "diff_sha256": None}
    digest = hashlib.sha256()
    digest.update((_git(["diff", "HEAD"], cwd) or "").encode())
    for line in sorted(lines):
        digest.update(line.encode())
        if line.startswith("??"):  # untracked files are invisible to `git diff`
            path = Path(cwd or ".") / line[3:].strip()
            if path.is_file():
                digest.update(path.read_bytes())
    return {
        "sha": sha,
        "dirty": True,
        "dirty_file_count": len(lines),
        "diff_sha256": digest.hexdigest(),
    }


# Canonical checklist from Phase 14 YAML block.
PHASE14_FIELDS: tuple[str, ...] = (
    "model",
    "context_length",
    "generation_length",
    "hardware",
    "batch_size",
    "compression_method",
    "compression_ratio",
    "calibration",
    "dataset",
    "seed",
    "precision",
)


def extract_phase14_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the standardized Phase 14 configuration dict from ``EvaluationResult.to_dict()``."""
    controlled = payload.get("controlled_conditions") or {}
    fixed = controlled.get("fixed") or {}
    variable = controlled.get("variable") or {}
    budget = variable.get("compression_budget") or {}

    fidelity = payload.get("fidelity") or {}
    memory = fidelity.get("memory") or {}
    cost = payload.get("cost") or {}
    compression_cost = cost.get("compression") or {}
    offline = cost.get("offline") or {}

    hardware = payload.get("hardware") or fixed.get("hardware")

    return {
        "model": fixed.get("model") or payload.get("model"),
        "context_length": fixed.get("context_length", payload.get("context_length")),
        "generation_length": fixed.get("generation_length"),
        "hardware": hardware,
        "batch_size": fixed.get("batch_size"),
        "compression_method": variable.get("compressor") or budget.get("compression_method"),
        "compression_ratio": {
            "measured": memory.get("compression_ratio"),
            "theoretical": compression_cost.get("theoretical_compression_ratio"),
            "actual": compression_cost.get("actual_compression_ratio"),
        },
        "calibration": offline,
        "dataset": fixed.get("dataset"),
        # Two independent seeds: the synthetic-task generators (fixed, recorded under
        # controlled_conditions.fixed.seed) and the compressor's own randomness (None for
        # deterministic methods such as identity / SnapKV / RocketKV / Palu).
        "seed": {"evaluation": fixed.get("seed"), "compressor": budget.get("seed")},
        "precision": fixed.get("precision"),
        "compression_budget": budget,
    }


def validate_phase14_manifest(
    payload: dict[str, Any],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> list[str]:
    """Return human-readable errors if the payload violates Phase 14 invariants."""
    errors: list[str] = []
    manifest = extract_phase14_manifest(payload)

    for field in PHASE14_FIELDS:
        value = manifest.get(field)
        if value is None or (field == "seed" and value.get("evaluation") is None):
            errors.append(f"missing Phase 14 field: {field}")

    controlled = payload.get("controlled_conditions") or {}
    fixed = controlled.get("fixed") or {}
    variable = controlled.get("variable") or {}
    budget = variable.get("compression_budget") or {}

    compressor = payload.get("compressor")
    if variable.get("compressor") != compressor:
        errors.append("controlled_conditions.variable.compressor != payload.compressor")
    if budget.get("compression_method") != compressor:
        errors.append("compression_budget.compression_method != payload.compressor")
    if manifest["compression_method"] != compressor:
        errors.append("manifest.compression_method != payload.compressor")

    ctx = payload.get("context_length")
    if fixed.get("context_length") != ctx:
        errors.append("fixed.context_length != payload.context_length")

    fidelity = payload.get("fidelity") or {}
    memory = fidelity.get("memory") or {}
    cost = payload.get("cost") or {}
    compression_cost = cost.get("compression") or {}

    measured = memory.get("compression_ratio")
    uncompressed = memory.get("uncompressed_bytes")
    compressed = memory.get("compressed_bytes")
    if measured is not None and uncompressed is not None and compressed is not None and compressed > 0:
        expected_ratio = uncompressed / compressed
        if abs(measured - expected_ratio) > max(atol, rtol * abs(expected_ratio)):
            errors.append(
                f"memory.compression_ratio {measured} != uncompressed/compressed {expected_ratio}"
            )

    actual_cost_ratio = compression_cost.get("actual_compression_ratio")
    if measured is not None and actual_cost_ratio is not None and measured != actual_cost_ratio:
        errors.append("cost.compression.actual_compression_ratio != fidelity.memory.compression_ratio")

    reduction = compression_cost.get("actual_memory_reduction_bytes")
    if (
        reduction is not None
        and uncompressed is not None
        and compressed is not None
        and reduction != uncompressed - compressed
    ):
        errors.append("cost.compression.actual_memory_reduction_bytes != uncompressed - compressed")

    theoretical = compression_cost.get("theoretical_compression_ratio")
    if theoretical is not None and theoretical <= 0:
        errors.append("theoretical_compression_ratio must be positive")

    return errors
