"""Audit P1 / P2: provenance records dirty state; manifest always has a seed."""

import subprocess

from eval.reproducibility.manifest import (
    collect_git_state,
    extract_phase14_manifest,
    validate_phase14_manifest,
)


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_git_state_clean_then_dirty_then_untracked(tmp_path, monkeypatch):
    for key in ("KV_GIT_SHA", "GIT_SHA", "GIT_COMMIT", "KV_GIT_DIRTY", "KV_GIT_DIFF_SHA256"):
        monkeypatch.delenv(key, raising=False)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")

    clean = collect_git_state(cwd=tmp_path)
    assert clean["dirty"] is False and clean["diff_sha256"] is None and clean["sha"]

    (tmp_path / "a.py").write_text("x = 2\n")
    dirty = collect_git_state(cwd=tmp_path)
    assert dirty["dirty"] is True and dirty["dirty_file_count"] == 1 and dirty["diff_sha256"]

    (tmp_path / "new.py").write_text("y = 1\n")  # untracked files must change the hash too
    with_untracked = collect_git_state(cwd=tmp_path)
    assert with_untracked["diff_sha256"] != dirty["diff_sha256"]


def test_env_stamp_wins_inside_containers_without_git(monkeypatch, tmp_path):
    monkeypatch.setenv("KV_GIT_SHA", "abc123")
    monkeypatch.setenv("KV_GIT_DIRTY", "1")
    monkeypatch.setenv("KV_GIT_DIFF_SHA256", "deadbeef")
    state = collect_git_state(cwd=tmp_path)
    assert state == {"sha": "abc123", "dirty": True, "dirty_file_count": None, "diff_sha256": "deadbeef"}


def test_manifest_seed_is_present_even_for_deterministic_compressors():
    payload = {
        "compressor": "snapkv",
        "context_length": 128,
        "controlled_conditions": {
            "fixed": {"seed": 0, "context_length": 128},
            "variable": {"compressor": "snapkv", "compression_budget": {"compression_method": "snapkv"}},
        },
    }
    manifest = extract_phase14_manifest(payload)
    assert manifest["seed"] == {"evaluation": 0, "compressor": None}
    assert not any("seed" in e for e in validate_phase14_manifest(payload))


def test_manifest_flags_missing_evaluation_seed():
    payload = {"compressor": "identity", "context_length": 128, "controlled_conditions": {"fixed": {}, "variable": {}}}
    assert any("seed" in e for e in validate_phase14_manifest(payload))
