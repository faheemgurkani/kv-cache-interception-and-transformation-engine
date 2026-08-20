"""Phase 31 bibliography audit — logical checks on reference.bib vs .tex."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_bibliography import (
    PHASE31_KEEP_KEYS,
    PHASE31_REMOVE_CANDIDATES,
    PHASE29_STAGING_KEYS,
    audit_bibliography,
    extract_bib_keys,
    extract_tex_cite_keys,
)

REPO = Path(__file__).resolve().parents[1]
BIB = REPO / "docs/research_paper_writeup/reference.bib"
TEX = REPO / "docs/research_paper_writeup/conference_101719.tex"
STAGING = REPO / "docs/literature/staging_entries.bib"


def test_reference_bib_contains_all_tex_cite_keys():
    bib_text = BIB.read_text(encoding="utf-8")
    tex_text = TEX.read_text(encoding="utf-8")
    bib_keys = extract_bib_keys(bib_text)
    tex_keys = extract_tex_cite_keys(tex_text)
    missing = sorted(tex_keys - bib_keys)
    assert missing == [], f"tex cites missing from reference.bib: {missing}"


def test_phase31_keep_list_present_in_reference_bib():
    bib_keys = extract_bib_keys(BIB.read_text(encoding="utf-8"))
    missing = sorted(PHASE31_KEEP_KEYS - bib_keys)
    assert missing == [], f"Phase 31 keep-list keys missing: {missing}"


def test_snapkv_venue_is_neurips_2024():
    report = audit_bibliography(BIB, TEX, STAGING)
    assert report.snapkv_neurips_ok is True
    assert "Neural Information Processing Systems" in report.snapkv_booktitle


def test_staging_entries_cover_phase29_priority_keys():
    staging_keys = extract_bib_keys(STAGING.read_text(encoding="utf-8"))
    missing = sorted(PHASE29_STAGING_KEYS - staging_keys)
    assert missing == [], f"staging_entries.bib missing Phase 29 keys: {missing}"


def test_audit_detects_anonymous_entries():
    report = audit_bibliography(BIB, TEX, STAGING)
    assert "expectedattn2026" in report.anonymous_entries
    assert "qjlcs2025" in report.anonymous_entries


def test_audit_flags_remove_candidates_still_cited():
    report = audit_bibliography(BIB, TEX, STAGING)
    cited_remove = set(report.remove_still_cited)
    assert PHASE31_REMOVE_CANDIDATES.issubset(cited_remove | report.tex_cite_keys)
    # All four remove candidates are currently cited — rewrite must address them
    assert cited_remove == sorted(PHASE31_REMOVE_CANDIDATES)


def test_full_audit_passes_on_repo_baseline():
    report = audit_bibliography(BIB, TEX, STAGING)
    assert report.ok is True
    assert report.errors == []
    assert len(report.staging_not_merged) == len(PHASE29_STAGING_KEYS)
