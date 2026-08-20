#!/usr/bin/env python3
"""Phase 31 bibliography audit — reference.bib vs conference_101719.tex."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIB = REPO_ROOT / "docs/research_paper_writeup/reference.bib"
DEFAULT_TEX = REPO_ROOT / "docs/research_paper_writeup/conference_101719.tex"
DEFAULT_STAGING = REPO_ROOT / "docs/literature/staging_entries.bib"

# Phase 31 — definitely keep (must resolve in reference.bib)
PHASE31_KEEP_KEYS: frozenset[str] = frozenset(
    {
        "zhang2023h2o",
        "liu2023scissorhands",
        "xiao2024streamingllm",
        "li2024snapkv",
        "cai2024pyramidkv",
        "liu2024minicache",
        "zandieh2025qjl",
        "chang2025palu",
        "su2025outlier",
        "su2025kvsink",
        "tao2025asymkv",
        "yang2025xquant",
        "qwen3",
        "olmo2",
        "zandieh2026turboquant",
        "wang2026hqekv",
        "chen2026pitfalls",
        "kvbench2026serving",
        "wikitext",
        "rocketkv",
    }
)

# Phase 31 — remove unless specifically needed (flag if still cited in .tex)
PHASE31_REMOVE_CANDIDATES: frozenset[str] = frozenset(
    {
        "costoptgqa2025",
        "qjlcs2025",
        "expectedattn2026",
        "yuan2026shortrl",
    }
)

# Phase 29 staging keys expected before full rewrite (informational)
PHASE29_STAGING_KEYS: frozenset[str] = frozenset(
    {
        "oaken2025",
        "scope2025",
        "turboattention2025",
        "rkv2025",
        "ojakv2026",
        "hybridkv2026",
        "cacheblend2025",
        "kvcachewild2025",
    }
)

CITE_PATTERN = re.compile(r"\\cite\{([^}]+)\}")
ENTRY_PATTERN = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.IGNORECASE)
ANONYMOUS_MARKERS = ("anonymous",)


@dataclass
class BibliographyAudit:
    bib_path: Path
    tex_path: Path
    staging_path: Path | None = None
    bib_keys: set[str] = field(default_factory=set)
    staging_keys: set[str] = field(default_factory=set)
    tex_cite_keys: set[str] = field(default_factory=set)
    missing_from_bib: list[str] = field(default_factory=list)
    keep_missing: list[str] = field(default_factory=list)
    remove_still_cited: list[str] = field(default_factory=list)
    anonymous_entries: list[str] = field(default_factory=list)
    snapkv_neurips_ok: bool = False
    snapkv_booktitle: str = ""
    staging_not_merged: list[str] = field(default_factory=list)
    verify_notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "bib_path": str(self.bib_path),
            "tex_path": str(self.tex_path),
            "bib_key_count": len(self.bib_keys),
            "tex_cite_key_count": len(self.tex_cite_keys),
            "missing_from_bib": self.missing_from_bib,
            "keep_missing": self.keep_missing,
            "remove_still_cited": self.remove_still_cited,
            "anonymous_entries": self.anonymous_entries,
            "snapkv_neurips_ok": self.snapkv_neurips_ok,
            "snapkv_booktitle": self.snapkv_booktitle,
            "staging_not_merged": self.staging_not_merged,
            "verify_notes": self.verify_notes,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_bib_keys(bib_text: str) -> set[str]:
    return set(ENTRY_PATTERN.findall(bib_text))


def extract_tex_cite_keys(tex_text: str) -> set[str]:
    keys: set[str] = set()
    for match in CITE_PATTERN.finditer(tex_text):
        for part in match.group(1).split(","):
            key = part.strip()
            if key:
                keys.add(key)
    return keys


def extract_entry_block(bib_text: str, key: str) -> str:
    pattern = re.compile(
        rf"@\w+\s*\{{{re.escape(key)}\s*,.*?(?=\n@\w+\s*\{{|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(bib_text)
    return match.group(0) if match else ""


def audit_bibliography(
    bib_path: Path = DEFAULT_BIB,
    tex_path: Path = DEFAULT_TEX,
    staging_path: Path | None = DEFAULT_STAGING,
) -> BibliographyAudit:
    result = BibliographyAudit(bib_path=bib_path, tex_path=tex_path, staging_path=staging_path)

    if not bib_path.is_file():
        result.errors.append(f"missing bibliography: {bib_path}")
        return result
    if not tex_path.is_file():
        result.errors.append(f"missing tex file: {tex_path}")
        return result

    bib_text = _read_text(bib_path)
    tex_text = _read_text(tex_path)
    result.bib_keys = extract_bib_keys(bib_text)
    result.tex_cite_keys = extract_tex_cite_keys(tex_text)

    result.missing_from_bib = sorted(k for k in result.tex_cite_keys if k not in result.bib_keys)
    if result.missing_from_bib:
        result.errors.append(
            f"tex cites keys absent from reference.bib: {', '.join(result.missing_from_bib)}"
        )

    result.keep_missing = sorted(k for k in PHASE31_KEEP_KEYS if k not in result.bib_keys)
    if result.keep_missing:
        result.errors.append(f"Phase 31 keep-list keys missing from bib: {', '.join(result.keep_missing)}")

    result.remove_still_cited = sorted(
        k for k in PHASE31_REMOVE_CANDIDATES if k in result.tex_cite_keys
    )
    for key in result.remove_still_cited:
        result.warnings.append(
            f"Phase 31 remove candidate still cited in tex: {key} — drop cite at rewrite or keep entry"
        )

    for key in result.bib_keys:
        block = extract_entry_block(bib_text, key)
        lower = block.lower()
        if any(marker in lower for marker in ANONYMOUS_MARKERS):
            result.anonymous_entries.append(key)

    for key in sorted(result.anonymous_entries):
        result.warnings.append(f"anonymous or unverified author entry: {key}")

    snap_block = extract_entry_block(bib_text, "li2024snapkv")
    booktitle_match = re.search(r"booktitle\s*=\s*\{([^}]+)\}", snap_block, re.IGNORECASE)
    result.snapkv_booktitle = booktitle_match.group(1) if booktitle_match else ""
    result.snapkv_neurips_ok = (
        "neural information processing systems" in result.snapkv_booktitle.lower()
        or "neurips" in result.snapkv_booktitle.lower()
    )
    if not result.snapkv_neurips_ok:
        result.errors.append(
            f"SnapKV venue should be NeurIPS 2024; got booktitle={result.snapkv_booktitle!r}"
        )

    if staging_path and staging_path.is_file():
        staging_text = _read_text(staging_path)
        result.staging_keys = extract_bib_keys(staging_text)
        result.staging_not_merged = sorted(k for k in PHASE29_STAGING_KEYS if k not in result.bib_keys)
        for key in result.staging_not_merged:
            result.verify_notes.append(f"Phase 29 staging key not yet in reference.bib: {key}")

    # Phase 31 verify/fix notes (manual follow-up)
    if "feng2024adakv" in result.bib_keys:
        adakv = extract_entry_block(bib_text, "feng2024adakv")
        if "arxiv" in adakv.lower() and "2025" in adakv:
            result.verify_notes.append("Ada-KV: NeurIPS 2025 with arXiv note — verify proceedings metadata")
    if "su2025kvsink" in result.bib_keys:
        kvsink = extract_entry_block(bib_text, "su2025kvsink")
        if "arxiv" in kvsink.lower():
            result.verify_notes.append("KVSink: arXiv only — verify venue if promoting to inproceedings")
    if "rocketkv" in result.bib_keys:
        rocket = extract_entry_block(bib_text, "rocketkv")
        if "icml" not in rocket.lower():
            result.verify_notes.append("RocketKV: verify ICML 2025 proceedings metadata")
    if "compresskv2026" in result.bib_keys:
        result.verify_notes.append("CompressKV: arXiv 2026 — verify if published version available")
    if "jin2025mha2gqa" in result.bib_keys:
        result.verify_notes.append("MHA→GQA: jin2025mha2gqa in bib; costoptgqa2025 is separate remove candidate")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit reference.bib against conference_101719.tex (Phase 31)")
    parser.add_argument("--bib", type=Path, default=DEFAULT_BIB)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--json", type=Path, help="Write JSON report")
    args = parser.parse_args()

    report = audit_bibliography(args.bib, args.tex, args.staging)
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    print(f"Bibliography audit: {'OK' if report.ok else 'FAILED'}")
    print(f"  bib keys: {len(report.bib_keys)}  tex cite keys: {len(report.tex_cite_keys)}")
    print(f"  SnapKV NeurIPS OK: {report.snapkv_neurips_ok}")
    if report.errors:
        print("Errors:")
        for err in report.errors:
            print(f"  - {err}")
    if report.warnings:
        print("Warnings:")
        for warn in report.warnings:
            print(f"  - {warn}")
    if report.verify_notes:
        print("Verify notes:")
        for note in report.verify_notes:
            print(f"  - {note}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
