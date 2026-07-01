"""Detect common credentials, private paths, and excluded release artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TEXT_SUFFIXES = {".json", ".md", ".py", ".txt", ".yaml", ".yml", ".toml"}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".dcm",
    ".jpg",
    ".nii",
    ".png",
    ".pt",
    ".pth",
    ".xls",
    ".xlsx",
}
FORBIDDEN_NAMES = {
    "ref_lits_train.json",
    "ref_lits_val.json",
    "ref_lits_test.json",
    "ref_lidc_train.json",
    "ref_lidc_val.json",
    "ref_lidc_test.json",
}
CONTENT_PATTERNS = {
    "possible API key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "Windows absolute path": re.compile(r"\b[A-Za-z]:\\"),
    "Unix home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "email address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an anonymous release tree.")
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    return parser.parse_args()


def audit(root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if ".git" in path.parts:
            findings.append(f"Git metadata must not be bundled: {path}")
            continue
        if not path.is_file():
            continue

        if path.name.lower() in FORBIDDEN_NAMES:
            findings.append(f"Excluded split manifest found: {path}")
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(f"Excluded binary/data file found: {path}")
        if suffix not in TEXT_SUFFIXES:
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"Non-UTF-8 text file: {path}")
            continue
        for label, pattern in CONTENT_PATTERNS.items():
            if pattern.search(content):
                findings.append(f"{label} found in {path}")
    return findings


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    findings = audit(root)
    if findings:
        print("Release audit failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"Release audit passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

