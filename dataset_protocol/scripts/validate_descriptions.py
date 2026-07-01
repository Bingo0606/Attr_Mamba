"""Run basic public checks on generated referring descriptions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROHIBITED_LANGUAGE = re.compile(
    r"\b(mask|ground[ -]?truth|annotation|bounding box|bbox|pixel|coordinate|contour)\b",
    re.IGNORECASE,
)
COORDINATE_PATTERN = re.compile(
    r"(?:\bx\s*=|\by\s*=|\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\))",
    re.IGNORECASE,
)
CONFLICTS = {
    "leftmost": "rightmost",
    "rightmost": "leftmost",
    "left": "right",
    "right": "left",
    "largest": "smallest",
    "smallest": "largest",
    "superior": "inferior",
    "inferior": "superior",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate public descriptions.")
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, required=True)
    return parser.parse_args()


def load_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return payload


def index_records(
    records: list[dict[str, Any]], required_field: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for position, record in enumerate(records):
        if not isinstance(record, dict) or "file_id" not in record:
            issues.append(f"Record {position} is missing file_id.")
            continue
        file_id = str(record["file_id"])
        if file_id in indexed:
            issues.append(f"Duplicate file_id: {file_id}")
        if required_field not in record:
            issues.append(f"Record {file_id} is missing {required_field}.")
        indexed[file_id] = record
    return indexed, issues


def flattened_anchor_text(attributes: dict[str, Any]) -> str:
    return " ".join(str(value).lower() for value in attributes.values())


def validate(
    attribute_records: list[dict[str, Any]],
    description_records: list[dict[str, Any]],
) -> list[str]:
    attributes, issues = index_records(attribute_records, "attributes")
    descriptions, description_issues = index_records(description_records, "sentence")
    issues.extend(description_issues)

    missing = sorted(set(attributes).difference(descriptions))
    extra = sorted(set(descriptions).difference(attributes))
    if missing:
        issues.append(f"Missing descriptions for: {', '.join(missing)}")
    if extra:
        issues.append(f"Descriptions without attributes: {', '.join(extra)}")

    for file_id in sorted(set(attributes).intersection(descriptions)):
        sentence = descriptions[file_id].get("sentence")
        if not isinstance(sentence, str) or not sentence.strip():
            issues.append(f"{file_id}: empty description.")
            continue

        if PROHIBITED_LANGUAGE.search(sentence):
            issues.append(f"{file_id}: annotation-related wording detected.")
        if COORDINATE_PATTERN.search(sentence):
            issues.append(f"{file_id}: possible coordinate leakage detected.")

        anchor_payload = attributes[file_id].get("attributes", {})
        if not isinstance(anchor_payload, dict):
            issues.append(f"{file_id}: attributes must be an object.")
            continue

        anchor_text = flattened_anchor_text(anchor_payload)
        sentence_lower = sentence.lower()
        for source, opposite in CONFLICTS.items():
            if re.search(rf"\b{source}\b", anchor_text) and not re.search(
                rf"\b{opposite}\b", anchor_text
            ):
                if re.search(rf"\b{opposite}\b", sentence_lower):
                    issues.append(
                        f"{file_id}: '{opposite}' conflicts with anchor '{source}'."
                    )
    return issues


def main() -> int:
    args = parse_args()
    issues = validate(load_list(args.attributes), load_list(args.descriptions))
    if issues:
        print("Validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

