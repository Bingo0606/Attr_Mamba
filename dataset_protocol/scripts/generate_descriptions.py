"""Generate public, attribute-anchored referring descriptions with GPT-5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PUBLIC_ATTRIBUTE_KEYS = {
    "anatomical_location",
    "relative_position",
    "size",
    "shape",
    "boundary",
    "texture",
    "density",
    "appearance",
}

PROHIBITED_INPUT_KEYS = {
    "bbox",
    "bounding_box",
    "centroid",
    "contour",
    "coordinates",
    "image",
    "image_path",
    "mask",
    "mask_path",
    "patient_id",
    "pixels",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate descriptions from non-pixel-level semantic anchors."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path(__file__).parents[1]
        / "prompts"
        / "representative_generation_prompt.txt",
    )
    parser.add_argument(
        "--model", default=os.environ.get("OPENAI_MODEL", "gpt-5")
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def validate_records(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("Input must be a JSON list.")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object.")
        if "file_id" not in record or "attributes" not in record:
            raise ValueError(f"Record {index} requires file_id and attributes.")

        file_id = str(record["file_id"])
        if file_id in seen_ids:
            raise ValueError(f"Duplicate file_id: {file_id}")
        seen_ids.add(file_id)

        attributes = record["attributes"]
        if not isinstance(attributes, dict) or not attributes:
            raise ValueError(f"Record {file_id} has no semantic attributes.")

        prohibited = PROHIBITED_INPUT_KEYS.intersection(attributes)
        if prohibited:
            names = ", ".join(sorted(prohibited))
            raise ValueError(f"Record {file_id} contains prohibited fields: {names}")

        unknown = set(attributes).difference(PUBLIC_ATTRIBUTE_KEYS)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Record {file_id} contains unsupported fields: {names}")

        clean_attributes: dict[str, str] = {}
        for key, value in attributes.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Record {file_id} has an invalid value for {key}.")
            clean_attributes[key] = value.strip()

        records.append({"file_id": record["file_id"], "attributes": clean_attributes})
    return records


def parse_model_json(text: str) -> dict[str, Any]:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model output did not contain a JSON object.")
        payload = json.loads(content[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object.")
    sentence = payload.get("description")
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("Model output is missing a non-empty description.")
    return {"description": " ".join(sentence.split())}


def generate_one(
    client: Any,
    model: str,
    prompt: str,
    attributes: dict[str, str],
    max_retries: int,
    retry_delay: float,
) -> str:
    model_input = json.dumps({"attributes": attributes}, ensure_ascii=False)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=prompt,
                input=model_input,
            )
            parsed = parse_model_json(response.output_text)
            return parsed["description"]
        except Exception as error:  # API and format errors are retried together.
            last_error = error
            if attempt + 1 < max_retries:
                time.sleep(retry_delay * (2**attempt))

    raise RuntimeError(f"Generation failed after {max_retries} attempts: {last_error}")


def main() -> int:
    args = parse_args()
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    records = validate_records(load_json(args.input))
    if args.limit is not None:
        records = records[: args.limit]

    if args.dry_run:
        print(f"Validated {len(records)} records; no API request was made.")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print("Install dependencies with: pip install -r requirements.txt", file=sys.stderr)
        return 2

    client = OpenAI()
    results = []
    for position, record in enumerate(records, start=1):
        sentence = generate_one(
            client=client,
            model=args.model,
            prompt=prompt,
            attributes=record["attributes"],
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        results.append({"file_id": record["file_id"], "sentence": sentence})
        print(f"Generated {position}/{len(records)}")

    write_json(args.output, results)
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata = {
        "requested_model": args.model,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "record_count": len(results),
    }
    write_json(metadata_path, metadata)
    print(f"Wrote {args.output} and {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

