# Public Filtering Rules

This document summarizes the public quality-control criteria used to audit
clinical-style referring descriptions generated from structured semantic
anchors. The purpose of these rules is to ensure that the released descriptions
remain structurally valid, semantically consistent, privacy-preserving, and
linguistically usable for Medical RIS research.

## 1. Structural validity

- Each released record must contain a valid local `file_id` and one non-empty
  English referring description.
- Outputs must be valid JSON and must not include generation commentary,
  Markdown formatting, or truncated text fragments.
- Duplicate IDs, missing descriptions, and malformed records are rejected.

## 2. Attribute consistency

- Each description must remain consistent with the supplied semantic anchors.
- Directional and relative expressions must not be reversed, including
  left/right, superior/inferior, and largest/smallest relationships.
- Unsupported diagnoses, anatomical structures, or lesion properties must not
  be introduced.
- Descriptions that are contradictory, target-ambiguous, or inconsistent with
  the supplied attributes are rejected or flagged for further review.

## 3. Target integrity and privacy protection

- Descriptions must not reveal coordinates, bounding boxes, contours, pixel
  counts, mask information, image paths, or patient identifiers.
- The released text must not mention annotations, ground truth, prompts,
  language models, or the generation procedure itself.
- Only non-pixel-level semantic anchors are exposed to the public generation
  interface.

## 4. Language quality and minimal normalization

- Descriptions should be concise, grammatical, clinically plausible, and
  sufficiently specific to identify the intended target.
- Unnatural wording, vague references, excessive repetition, and irrelevant
  phrases are rejected or minimally normalized without changing the intended
  semantics.
- Exact duplicate descriptions may be flagged when needed to reduce
  unnecessary linguistic redundancy in the public release.

## 5. Fairness and release safeguards

- The same public filtering criteria are applied across training, validation,
  and test splits.
- Segmentation predictions, evaluation scores, or test-set performance are not
  used to select, rewrite, or remove descriptions.
- Filtering does not alter image masks, target identities, or dataset splits.
- Released descriptions are intended as audited public text resources rather
  than performance-optimized annotations tuned to downstream results.
