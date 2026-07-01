# Ref-LITS and Ref-LIDC Dataset Construction Protocol

This directory contains the public resources accompanying the construction and quality control of the Ref-LITS and Ref-LIDC clinical-style referring descriptions.

## Contents

```text
dataset_protocol/
|-- descriptions/                         # Audited public description files
|-- docs/FILTERING_RULES.md                # Quality-control criteria
|-- examples/                              # Example anchors and descriptions
|-- prompts/representative_generation_prompt.txt
|-- schemas/                               # Public input/output JSON schemas
|-- scripts/generate_descriptions.py       # Representative generation interface
|-- scripts/validate_descriptions.py       # Structural and semantic checks
|-- tools/audit_release.py                 # Package-level release audit
`-- requirements.txt
```

Released descriptions:

| File | Records |
| --- | ---: |
| `descriptions/Ref-LITS_descriptions.json` | 14,883 |
| `descriptions/Ref-LIDC_descriptions.json` | 8,721 |

## Protocol Summary

1. Fix CT-volume or patient/scan-level splits before description generation.
2. Derive non-pixel-level semantic anchors for each target, including anatomical location, relative position, size, shape, boundary, and texture-related cues.
3. Generate a concise clinical-style description through the representative GPT-5 prompt interface.
4. Validate structure, protected attributes, privacy constraints, and linguistic quality.
5. Audit the final descriptions without changing file identifiers, targets, masks, or dataset splits.

Medical images, pixel-wise masks, exact coordinates, contours, image paths, and patient identifiers are not exposed to the language model. The same public validation criteria are applied independently to all splits.

## Quick Start

Install the protocol dependencies:

```bash
python -m pip install -r requirements.txt
```

Validate the included examples:

```bash
python scripts/validate_descriptions.py \
  --attributes examples/example_attributes.json \
  --descriptions examples/example_descriptions.json
```

Audit the complete protocol directory:

```bash
python tools/audit_release.py .
```

See [`docs/FILTERING_RULES.md`](docs/FILTERING_RULES.md) for the released filtering and consistency criteria.
