"""Smoke-check prompt templates for required keys/placeholders.

This script is intentionally lightweight and dependency-minimal so it can run
early in CI and fail fast when prompt files are broken.
"""

from __future__ import annotations

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"


REQUIRED_FILES = (
    "evaluation.yaml",
    "validator.yaml",
    "flagging.yaml",
    "subfield_detection.yaml",
    "extraction_system_template.txt",
    "extraction_user_template.txt",
)

REQUIRED_YAML_KEYS = {
    "evaluation.yaml": ("system", "user_template"),
    "validator.yaml": (
        "system",
        "stage1_notes_prefix",
        "aggregated_data_prefix",
        "evaluation_feedback_header",
        "validation_tail",
    ),
    "flagging.yaml": (
        "intro",
        "run_stats_header",
        "manager_header",
        "review_header",
        "completeness_header",
        "task_header",
        "output_requirements",
    ),
    "subfield_detection.yaml": ("prompt_template",),
}

REQUIRED_PLACEHOLDERS = {
    "extraction_system_template.txt": ("{sub_field_line}",),
    "extraction_user_template.txt": ("{paper_text}",),
    "evaluation.yaml:user_template": ("{extracted_data}",),
    "subfield_detection.yaml:prompt_template": ("{paper_text}",),
}


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping/object at top level")
    return data


def main() -> int:
    errors: list[str] = []

    for filename in REQUIRED_FILES:
        path = PROMPTS_DIR / filename
        if not path.exists():
            errors.append(f"Missing required prompt file: {filename}")
            continue
        if not path.read_text(encoding="utf-8").strip():
            errors.append(f"Prompt file is empty: {filename}")

    for filename, keys in REQUIRED_YAML_KEYS.items():
        path = PROMPTS_DIR / filename
        if not path.exists():
            continue
        try:
            data = _read_yaml(path)
        except Exception as exc:
            errors.append(f"Invalid YAML in {filename}: {exc}")
            continue
        for key in keys:
            value = data.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{filename} missing non-empty string key: {key}")

    # Placeholder checks in plain text files
    for scoped_name, placeholders in REQUIRED_PLACEHOLDERS.items():
        if ":" in scoped_name:
            filename, key = scoped_name.split(":", 1)
            path = PROMPTS_DIR / filename
            if not path.exists():
                continue
            try:
                data = _read_yaml(path)
                content = data.get(key, "")
            except Exception as exc:
                errors.append(f"Cannot read {scoped_name}: {exc}")
                continue
        else:
            path = PROMPTS_DIR / scoped_name
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
        for ph in placeholders:
            if ph not in content:
                errors.append(f"{scoped_name} missing required placeholder: {ph}")

    if errors:
        print("Prompt validation FAILED:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Prompt validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

