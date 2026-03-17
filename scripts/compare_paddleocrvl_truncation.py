#!/usr/bin/env python3
"""Run OCR comparison across PaddleOCR-VL versions and trim modes.

This script executes four OCR-only runs of KnowMat:
- PaddleOCR-VL 1.5 + trim references
- PaddleOCR-VL 1.5 + keep full text
- PaddleOCR-VL 1.0 + trim references
- PaddleOCR-VL 1.0 + keep full text

Outputs are isolated under a compare directory with four subfolders.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class VariantConfig:
    folder_name: str
    version: str
    trim_references: bool


VARIANTS: tuple[VariantConfig, ...] = (
    VariantConfig(folder_name="paddleocrvl1_5_trim", version="1.5", trim_references=True),
    VariantConfig(folder_name="paddleocrvl1_5_no_trim", version="1.5", trim_references=False),
    VariantConfig(folder_name="paddleocrvl1_0_trim", version="1.0", trim_references=True),
    VariantConfig(folder_name="paddleocrvl1_0_no_trim", version="1.0", trim_references=False),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PaddleOCR-VL 1.5/1.0 with trim/non-trim by running "
            "KnowMat OCR-only pipeline into four isolated folders."
        )
    )
    parser.add_argument(
        "--input-folder",
        default="data/raw",
        help="Source folder containing PDF files (default: data/raw).",
    )
    parser.add_argument(
        "--compare-dir",
        default="compare",
        help="Output root folder for four comparison subfolders (default: compare).",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to run 'python -m knowmat' (default: current interpreter).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional paper filter (stem or full filename), forwarded to KnowMat.",
    )
    return parser.parse_args(argv)


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_set(items: Iterable[str] | None) -> set[str]:
    if not items:
        return set()
    return {item.strip() for item in items if item and item.strip()}


def _collect_target_pdfs(input_folder: Path, only_filters: Sequence[str] | None) -> list[Path]:
    pdfs = sorted(
        [path for path in input_folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"],
        key=lambda path: path.name.lower(),
    )
    if not only_filters:
        return pdfs

    requested = _as_set(only_filters)
    filtered = [path for path in pdfs if path.stem in requested or path.name in requested]
    return filtered


def _prepare_variant_input(variant_input_dir: Path, selected_pdfs: Sequence[Path]) -> None:
    if variant_input_dir.exists():
        shutil.rmtree(variant_input_dir)
    variant_input_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in selected_pdfs:
        shutil.copy2(pdf_path, variant_input_dir / pdf_path.name)


def _bool_env(flag: bool) -> str:
    return "true" if flag else "false"


def _resolve_model_dir_for_version(repo_root: Path, version: str, inherited_env: dict[str, str]) -> Path:
    if version == "1.0":
        model_dir = inherited_env.get("PADDLEOCRVL1_0_MODEL_DIR")
        if model_dir:
            return Path(model_dir).expanduser().resolve()
        return (repo_root / "models" / "paddleocrvl1_0").resolve()

    model_dir = inherited_env.get("PADDLEOCRVL_MODEL_DIR")
    if model_dir:
        return Path(model_dir).expanduser().resolve()
    return (repo_root / "models" / "paddleocrvl1_5").resolve()


def _run_one_variant(
    *,
    repo_root: Path,
    python_executable: str,
    variant: VariantConfig,
    variant_input_dir: Path,
    only_filters: Sequence[str] | None,
) -> int:
    env = dict(os.environ)
    env["PADDLEOCRVL_VERSION"] = variant.version
    env["KNOWMAT2_TRIM_REFERENCES_SECTION"] = _bool_env(variant.trim_references)
    env["PADDLEOCRVL_MODEL_DIR"] = str(_resolve_model_dir_for_version(repo_root, variant.version, env))

    cmd: list[str] = [
        python_executable,
        "-m",
        "knowmat",
        "--input-folder",
        str(variant_input_dir),
        "--ocr-only",
        "--force-rerun",
    ]
    if only_filters:
        cmd.extend(["--only", *only_filters])

    print("=" * 72)
    print(f"[RUN] {variant.folder_name}")
    print(f"  - PADDLEOCRVL_VERSION={variant.version}")
    print(f"  - KNOWMAT2_TRIM_REFERENCES_SECTION={_bool_env(variant.trim_references)}")
    print(f"  - PADDLEOCRVL_MODEL_DIR={env['PADDLEOCRVL_MODEL_DIR']}")
    print(f"  - INPUT={variant_input_dir}")
    print("=" * 72)

    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        check=False,
    )
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = _resolve_repo_root()

    input_folder = (repo_root / args.input_folder).resolve()
    compare_dir = (repo_root / args.compare_dir).resolve()

    if not input_folder.exists() or not input_folder.is_dir():
        print(f"[ERROR] input-folder does not exist or is not a directory: {input_folder}")
        return 2

    selected_pdfs = _collect_target_pdfs(input_folder, args.only)
    if not selected_pdfs:
        if args.only:
            print(f"[ERROR] no PDF matched --only in: {input_folder}")
            print(f"        --only values: {args.only}")
        else:
            print(f"[ERROR] no PDF files found in: {input_folder}")
        return 3

    compare_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Repo root: {repo_root}")
    print(f"[INFO] Source PDFs: {input_folder}")
    print(f"[INFO] Compare root: {compare_dir}")
    print(f"[INFO] Selected PDF count: {len(selected_pdfs)}")

    failed_variants: list[str] = []
    for variant in VARIANTS:
        variant_input_dir = compare_dir / variant.folder_name
        _prepare_variant_input(variant_input_dir, selected_pdfs)

        return_code = _run_one_variant(
            repo_root=repo_root,
            python_executable=args.python_executable,
            variant=variant,
            variant_input_dir=variant_input_dir,
            only_filters=args.only,
        )
        if return_code != 0:
            failed_variants.append(variant.folder_name)
            print(f"[FAIL] {variant.folder_name} (exit_code={return_code})")
        else:
            print(f"[OK]   {variant.folder_name}")

    print("\n" + "=" * 72)
    if failed_variants:
        print("[DONE] finished with failures.")
        print(f"[FAILED_VARIANTS] {', '.join(failed_variants)}")
        return 1
    print("[DONE] all four variants finished successfully.")
    print("Generated folders:")
    for variant in VARIANTS:
        print(f"- {compare_dir / variant.folder_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
