"""OCR engine creation and execution helpers."""

from __future__ import annotations

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Tuple


def default_model_dir() -> Path:
    """Return the default local model directory in this project."""
    repo_root = Path(__file__).resolve().parents[3]
    model_dir = os.getenv("PADDLEOCRVL_MODEL_DIR", str(repo_root / "models" / "paddleocrvl1_5"))
    return Path(model_dir).expanduser().resolve()


def _prepare_ocr_home(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLEOCR_HOME"] = str(model_dir)
    if "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK" not in os.environ:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def create_ocr_engine(model_dir: Path) -> Tuple[Any, str]:
    """Create a PaddleOCR-VL engine, or fallback to PaddleOCR."""
    _prepare_ocr_home(model_dir)

    try:
        from paddleocr import PaddleOCRVL  # type: ignore

        for kwargs in [{}]:
            try:
                return PaddleOCRVL(**kwargs), "paddleocrvl"
            except TypeError:
                continue
            except Exception:
                break
    except ImportError:
        pass

    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError as exc:
        raise ImportError("PaddleOCR is not installed. Install with: pip install knowmat[ocr]") from exc

    init_candidates = (
        {"use_angle_cls": True, "lang": "en"},
        {"lang": "en"},
        {},
    )
    last_error: Exception | None = None
    for kwargs in init_candidates:
        try:
            return PaddleOCR(**kwargs), "paddleocr"
        except TypeError as exc:
            last_error = exc

    if last_error:
        raise RuntimeError(f"Failed to initialize PaddleOCR engine: {last_error}") from last_error
    raise RuntimeError("Failed to initialize PaddleOCR engine.")


def _run_ocr(engine: Any, image_path: Path) -> Any:
    img = str(image_path)
    if hasattr(engine, "predict"):
        try:
            return engine.predict(img)
        except TypeError:
            return engine.predict(input=img)
        except (RuntimeError, ValueError, OSError):
            pass
    if hasattr(engine, "ocr"):
        try:
            return engine.ocr(img, cls=True)
        except TypeError:
            return engine.ocr(img)
    if callable(engine):
        return engine(img)
    raise RuntimeError("OCR engine does not expose a known inference method.")


def supports_batch_predict(engine: Any) -> bool:
    if hasattr(engine, "predict_batch"):
        return True
    if hasattr(engine, "predict"):
        try:
            import inspect

            sig = inspect.signature(engine.predict)
            for param in sig.parameters.values():
                if param.annotation in ("list", "List", List):
                    return True
        except (ValueError, TypeError):
            pass
    return False


def run_ocr_batch(engine: Any, image_paths: List[Path], batch_size: int) -> List[Any]:
    results: List[Any] = [None] * len(image_paths)
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        str_paths = [str(p) for p in chunk]
        try:
            batch_raw = engine.predict_batch(str_paths)
        except (TypeError, AttributeError):
            batch_raw = [_run_ocr(engine, p) for p in chunk]
        for j, raw in enumerate(batch_raw):
            results[start + j] = raw
    return results


def run_ocr_parallel(engine: Any, image_paths: List[Path], max_workers: int) -> List[Any]:
    if max_workers <= 1:
        return [_run_ocr(engine, p) for p in image_paths]

    results: List[Any] = [None] * len(image_paths)
    sem = threading.Semaphore(max_workers)

    def _worker(idx: int, path: Path) -> None:
        with sem:
            results[idx] = _run_ocr(engine, path)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_worker, i, p) for i, p in enumerate(image_paths)]
        for fut in futures:
            fut.result()
    return results


def collect_text(obj: Any, out: List[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        text = obj.strip()
        if text:
            out.append(text)
        return
    if isinstance(obj, dict):
        for key in ("text", "rec_text", "ocr_text", "content", "transcription"):
            val = obj.get(key)
            if isinstance(val, str):
                val = val.strip()
                if val:
                    out.append(val)
        for val in obj.values():
            collect_text(val, out)
        return
    if isinstance(obj, (list, tuple)):
        if len(obj) == 2 and isinstance(obj[1], (list, tuple)) and len(obj[1]) > 0 and isinstance(obj[1][0], str):
            text = obj[1][0].strip()
            if text:
                out.append(text)
            return
        for item in obj:
            collect_text(item, out)


def normalize_lines(lines: List[str]) -> List[str]:
    normalized: List[str] = []
    prev = ""
    for line in lines:
        text = re.sub(r"\s+", " ", line).strip()
        if not text or text == prev:
            continue
        normalized.append(text)
        prev = text
    return normalized

