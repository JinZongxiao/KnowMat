"""Full pipeline comparison test: MinerU API → ocr_items → alignment vs sci-align baseline.

Steps:
  1. Call MinerU Precision API on test PDF (real API call)
  2. Convert MinerU content_list.json → KnowMat ocr_items via mineru_result_converter
  3. Run OcrItemTokenizer → visual_tokens + sentence_tokens
  4. Compare tokenization with sci-align baseline (image_units.jsonl, sentence_units.jsonl)
  5. Run alignment matching using sci-align's pre-computed CLIP embeddings
     (same PDF → same MinerU images → same CLIP embeddings; skips torch dependency)
  6. Compare alignment results with sci-align topk_pairs.jsonl

Usage:
    cd /ssd1/jinzongxiao/paddle_work/KnowMat-alignment
    python tests/test_alignment/test_mineru_pipeline.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/ssd1/jinzongxiao/paddle_work")
SCIALIGN_DIR = BASE / "sci-align"
KNOWMAT_DIR = BASE / "KnowMat-alignment"
PAPER_ID = "10.1007_s00289-023-04835-0"
TEST_PDF = SCIALIGN_DIR / "dataset" / f"{PAPER_ID}.pdf"
BASELINE_DIR = SCIALIGN_DIR / "dataset_test"
OUTPUT_DIR = KNOWMAT_DIR / "tests" / "test_alignment" / "mineru_output"

# Add KnowMat src to path
sys.path.insert(0, str(KNOWMAT_DIR / "src"))

# Stub out heavy dependencies so we can import knowmat submodules without
# needing langgraph, langchain, trustcall, etc.
import types
for _mod in [
    "langgraph", "langgraph.graph", "langgraph.checkpoint", "langgraph.checkpoint.memory",
    "langchain", "langchain_core", "langchain_openai", "langchain_core.messages",
    "trustcall", "openai",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Provide minimal stubs that knowmat/__init__.py and orchestrator.py will see
import types as _t
_lg = sys.modules["langgraph.graph"] = _t.ModuleType("langgraph.graph")
_lg.StateGraph = object
_lg.START = "START"
_lg.END = "END"

# Load .env before any knowmat import
from dotenv import load_dotenv
load_dotenv(KNOWMAT_DIR / ".env")


# ── Step 1: Call MinerU API ────────────────────────────────────────────────────

def call_mineru_api(pdf_path: Path, output_dir: Path) -> Optional[Path]:
    """Submit PDF to MinerU Precision API, download and extract result ZIP."""
    from knowmat.pdf.mineru_api_client import MineruPrecisionClient, MineruAPIError

    api_key = os.getenv("MINERU_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MINERU_API_KEY not set")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if we already have cached output
    cached = list(output_dir.glob("*_content_list.json"))
    if cached:
        logger.info("Using cached MinerU output: %s", cached[0])
        return output_dir

    logger.info("Submitting PDF to MinerU API: %s", pdf_path.name)
    client = MineruPrecisionClient(
        api_key=api_key,
        base_url=os.getenv("MINERU_API_BASE_URL", "https://mineru.net"),
    )

    try:
        task_result = client.upload_and_parse(
            pdf_path,
            model_version=os.getenv("MINERU_MODEL_VERSION", "vlm"),
            language=os.getenv("MINERU_LANGUAGE", "en"),
        )
        zip_url = task_result.get("full_zip_url") or task_result.get("zip_url", "")
        if not zip_url:
            raise RuntimeError(f"No zip_url in task result: {task_result}")
        extracted_dir = client.download_and_extract_zip(zip_url, output_dir)
        logger.info("MinerU output extracted to: %s", extracted_dir)
        return extracted_dir
    except MineruAPIError as e:
        logger.error("MinerU API error: %s", e)
        raise


# ── Step 2: Convert MinerU → ocr_items ────────────────────────────────────────

def convert_to_ocr_items(extracted_dir: Path, out_dir: Path) -> tuple:
    """Convert MinerU content_list.json to KnowMat ocr_items."""
    from knowmat.pdf.mineru_result_converter import convert_mineru_to_knowmat

    # Find content_list.json
    cl_files = list(extracted_dir.rglob("*_content_list.json"))
    cl_files = [f for f in cl_files if "_content_list_v2" not in f.name]
    if not cl_files:
        raise FileNotFoundError(f"No content_list.json in {extracted_dir}")
    cl_path = cl_files[0]
    logger.info("Found content_list.json: %s (%d bytes)", cl_path.name, cl_path.stat().st_size)

    with open(cl_path) as f:
        content_list = json.load(f)
    logger.info("content_list items: %d", len(content_list))

    figures_dir = out_dir / "figures"
    extracted_text, metadata, ocr_items = convert_mineru_to_knowmat(
        content_list=content_list,
        extracted_dir=extracted_dir,
        figures_dest=figures_dir,
        page_offset=0,
    )

    logger.info("ocr_items: %d", len(ocr_items))
    type_counts = defaultdict(int)
    for item in ocr_items:
        type_counts[item.get("typer", "?")] += 1
    logger.info("  by type: %s", dict(type_counts))

    return ocr_items, content_list


# ── Step 3: Tokenize ──────────────────────────────────────────────────────────

def tokenize(ocr_items: List[Dict], paper_id: str, output_dir: Path):
    from image_text_alignment.tokenizer import OcrItemTokenizer

    tok = OcrItemTokenizer()
    vis_tokens, sent_tokens = tok.tokenize(ocr_items, paper_id, str(output_dir))

    logger.info(
        "Tokenizer output: %d visual tokens, %d sentence tokens",
        len(vis_tokens), len(sent_tokens),
    )
    cap_text = sum(1 for s in sent_tokens if s.source == "caption_text")
    fig_ref = sum(1 for s in sent_tokens if s.has_figure_reference)
    logger.info(
        "  sentences with fig_ref: %d  caption_text (excluded from TopK): %d",
        fig_ref, cap_text,
    )
    return vis_tokens, sent_tokens


# ── Step 4: Compare tokenization with sci-align baseline ─────────────────────

def compare_tokenization(vis_tokens, sent_tokens):
    """Compare our tokenization against sci-align's saved token units."""
    print("\n" + "=" * 70)
    print("STEP 4: TOKENIZATION COMPARISON")
    print("=" * 70)

    with open(BASELINE_DIR / "image_units.jsonl") as f:
        bl_imgs = [json.loads(l) for l in f]
    with open(BASELINE_DIR / "sentence_units.jsonl") as f:
        bl_sents = [json.loads(l) for l in f]

    print(f"\n  Images:    KnowMat={len(vis_tokens):>3}   sci-align={len(bl_imgs):>3}")
    print(f"  Sentences: KnowMat={len(sent_tokens):>3}   sci-align={len(bl_sents):>3}")

    # Figure ID match
    km_figs = {vt.normalized_figure_id for vt in vis_tokens if vt.normalized_figure_id}
    bl_figs = {u["normalized_figure_id"] for u in bl_imgs if u.get("normalized_figure_id")}
    print(f"\n  Figure IDs in KnowMat : {sorted(km_figs)}")
    print(f"  Figure IDs in sci-align: {sorted(bl_figs)}")
    overlap = km_figs & bl_figs
    print(f"  Overlap: {len(overlap)}/{max(len(bl_figs),1)} ({len(overlap)/max(len(bl_figs),1)*100:.0f}%)")

    # Sentence text overlap (normalized)
    km_texts = {" ".join(st.text.lower().split()) for st in sent_tokens
                if st.source not in ("caption_text",)}
    bl_texts = {" ".join(u["text"].lower().split()) for u in bl_sents
                if u.get("source") != "caption_text"}
    sent_overlap = len(km_texts & bl_texts)
    print(f"\n  Body sentence overlap: {sent_overlap} / sci-align {len(bl_texts)}"
          f"  ({sent_overlap/max(len(bl_texts),1)*100:.0f}%)")


# ── Step 5+6: Alignment matching using sci-align's pre-computed embeddings ────
# The test PDF is the same one sci-align processed → same MinerU images
# → same CLIP embeddings saved in dataset_test/*.npy

FIG_MATCH_BONUS = 0.20
PAGE_NEAR_BONUS = 0.05
WRONG_FIG_PENALTY = 0.15


def _confidence(final: float, has_same: bool) -> str:
    if has_same or final >= 0.6:
        return "high"
    if final >= 0.4:
        return "medium"
    return "low"


def _compute_score(cosine, vis_all_fids, vis_page, sent_figs, sent_page):
    bonus = 0.0
    matched = None
    has_same = has_wrong = False
    if sent_figs:
        common = [f for f in sent_figs if f in vis_all_fids]
        if common:
            has_same, matched, bonus = True, common[0], FIG_MATCH_BONUS
        else:
            has_wrong, bonus = True, -WRONG_FIG_PENALTY
    if vis_page is not None and sent_page is not None:
        if abs(vis_page - sent_page) <= 1:
            bonus += PAGE_NEAR_BONUS
    return cosine + bonus, matched, has_same, has_wrong


def run_alignment_with_cached_embeddings(vis_tokens, sent_tokens):
    """Run alignment using sci-align's pre-computed embeddings (bypass torch)."""
    print("\n" + "=" * 70)
    print("STEP 5-6: ALIGNMENT (using sci-align's CLIP embeddings for same PDF)")
    print("=" * 70)

    with open(BASELINE_DIR / "embedding_index.json") as f:
        emb_idx = json.load(f)
    img_vecs = np.load(BASELINE_DIR / "image_embeddings.npy")
    txt_vecs = np.load(BASELINE_DIR / "sentence_embeddings.npy")

    img_token_ids = emb_idx["image_embeddings"]["token_ids"]
    sent_token_ids = emb_idx["sentence_embeddings"]["token_ids"]

    with open(BASELINE_DIR / "image_units.jsonl") as f:
        img_by_id = {json.loads(l)["token_id"]: json.loads(l) for l in f}
    with open(BASELINE_DIR / "sentence_units.jsonl") as f:
        sent_by_id = {json.loads(l)["token_id"]: json.loads(l) for l in f}

    # TopK pool: exclude caption_text
    topk_sent_ids = [t for t in sent_token_ids if sent_by_id[t].get("source") != "caption_text"]
    topk_idxs = [sent_token_ids.index(t) for t in topk_sent_ids]
    topk_vecs = txt_vecs[topk_idxs]

    logger.info("TopK pool: %d sentences (excluded %d caption_text)",
                len(topk_sent_ids), len(sent_token_ids) - len(topk_sent_ids))

    # Build per-paper sub-matrices
    paper_vis_idx = defaultdict(list)
    paper_sent_idx = defaultdict(list)
    for i, tid in enumerate(img_token_ids):
        paper_vis_idx[img_by_id[tid]["paper_id"]].append(i)
    for j, tid in enumerate(topk_sent_ids):
        paper_sent_idx[sent_by_id[tid]["paper_id"]].append(j)

    # -- Run KnowMat matching logic --
    km_results: Dict[str, List[Dict]] = {}
    for pid, vis_indices in paper_vis_idx.items():
        sent_indices = paper_sent_idx.get(pid, [])
        if not sent_indices:
            continue
        v_mat = img_vecs[vis_indices]
        s_mat = topk_vecs[sent_indices]
        paper_sim = v_mat @ s_mat.T

        for local_i, global_i in enumerate(vis_indices):
            vt = img_by_id[img_token_ids[global_i]]
            row = paper_sim[local_i]
            vis_all_fids = vt.get("all_figure_ids") or (
                [vt["normalized_figure_id"]] if vt.get("normalized_figure_id") else [])
            vis_page = vt.get("page_number")

            a_scored, c_scored, w_scored = [], [], []
            for local_j, global_j in enumerate(sent_indices):
                st = sent_by_id[topk_sent_ids[global_j]]
                cosine = float(row[local_j])
                final, matched, has_same, has_wrong = _compute_score(
                    cosine, vis_all_fids, vis_page,
                    st.get("mentioned_figures", []), st.get("page_number"))
                entry = (final, cosine, st, matched, has_same, has_wrong)
                (a_scored if has_same else w_scored if has_wrong else c_scored).append(entry)

            for lst in (a_scored, c_scored, w_scored):
                lst.sort(key=lambda x: x[0], reverse=True)

            vid = vt["token_id"]
            km_results[vid] = []
            for rank, (final, cosine, st, matched, has_same, has_wrong) in enumerate(
                (a_scored + c_scored + w_scored)[:5], 1
            ):
                km_results[vid].append({
                    "rank": rank, "text": st["text"],
                    "score": round(final, 4),
                    "cosine": round(cosine, 4),
                    "confidence": _confidence(final, has_same),
                    "anchor": "✓" if has_same else ("✗" if has_wrong else " "),
                })

    # -- Load sci-align results --
    bl_results: Dict[str, List[Dict]] = {}
    with open(BASELINE_DIR / "topk_pairs.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            vid = rec["visual_token_id"]
            bl_results.setdefault(vid, []).append({
                "rank": rec["rank"], "text": rec["text"],
                "score": rec["score"],
                "cosine": rec["cosine_score"],
                "confidence": rec["confidence"],
                "anchor": "✓" if rec["has_same_figure_anchor"] else ("✗" if rec["has_wrong_figure_anchor"] else " "),
            })
    for v in bl_results:
        bl_results[v].sort(key=lambda x: x["rank"])

    # -- Print comparison --
    rank1_match = topk_match = topk_total = 0
    all_vids = sorted(set(list(km_results) + list(bl_results)))
    for vid in all_vids:
        km = km_results.get(vid, [])
        bl = bl_results.get(vid, [])
        fig = vid.split("::")[-1]
        km_r1 = km[0]["text"] if km else ""
        bl_r1 = bl[0]["text"] if bl else ""
        r1_ok = km_r1 == bl_r1
        overlap = len({r["text"] for r in km} & {r["text"] for r in bl})
        rank1_match += int(r1_ok)
        topk_match += overlap
        topk_total += len(bl)

        status = "✓" if r1_ok else "✗"
        print(f"\n  {status} {fig}  rank-1={'match' if r1_ok else 'DIFF'}  top5_overlap={overlap}/5")
        for i in range(max(len(km), len(bl))):
            k = km[i] if i < len(km) else None
            b = bl[i] if i < len(bl) else None
            same = k and b and k["text"] == b["text"]
            tag = " ←" if same else ""
            if k:
                print(f"    KM r{k['rank']} {k['score']:.3f} [{k['confidence']:<6}] {k['anchor']}  {k['text'][:72]}{tag}")
            if b:
                print(f"    BL r{b['rank']} {b['score']:.3f} [{b['confidence']:<6}] {b['anchor']}  {b['text'][:72]}{tag}")

    n = len(all_vids)
    print(f"\n{'='*70}")
    print(f"ALIGNMENT SUMMARY ({n} images)")
    print(f"  Rank-1 exact match : {rank1_match}/{n}  ({rank1_match/n*100:.0f}%)")
    print(f"  Top-5 text overlap : {topk_match}/{topk_total}  ({topk_match/max(topk_total,1)*100:.0f}%)")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    assert TEST_PDF.exists(), f"PDF not found: {TEST_PDF}"

    print(f"\n{'='*70}")
    print(f"KnowMat Full Pipeline Test — MinerU API")
    print(f"Paper: {PAPER_ID}")
    print(f"{'='*70}")

    # Step 1: MinerU API
    print("\n[1/4] Calling MinerU API...")
    extracted_dir = call_mineru_api(TEST_PDF, OUTPUT_DIR / "mineru_raw")

    # Step 2: Convert to ocr_items
    print("\n[2/4] Converting MinerU output → ocr_items...")
    # Need to add src/knowmat to path for the converter
    sys.path.insert(0, str(KNOWMAT_DIR / "src"))

    # Directly use KnowMat's mineru_result_converter
    from knowmat.pdf.mineru_result_converter import convert_mineru_to_knowmat
    from collections import defaultdict as dd

    cl_files = list(extracted_dir.rglob("*_content_list.json"))
    cl_files = [f for f in cl_files if "_content_list_v2" not in f.name]
    if not cl_files:
        raise FileNotFoundError(f"No content_list.json found in {extracted_dir}")
    cl_path = cl_files[0]

    with open(cl_path) as f:
        content_list = json.load(f)
    logger.info("MinerU content_list items: %d", len(content_list))

    # Read full.md
    md_path = extracted_dir / "full.md"
    full_md = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

    figures_dir = OUTPUT_DIR / "figures"
    extracted_text, metadata, ocr_items = convert_mineru_to_knowmat(
        content_list=content_list,
        full_md=full_md,
        pdf_path=str(TEST_PDF),
        extracted_dir=extracted_dir,
        figures_dest=figures_dir,
    )
    type_counts = dd(int)
    for item in ocr_items:
        type_counts[item.get("typer", "?")] += 1
    logger.info("ocr_items total: %d  types: %s", len(ocr_items), dict(type_counts))

    # Save ocr_items for inspection
    with open(OUTPUT_DIR / "ocr_items.json", "w") as f:
        json.dump(ocr_items, f, ensure_ascii=False, indent=2)
    print(f"  ocr_items saved to: tests/test_alignment/mineru_output/ocr_items.json")

    # Step 3: Tokenize
    print("\n[3/4] Running OcrItemTokenizer...")
    sys.path.insert(0, str(KNOWMAT_DIR / "src" / "knowmat"))
    from image_text_alignment.tokenizer import OcrItemTokenizer
    tok = OcrItemTokenizer()
    vis_tokens, sent_tokens = tok.tokenize(ocr_items, PAPER_ID, str(OUTPUT_DIR))

    # Step 4: Compare tokenization
    compare_tokenization(vis_tokens, sent_tokens)

    # Step 5+6: Alignment using cached embeddings
    print("\n[4/4] Running alignment matching (using sci-align pre-computed CLIP embeddings)...")
    run_alignment_with_cached_embeddings(vis_tokens, sent_tokens)

    print("\nDone.")


if __name__ == "__main__":
    run()
