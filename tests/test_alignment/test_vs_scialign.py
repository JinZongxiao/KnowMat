"""Compare KnowMat alignment matching logic against sci-align baseline.

Uses pre-computed embeddings from sci-align's dataset_test/ to bypass torch dependency.
Tests whether KnowMat's reranking/TopK logic produces identical results to sci-align.

Usage:
    cd /ssd1/jinzongxiao/paddle_work/KnowMat-alignment
    python tests/test_alignment/test_vs_scialign.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCIALIGN_DIR = Path("/ssd1/jinzongxiao/paddle_work/sci-align")
DATASET_TEST = SCIALIGN_DIR / "dataset_test"
PAPER_ID = "10.1007_s00289-023-04835-0"

KNOWMAT_SRC = Path(__file__).parent.parent.parent / "src" / "knowmat"
sys.path.insert(0, str(KNOWMAT_SRC))


# ── Load sci-align tokens + embeddings ────────────────────────────────────────

def load_sci_align_data():
    """Load pre-computed embeddings and token metadata from sci-align dataset_test/."""
    with open(DATASET_TEST / "embedding_index.json") as f:
        emb_idx = json.load(f)

    img_vecs = np.load(DATASET_TEST / "image_embeddings.npy")
    txt_vecs = np.load(DATASET_TEST / "sentence_embeddings.npy")

    with open(DATASET_TEST / "image_units.jsonl") as f:
        image_units = [json.loads(l) for l in f]
    with open(DATASET_TEST / "sentence_units.jsonl") as f:
        sentence_units = [json.loads(l) for l in f]

    img_token_ids = emb_idx["image_embeddings"]["token_ids"]
    sent_token_ids = emb_idx["sentence_embeddings"]["token_ids"]

    # Build lookup: token_id → index
    img_id_to_idx = {tid: i for i, tid in enumerate(img_token_ids)}
    sent_id_to_idx = {tid: i for i, tid in enumerate(sent_token_ids)}

    # Build lookup: token_id → unit dict
    img_by_id = {u["token_id"]: u for u in image_units}
    sent_by_id = {u["token_id"]: u for u in sentence_units}

    logger.info(
        "Loaded: %d images, %d sentences, img_emb=%s, txt_emb=%s",
        len(image_units), len(sentence_units), img_vecs.shape, txt_vecs.shape,
    )
    return img_vecs, txt_vecs, img_id_to_idx, sent_id_to_idx, img_by_id, sent_by_id


def load_baseline_topk() -> Dict[str, List[Dict]]:
    """Load sci-align topk_pairs.jsonl, grouped by visual_token_id."""
    groups: Dict[str, List[Dict]] = {}
    with open(DATASET_TEST / "topk_pairs.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            vid = rec["visual_token_id"]
            groups.setdefault(vid, []).append(rec)
    for vid in groups:
        groups[vid].sort(key=lambda r: r["rank"])
    return groups


# ── KnowMat matching logic (same as aligner.py, no torch needed) ──────────────

FIG_MATCH_BONUS = 0.20
PAGE_NEAR_BONUS = 0.05
WRONG_FIG_PENALTY = 0.15


def _confidence(final: float, has_same: bool) -> str:
    if has_same or final >= 0.6:
        return "high"
    if final >= 0.4:
        return "medium"
    return "low"


def _compute_final_score(cosine: float, vis_all_fids: List[str], vis_page: Optional[int],
                          sent_figs: List[str], sent_page: Optional[int]):
    bonus = 0.0
    matched_fig = None
    has_same = False
    has_wrong = False

    if sent_figs:
        common = [f for f in sent_figs if f in vis_all_fids]
        if common:
            has_same = True
            matched_fig = common[0]
            bonus += FIG_MATCH_BONUS
        else:
            has_wrong = True
            bonus -= WRONG_FIG_PENALTY

    if vis_page is not None and sent_page is not None:
        if abs(vis_page - sent_page) <= 1:
            bonus += PAGE_NEAR_BONUS

    return cosine + bonus, matched_fig, has_same, has_wrong


def run_knowmat_matching(
    img_vecs: np.ndarray,
    txt_vecs: np.ndarray,
    img_token_ids: List[str],
    sent_token_ids: List[str],
    img_by_id: Dict,
    sent_by_id: Dict,
    top_k: int = 5,
    min_score: float = 0.0,
):
    """Run KnowMat TopK matching on pre-computed embeddings."""
    # Filter out caption_text sentences from TopK pool
    topk_sent_ids = [
        tid for tid in sent_token_ids
        if sent_by_id[tid].get("source") != "caption_text"
    ]
    topk_sent_idxs = [sent_token_ids.index(tid) for tid in topk_sent_ids]
    topk_txt_vecs = txt_vecs[topk_sent_idxs]

    logger.info(
        "TopK pool: %d body sentences (excluded %d caption_text)",
        len(topk_sent_ids),
        len(sent_token_ids) - len(topk_sent_ids),
    )

    # Per-paper sub-matrices
    paper_vis_idx: Dict = defaultdict(list)
    paper_sent_idx: Dict = defaultdict(list)
    for i, tid in enumerate(img_token_ids):
        pid = sent_by_id.get(tid, {}).get("paper_id") or img_by_id[tid]["paper_id"]
        paper_vis_idx[pid].append(i)
    for j, tid in enumerate(topk_sent_ids):
        pid = sent_by_id[tid]["paper_id"]
        paper_sent_idx[pid].append(j)

    results: Dict[str, List[Dict]] = {}  # visual_token_id → ranked sentences

    for pid, vis_indices in paper_vis_idx.items():
        sent_indices = paper_sent_idx.get(pid, [])
        if not sent_indices:
            continue

        v_mat = img_vecs[vis_indices]                    # [n_vis, D]
        s_mat = topk_txt_vecs[sent_indices]              # [n_sent, D]
        paper_sim = v_mat @ s_mat.T                      # [n_vis, n_sent]

        for local_i, global_i in enumerate(vis_indices):
            vt = img_by_id[img_token_ids[global_i]]
            row = paper_sim[local_i]

            vis_all_fids = vt.get("all_figure_ids") or (
                [vt["normalized_figure_id"]] if vt.get("normalized_figure_id") else []
            )
            vis_page = vt.get("page_number")

            anchor_scored, clean_scored, wrong_scored = [], [], []
            for local_j, global_j in enumerate(sent_indices):
                st = sent_by_id[topk_sent_ids[global_j]]
                cosine = float(row[local_j])
                sent_figs = st.get("mentioned_figures", [])
                sent_page = st.get("page_number")

                final, matched_fig, has_same, has_wrong = _compute_final_score(
                    cosine, vis_all_fids, vis_page, sent_figs, sent_page
                )
                entry = (final, cosine, global_j, st, matched_fig, has_same, has_wrong)
                if has_same:
                    anchor_scored.append(entry)
                elif has_wrong:
                    wrong_scored.append(entry)
                else:
                    clean_scored.append(entry)

            anchor_scored.sort(key=lambda x: x[0], reverse=True)
            clean_scored.sort(key=lambda x: x[0], reverse=True)
            wrong_scored.sort(key=lambda x: x[0], reverse=True)
            selected = (anchor_scored + clean_scored + wrong_scored)[:top_k]

            vid = vt["token_id"]
            results[vid] = []
            for rank, (final, cosine, global_j, st, matched_fig, has_same, has_wrong) in enumerate(selected, 1):
                if final < min_score:
                    continue
                results[vid].append({
                    "rank": rank,
                    "text": st["text"],
                    "source": st["source"],
                    "cosine_score": round(cosine, 6),
                    "final_score": round(final, 6),
                    "score": round(final, 6),
                    "mentioned_figures": st.get("mentioned_figures", []),
                    "matched_figure_id": matched_fig,
                    "has_same_figure_anchor": has_same,
                    "has_wrong_figure_anchor": has_wrong,
                    "confidence": _confidence(final, has_same),
                })

    return results


# ── Comparison ────────────────────────────────────────────────────────────────

def compare(knowmat_results: Dict, baseline: Dict, top_k: int = 5):
    """Print side-by-side comparison and summary stats."""
    print("\n" + "=" * 80)
    print("COMPARISON: KnowMat  vs  sci-align baseline")
    print("  (same CLIP embeddings — only matching/reranking logic differs)")
    print("=" * 80)

    rank1_match = 0
    topk_overlap_total = 0
    topk_baseline_total = 0

    all_vids = sorted(set(list(knowmat_results.keys()) + list(baseline.keys())))

    for vid in all_vids:
        km_recs = knowmat_results.get(vid, [])
        bl_recs = baseline.get(vid, [])
        vt_info = vid.split("::")[-1]

        km_texts = [r["text"] for r in km_recs]
        bl_texts = [r["text"] for r in bl_recs]
        km_r1 = km_texts[0] if km_texts else ""
        bl_r1 = bl_texts[0] if bl_texts else ""
        r1_match = km_r1 == bl_r1
        overlap = len(set(km_texts) & set(bl_texts))

        rank1_match += int(r1_match)
        topk_overlap_total += overlap
        topk_baseline_total += len(bl_texts)

        print(f"\n{'─'*70}")
        print(f"Image: {vt_info}  Rank-1: {'✓' if r1_match else '✗'}  TopK overlap: {overlap}/{len(bl_texts)}")

        max_k = max(len(km_recs), len(bl_recs))
        for i in range(max_k):
            km = km_recs[i] if i < len(km_recs) else None
            bl = bl_recs[i] if i < len(bl_recs) else None
            same_text = km and bl and km["text"] == bl["text"]
            marker = " ←same" if same_text else ""
            if km:
                anchor = "✓" if km["has_same_figure_anchor"] else " "
                print(f"  KM r{km['rank']} {km['score']:.4f} [{km['confidence']:<6}] {anchor} {km['text'][:75]}{marker}")
            if bl:
                anchor = "✓" if bl["has_same_figure_anchor"] else " "
                print(f"  BL r{bl['rank']} {bl['score']:.4f} [{bl['confidence']:<6}] {anchor} {bl['text'][:75]}{marker}")
            if km or bl:
                print()

    n = len(all_vids)
    print(f"\n{'='*80}")
    print(f"SUMMARY ({n} images)")
    print(f"  Rank-1 exact match : {rank1_match}/{n}  ({rank1_match/n*100:.0f}%)")
    print(f"  Top-{top_k} text overlap: {topk_overlap_total}/{topk_baseline_total}  ({topk_overlap_total/max(topk_baseline_total,1)*100:.0f}%)")
    print("=" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_test():
    (img_vecs, txt_vecs, img_id_to_idx, sent_id_to_idx,
     img_by_id, sent_by_id) = load_sci_align_data()

    img_token_ids = list(img_id_to_idx.keys())  # ordered
    sent_token_ids = list(sent_id_to_idx.keys())

    baseline = load_baseline_topk()
    logger.info("Baseline: %d image groups", len(baseline))

    logger.info("Running KnowMat matching logic...")
    knowmat_results = run_knowmat_matching(
        img_vecs, txt_vecs, img_token_ids, sent_token_ids, img_by_id, sent_by_id
    )
    logger.info("Got results for %d images", len(knowmat_results))

    compare(knowmat_results, baseline)


def test_alignment_matches_baseline():
    run_test()


if __name__ == "__main__":
    run_test()
