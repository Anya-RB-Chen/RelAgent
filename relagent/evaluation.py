"""
Evaluation: compare predicted graph (substructures + relationships) to ground truth.
Metrics: CNER, Grounding (Exact Match), Grounding (IoU), Relationships.
"""
from typing import List, Dict, Tuple, Any


def _normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def _normalize_substructure(s: dict) -> Tuple[str, Tuple[int, ...]]:
    """Normalize substructure dict to comparable form: (name, sorted atom_indices)."""
    name = _normalize_name(s.get("name") or "")
    indices = s.get("atom_indices") or s.get("indices") or []
    return (name, tuple(sorted(indices)))


def _normalize_relationship(r: dict) -> Tuple[str, str, str]:
    """Normalize relationship to (id_a, id_b, type) with canonical order."""
    a = (r.get("substructure_a") or "").strip()
    b = (r.get("substructure_b") or "").strip()
    rel = _normalize_name(r.get("relationship") or "")
    if a > b:
        a, b = b, a
    return (a, b, rel)


def _compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_iou(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0


# ---------- CNER: Concept Name Entity Recognition (entity names only) ----------
def _evaluate_cner(gt_subs: List[dict], pred_subs: List[dict]) -> Tuple[float, float, float]:
    """Precision, Recall, F1 on unique normalized entity names."""
    gt_names = {_normalize_name(s.get("name") or "") for s in (gt_subs or []) if s.get("name")}
    pred_names = {_normalize_name(s.get("name") or "") for s in (pred_subs or []) if s.get("name")}
    tp = len(gt_names & pred_names)
    precision = tp / len(pred_names) if pred_names else 0.0
    recall = tp / len(gt_names) if gt_names else 0.0
    f1 = _compute_f1(precision, recall)
    return precision, recall, f1


# ---------- Grounding: Exact Match (name + atom_indices) ----------
def _evaluate_grounding_exact(gt_subs: List[dict], pred_subs: List[dict]) -> Tuple[float, float, float]:
    """Precision, Recall, F1 on (name, sorted atom_indices) match."""
    gt_set = {_normalize_substructure(s) for s in (gt_subs or [])}
    pred_set = {_normalize_substructure(s) for s in (pred_subs or [])}
    tp = len(gt_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = _compute_f1(precision, recall)
    return precision, recall, f1


# ---------- Grounding: IoU (intersection over union of atom sets) ----------
def _evaluate_grounding_iou(gt_subs: List[dict], pred_subs: List[dict]) -> Dict[str, Any]:
    """
    For each predicted substructure, best IoU with a GT of the same name; then average.
    Coverage = (# of GT instances matched) / (# GT instances).
    """
    gt_subs = gt_subs or []
    pred_subs = pred_subs or []
    gt_by_name = {}
    for sub in gt_subs:
        name = _normalize_name(sub.get("name") or "")
        if not name:
            continue
        indices = sub.get("atom_indices") or sub.get("indices") or []
        gt_by_name.setdefault(name, []).append(set(indices))

    iou_scores = []
    matched_gt = set()  # (name, idx) for each matched GT instance

    for pred in pred_subs:
        name = _normalize_name(pred.get("name") or "")
        pred_atoms = set(pred.get("atom_indices") or pred.get("indices") or [])

        if name not in gt_by_name:
            iou_scores.append(0.0)
            continue

        best_iou = 0.0
        best_idx = -1
        for idx, gt_atoms in enumerate(gt_by_name[name]):
            iou = _compute_iou(pred_atoms, gt_atoms)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0:
            matched_gt.add((name, best_idx))
        iou_scores.append(best_iou)

    avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    total_gt = len(gt_subs)
    coverage = len(matched_gt) / total_gt if total_gt else 0.0

    return {
        "Average_IoU": round(avg_iou, 4),
        "Coverage": round(coverage, 4),
        "Matched": len(matched_gt),
        "Total_GT": total_gt,
    }


# ---------- Relationships (normalized by substructure identity: name + indices) ----------
def _subs_by_id(subs: List[dict]) -> dict:
    return {s.get("instance_id", ""): _normalize_substructure(s) for s in (subs or [])}


def _evaluate_relationships(
    gt_rels: List[dict], pred_rels: List[dict], gt_subs: List[dict], pred_subs: List[dict]
) -> Tuple[float, float, float]:
    """Precision, Recall, F1 on relationships; align by (name, atom_indices) so instance_ids can differ."""
    gt_by_id = _subs_by_id(gt_subs)
    pred_by_id = _subs_by_id(pred_subs)

    gt_norm = set()
    for r in gt_rels or []:
        a_id = r.get("substructure_a")
        b_id = r.get("substructure_b")
        if a_id not in gt_by_id or b_id not in gt_by_id:
            continue
        na, nb = gt_by_id[a_id], gt_by_id[b_id]
        if na > nb:
            na, nb = nb, na
        gt_norm.add((na, nb, _normalize_name(r.get("relationship") or "")))

    pred_norm = set()
    for r in pred_rels or []:
        a_id = r.get("substructure_a")
        b_id = r.get("substructure_b")
        if a_id not in pred_by_id or b_id not in pred_by_id:
            continue
        na, nb = pred_by_id[a_id], pred_by_id[b_id]
        if na > nb:
            na, nb = nb, na
        pred_norm.add((na, nb, _normalize_name(r.get("relationship") or "")))

    tp = len(gt_norm & pred_norm)
    precision = tp / len(pred_norm) if pred_norm else 0.0
    recall = tp / len(gt_norm) if gt_norm else 0.0
    f1 = _compute_f1(precision, recall)
    return precision, recall, f1


# ---------- Single-sample evaluation (SRG = Substructure Relationship Graph) ----------
def evaluate_srg(gt_dict: dict, pred_dict: dict) -> Dict[str, Any]:
    """
    Evaluate one prediction against one ground truth (both with 'substructures' and 'relationships').
    Returns dict with CNER, Grounding (Exact Match), Grounding (IoU), Relationships.
    """
    gt_subs = gt_dict.get("substructures") or []
    gt_rels = gt_dict.get("relationships") or []
    pred_subs = pred_dict.get("substructures") or [] if pred_dict else []
    pred_rels = pred_dict.get("relationships") or [] if pred_dict else []

    cner_p, cner_r, cner_f1 = _evaluate_cner(gt_subs, pred_subs)
    gnd_p, gnd_r, gnd_f1 = _evaluate_grounding_exact(gt_subs, pred_subs)
    gnd_iou = _evaluate_grounding_iou(gt_subs, pred_subs)
    rel_p, rel_r, rel_f1 = _evaluate_relationships(gt_rels, pred_rels, gt_subs, pred_subs)

    return {
        "CNER": {
            "Precision": round(cner_p, 4),
            "Recall": round(cner_r, 4),
            "F1": round(cner_f1, 4),
        },
        "Grounding (Exact Match)": {
            "Precision": round(gnd_p, 4),
            "Recall": round(gnd_r, 4),
            "F1": round(gnd_f1, 4),
        },
        "Grounding (IoU)": {
            "Average_IoU": gnd_iou["Average_IoU"],
            "Coverage": gnd_iou["Coverage"],
            "Matched": gnd_iou["Matched"],
            "Total_GT": gnd_iou["Total_GT"],
        },
        "Relationships": {
            "Precision": round(rel_p, 4),
            "Recall": round(rel_r, 4),
            "F1": round(rel_f1, 4),
        },
    }


def evaluate_one(gt: dict, pred: dict) -> dict:
    """
    Same as evaluate_srg; alias for backward compatibility.
    Adds "id" only when caller sets it.
    """
    return evaluate_srg(gt, pred)


def evaluate_srg_from_text(gt_dict: dict, pred_text: str) -> Dict[str, Any]:
    """
    Evaluate when prediction is raw LLM output (string). Parses JSON from text then runs evaluate_srg.
    Returns same structure as evaluate_srg; returns {} if parsing fails.
    """
    from relagent.util import extract_json_from_text
    pred_json = extract_json_from_text(pred_text)
    if pred_json is None:
        return {}
    return evaluate_srg(gt_dict, pred_json)


# ---------- Batch evaluation and aggregation ----------
def evaluate(test_data: List[dict], predictions: List[dict]) -> Dict[str, Any]:
    """
    test_data: list of {id, ground_truth}; predictions: list of {id, graph}.
    Align by id, compute per-sample metrics, then aggregate to the same format as the old pipeline.
    """
    gt_by_id = {t["id"]: t.get("ground_truth") or {} for t in test_data}
    pred_by_id = {p["id"]: p.get("graph") or {} for p in predictions}

    all_ids = sorted(set(gt_by_id.keys()) & set(pred_by_id.keys()))
    per_sample = []

    for sample_id in all_ids:
        gt = gt_by_id[sample_id]
        pred = pred_by_id.get(sample_id)
        if not gt:
            continue
        try:
            r = evaluate_srg(gt, pred)
            r["id"] = sample_id
            per_sample.append(r)
        except Exception:
            continue

    n = len(per_sample)
    if n == 0:
        return {
            "CNER": {"Precision": 0.0, "Recall": 0.0, "F1": 0.0},
            "Grounding (Exact Match)": {"Precision": 0.0, "Recall": 0.0, "F1": 0.0},
            "Grounding (IoU)": {"Average_IoU": 0.0, "Coverage": 0.0, "Matched": 0, "Total_GT": 0},
            "Relationships": {"Precision": 0.0, "Recall": 0.0, "F1": 0.0},
            "Total Evaluated": 0,
            "per_sample": [],
        }

    # Aggregate: average over samples for P/R/F1 and Average_IoU; coverage = total matched / total GT
    total_cner_p = sum(s["CNER"]["Precision"] for s in per_sample)
    total_cner_r = sum(s["CNER"]["Recall"] for s in per_sample)
    total_cner_f1 = sum(s["CNER"]["F1"] for s in per_sample)

    total_gnd_p = sum(s["Grounding (Exact Match)"]["Precision"] for s in per_sample)
    total_gnd_r = sum(s["Grounding (Exact Match)"]["Recall"] for s in per_sample)
    total_gnd_f1 = sum(s["Grounding (Exact Match)"]["F1"] for s in per_sample)

    total_iou = sum(s["Grounding (IoU)"]["Average_IoU"] for s in per_sample)
    total_matched = sum(s["Grounding (IoU)"]["Matched"] for s in per_sample)
    total_gt = sum(s["Grounding (IoU)"]["Total_GT"] for s in per_sample)

    total_rel_p = sum(s["Relationships"]["Precision"] for s in per_sample)
    total_rel_r = sum(s["Relationships"]["Recall"] for s in per_sample)
    total_rel_f1 = sum(s["Relationships"]["F1"] for s in per_sample)

    return {
        "CNER": {
            "Precision": round(total_cner_p / n, 4),
            "Recall": round(total_cner_r / n, 4),
            "F1": round(total_cner_f1 / n, 4),
        },
        "Grounding (Exact Match)": {
            "Precision": round(total_gnd_p / n, 4),
            "Recall": round(total_gnd_r / n, 4),
            "F1": round(total_gnd_f1 / n, 4),
        },
        "Grounding (IoU)": {
            "Average_IoU": round(total_iou / n, 4),
            "Coverage": round(total_matched / total_gt, 4) if total_gt > 0 else 0.0,
            "Matched": total_matched,
            "Total_GT": total_gt,
        },
        "Relationships": {
            "Precision": round(total_rel_p / n, 4),
            "Recall": round(total_rel_r / n, 4),
            "F1": round(total_rel_f1 / n, 4),
        },
        "Total Evaluated": n,
        "per_sample": per_sample,
    }
