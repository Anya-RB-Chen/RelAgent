"""
Evaluation: compare predicted graph (substructures + relationships) to ground truth.
"""
import json


def _normalize_substructure(s):
    """Normalize substructure dict to comparable form: (name, sorted atom_indices)."""
    name = (s.get("name") or "").strip().lower()
    indices = s.get("atom_indices") or s.get("indices") or []
    return (name, tuple(sorted(indices)))


def _normalize_relationship(r):
    """Normalize relationship to (id_a, id_b, type) with canonical order."""
    a = (r.get("substructure_a") or "").strip()
    b = (r.get("substructure_b") or "").strip()
    rel = (r.get("relationship") or "").strip()
    if a > b:
        a, b = b, a
    return (a, b, rel)


def _build_id_to_sub(pred_subs):
    """Map instance_id -> (name, tuple(indices)) for prediction."""
    return {s.get("instance_id", ""): _normalize_substructure(s) for s in (pred_subs or [])}


def _substructure_match(gt_subs, pred_subs):
    """
    Match GT substructures to predicted. GT and pred are lists of {name, instance_id, atom_indices}.
    Returns (tp, fp, fn) for substructure instances: match by (name, set(atom_indices)).
    """
    gt_set = {_normalize_substructure(s) for s in (gt_subs or [])}
    pred_set = {_normalize_substructure(s) for s in (pred_subs or [])}
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    return tp, fp, fn


def _relationship_match(gt_rels, pred_rels, gt_subs, pred_subs):
    """
    Match relationships. We need to align instance_ids between GT and pred (they may differ).
    Strategy: normalize by (name, atom_indices) so we can compare relationship type between
    (name_a, indices_a)-(name_b, indices_b). So normalize each rel to (norm_a, norm_b, type)
    where norm = (name, tuple(sorted(indices))) from the corresponding substructure.
    """
    def subs_by_id(subs):
        return {s.get("instance_id"): _normalize_substructure(s) for s in (subs or [])}

    gt_by_id = subs_by_id(gt_subs)
    pred_by_id = subs_by_id(pred_subs)
    # Build (norm_a, norm_b) -> type for GT
    gt_norm = set()
    for r in gt_rels or []:
        a_id = r.get("substructure_a")
        b_id = r.get("substructure_b")
        if a_id not in gt_by_id or b_id not in gt_by_id:
            continue
        na, nb = gt_by_id[a_id], gt_by_id[b_id]
        if na > nb:
            na, nb = nb, na
        gt_norm.add((na, nb, (r.get("relationship") or "").strip()))
    pred_norm = set()
    for r in pred_rels or []:
        a_id = r.get("substructure_a")
        b_id = r.get("substructure_b")
        if a_id not in pred_by_id or b_id not in pred_by_id:
            continue
        na, nb = pred_by_id[a_id], pred_by_id[b_id]
        if na > nb:
            na, nb = nb, na
        pred_norm.add((na, nb, (r.get("relationship") or "").strip()))
    # Match by (norm_a, norm_b); type must match
    tp = len(gt_norm & pred_norm)
    fp = len(pred_norm - gt_norm)
    fn = len(gt_norm - pred_norm)
    return tp, fp, fn


def evaluate_one(gt: dict, pred: dict) -> dict:
    """
    Compare one prediction to one ground truth. gt and pred have 'substructures' and 'relationships'.
    Returns dict with substructure_metrics and relationship_metrics (precision, recall, f1).
    """
    gt_subs = gt.get("substructures") or []
    gt_rels = gt.get("relationships") or []
    pred_subs = pred.get("substructures") if pred else []
    pred_rels = pred.get("relationships") if pred else []

    stp, sfp, sfn = _substructure_match(gt_subs, pred_subs)
    rtp, rfp, rfn = _relationship_match(gt_rels, pred_rels, gt_subs, pred_subs)

    def metrics(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    return {
        "substructure": metrics(stp, sfp, sfn),
        "relationship": metrics(rtp, rfp, rfn),
    }


def evaluate(test_data: list, predictions: list) -> dict:
    """
    test_data: list of {id, ground_truth}; predictions: list of {id, graph}.
    Align by id and compute per-sample metrics, then aggregate.
    """
    gt_by_id = {t["id"]: t.get("ground_truth") or {} for t in test_data}
    pred_by_id = {p["id"]: p.get("graph") or {} for p in predictions}

    all_ids = sorted(set(gt_by_id.keys()) & set(pred_by_id.keys()))
    results = []
    for sample_id in all_ids:
        gt = gt_by_id[sample_id]
        pred = pred_by_id.get(sample_id)
        if not gt:
            continue
        r = evaluate_one(gt, pred)
        r["id"] = sample_id
        results.append(r)

    # Aggregate
    def avg(key, subkey):
        vals = [r[key][subkey] for r in results if r.get(key)]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "num_samples": len(results),
        "substructure": {
            "precision": avg("substructure", "precision"),
            "recall": avg("substructure", "recall"),
            "f1": avg("substructure", "f1"),
        },
        "relationship": {
            "precision": avg("relationship", "precision"),
            "recall": avg("relationship", "recall"),
            "f1": avg("relationship", "f1"),
        },
        "per_sample": results,
    }
