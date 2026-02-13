#!/usr/bin/env python3
"""
Generate synthetic EE and REL output files from test data (entity_extraction and ground_truth).
Useful for testing the pipeline without running an LLM. With these, run_baseline.py will
evaluate against ground truth (synthetic REL = GT), so metrics indicate upper bound / sanity check.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TEST = os.path.join(ROOT, "data", "molground", "molground_test.json")
OUTPUT_DIR = os.path.join(ROOT, "outputs")


def load_test_data(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_data", default=DEFAULT_TEST)
    ap.add_argument("--out_dir", default=os.path.join(OUTPUT_DIR, "synthetic"))
    ap.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (default: all)")
    args = ap.parse_args()

    if not os.path.isfile(args.test_data):
        print(f"Error: test data not found: {args.test_data}")
        sys.exit(1)
    data = load_test_data(args.test_data)
    if args.max_samples:
        data = data[: args.max_samples]

    # EE: one output per sample = entity_extraction as JSON string (or empty list)
    ee_outputs = []
    for item in data:
        ee = item.get("entity_extraction") or []
        # Format as LLM would: list of {name, smiles}
        out_str = json.dumps(ee, indent=4)
        ee_outputs.append({"id": item["id"], "outputs": [out_str]})

    # REL: one output per sample = ground_truth formatted as in REL template (substructures + relationships)
    rel_outputs = []
    for item in data:
        gt = item.get("ground_truth") or {}
        subs = gt.get("substructures") or []
        rels = gt.get("relationships") or []
        # Single "perfect" output string (markdown-wrapped JSON like LLM)
        out_obj = {"substructures": subs, "relationships": rels}
        out_str = "```json\n" + json.dumps(out_obj, indent=4) + "\n```"
        # We need entities/relationships for compatibility; use from pipeline if we had run it
        rel_outputs.append({
            "id": item["id"],
            "entities": [],  # optional for selection
            "relationships": [],
            "outputs": [out_str],
        })

    os.makedirs(args.out_dir, exist_ok=True)
    ee_path = os.path.join(args.out_dir, "ee_outputs.json")
    rel_path = os.path.join(args.out_dir, "rel_outputs.json")
    with open(ee_path, "w") as f:
        json.dump(ee_outputs, f, indent=2)
    with open(rel_path, "w") as f:
        json.dump(rel_outputs, f, indent=2)
    print(f"Wrote {len(ee_outputs)} EE outputs to {ee_path}")
    print(f"Wrote {len(rel_outputs)} REL outputs to {rel_path}")
    print(f"Run: python run_baseline.py --ee_outputs {ee_path} --rel_outputs {rel_path} --output_dir {args.out_dir}/eval")


if __name__ == "__main__":
    main()
