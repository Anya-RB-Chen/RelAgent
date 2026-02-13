#!/usr/bin/env python3
"""
Run the RelAgent baseline: load EE and REL outputs, select predictions, and evaluate;
or run end-to-end on a single (SMILES, caption) with vLLM.

Usage:
  # End-to-end: one molecule, output full graph (requires vLLM + GPU):
  python run_baseline.py --run_e2e --smiles "CC1=CC=CC=C1" --caption "Benzene ring" --model meta-llama/Llama-3.2-1B-Instruct

  # Evaluate with precomputed EE and REL outputs (e.g. from batch LLM runs):
  python run_baseline.py --test_data data/molground/molground_test.json \\
    --ee_outputs path/to/ee_outputs.json --rel_outputs path/to/rel_outputs.json \\
    --output_dir outputs/run1 --policy majority_voting

  # Optionally use MolGenie ontology for relationship context (if you have the pickles):
  python run_baseline.py ... --molgenie_dir data/molgenie

  # Generate REL prompts from EE outputs (without running LLM):
  python run_baseline.py --test_data data/molground/molground_test.json \\
    --ee_outputs path/to/ee_outputs.json --build_prompts_only --prompts_out outputs/prompts.json
"""
import argparse
import json
import os
import sys

# Ensure relagent is importable from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relagent.config import DEFAULT_TEST_PATH, MOLGROUND_DIR, MOLONTO_CACHE_DIR, OUTPUT_DIR
from relagent.pipeline import (
    load_test_data,
    load_ee_outputs,
    load_rel_outputs,
    ee_to_rel_prompts,
    select_outputs,
    parse_predictions,
    run_end_to_end,
)
from relagent.evaluation import evaluate
from relagent.run_recorder import (
    start_run,
    finish_run,
    fail_run,
    record_output,
    _serializable_config,
    ARTIFACTS,
    RUN_SUMMARY_FILENAME,
)


def main():
    ap = argparse.ArgumentParser(description="RelAgent baseline: evaluate or run end-to-end with vLLM")
    ap.add_argument("--run_e2e", action="store_true", help="Run end-to-end: input SMILES + caption, output graph (uses vLLM)")
    ap.add_argument("--smiles", default=None, help="SMILES string (required for --run_e2e)")
    ap.add_argument("--caption", default=None, help="Caption string (required for --run_e2e)")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct", help="vLLM model for both EE and REL (used if --ee_model/--rel_model not set)")
    ap.add_argument("--ee_model", default=None, help="Separate vLLM model for Entity Extraction (overrides --model for EE)")
    ap.add_argument("--rel_model", default=None, help="Separate vLLM model for Relationship Reasoning (overrides --model for REL)")
    ap.add_argument("--n_ee_samples", type=int, default=1, help="Number of EE samples for --run_e2e (Best-of-N)")
    ap.add_argument("--e2e_policy", default="first", choices=["first", "majority_voting", "random"],
                    help="REL selection policy for --run_e2e")
    ap.add_argument("--e2e_out", default=None, help="Write output graph JSON to this path (--run_e2e)")
    ap.add_argument("--test_data", default=DEFAULT_TEST_PATH, help="Path to molground test JSON")
    ap.add_argument("--ee_outputs", default=None, help="Path to EE outputs JSON (id, outputs list)")
    ap.add_argument("--rel_outputs", default=None, help="Path to REL outputs JSON (id, entities, relationships, outputs)")
    ap.add_argument("--output_dir", default=OUTPUT_DIR, help="Directory to write selected outputs and metrics")
    ap.add_argument("--policy", default="first", choices=["first", "majority_voting", "random"],
                    help="Selection policy for REL outputs")
    ap.add_argument("--molgenie_dir", default=None,
                    help="Optional: dir with molgenie_all_mols.pkl and molgenie_all_nodes_dict.pkl")
    ap.add_argument("--build_prompts_only", action="store_true",
                    help="Only build REL prompts from EE outputs and save; do not evaluate")
    ap.add_argument("--prompts_out", default=None, help="When --build_prompts_only, save prompts to this path")
    args = ap.parse_args()

    if args.run_e2e:
        if not args.smiles or not args.caption:
            print("Error: --run_e2e requires --smiles and --caption")
            sys.exit(1)
        out_dir = os.path.dirname(os.path.abspath(args.e2e_out)) if args.e2e_out else args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        start_run(out_dir, "e2e", _serializable_config(args))
        try:
            from relagent.llm import RelAgentLLM
            # Create LLM instance(s): use separate models if provided, else use single model
            llm = RelAgentLLM(
                model=args.model if not (args.ee_model or args.rel_model) else None,
                ee_model=args.ee_model,
                rel_model=args.rel_model,
            )
            graph = run_end_to_end(
                args.smiles,
                args.caption,
                llm=llm,
                n_ee_samples=args.n_ee_samples,
                rel_selection_policy=args.e2e_policy,
                molgenie_dir=args.molgenie_dir,
            )
            if graph is None:
                fail_run(out_dir, "Pipeline did not produce a graph (e.g. no valid EE/localization).")
                sys.exit(1)
            graph_path = args.e2e_out or os.path.join(out_dir, ARTIFACTS["graph"])
            os.makedirs(os.path.dirname(os.path.abspath(graph_path)) or ".", exist_ok=True)
            with open(graph_path, "w") as f:
                json.dump(graph, f, indent=2)
            outputs = [record_output(out_dir, os.path.basename(graph_path), "Output graph (substructures + relationships)")]
            finish_run(out_dir, {"success": True, "graph_path": graph_path, "num_substructures": len(graph.get("substructures", [])), "num_relationships": len(graph.get("relationships", []))}, outputs)
            print(json.dumps(graph, indent=2))
            print(f"\nRun summary: {os.path.join(out_dir, RUN_SUMMARY_FILENAME)}")
        except Exception as e:
            fail_run(out_dir, str(e))
            raise
        return

    test_data = load_test_data(args.test_data)

    if args.build_prompts_only:
        if not args.ee_outputs:
            print("Error: --build_prompts_only requires --ee_outputs")
            sys.exit(1)
        out_dir = os.path.dirname(os.path.abspath(args.prompts_out)) if args.prompts_out else args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        start_run(out_dir, "build_prompts_only", _serializable_config(args))
        prompts_path = args.prompts_out or os.path.join(out_dir, ARTIFACTS["prompts"])
        ee_outputs = load_ee_outputs(args.ee_outputs)
        prompts_data = ee_to_rel_prompts(
            test_data,
            ee_outputs,
            molgenie_dir=args.molgenie_dir or os.path.join(os.path.dirname(args.test_data), "..", "molgenie"),
            molonto_cache_dir=os.path.join(os.path.dirname(args.test_data), "..", "molonto"),
        )
        os.makedirs(os.path.dirname(os.path.abspath(prompts_path)) or ".", exist_ok=True)
        to_save = [
            {"id": p["id"], "num_prompts": len(p["prompts_list"]), "prompts": [x for x in p["prompts_list"] if x]}
            for p in prompts_data
        ]
        with open(prompts_path, "w") as f:
            json.dump(to_save, f, indent=2)
        outputs = [record_output(out_dir, os.path.basename(prompts_path), "REL prompts per sample")]
        finish_run(out_dir, {"prompts_path": prompts_path, "num_samples": len(to_save)}, outputs)
        print(f"Prompts saved: {prompts_path}\nRun summary: {os.path.join(out_dir, RUN_SUMMARY_FILENAME)}")
        return

    if not args.rel_outputs:
        print("Error: --rel_outputs required for evaluation (or use --build_prompts_only)")
        sys.exit(1)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    start_run(out_dir, "evaluate", _serializable_config(args))

    rel_outputs = load_rel_outputs(args.rel_outputs)
    if len(rel_outputs) != len(test_data):
        print(f"Warning: rel_outputs has {len(rel_outputs)} samples, test_data has {len(test_data)}; aligning by id")
    selected = select_outputs(rel_outputs, policy=args.policy)
    predictions = parse_predictions(selected)

    selected_path = os.path.join(out_dir, ARTIFACTS["selected"])
    predictions_path = os.path.join(out_dir, ARTIFACTS["predictions"])
    metrics_path = os.path.join(out_dir, ARTIFACTS["metrics"])
    per_sample_path = os.path.join(out_dir, ARTIFACTS["per_sample"])

    with open(selected_path, "w") as f:
        json.dump(selected, f, indent=2)
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)

    metrics = evaluate(test_data, predictions)
    agg = {k: v for k, v in metrics.items() if k != "per_sample"}
    with open(metrics_path, "w") as f:
        json.dump(agg, f, indent=2)

    outputs = [
        record_output(out_dir, ARTIFACTS["selected"], "Selected REL raw output per sample"),
        record_output(out_dir, ARTIFACTS["predictions"], "Parsed graphs {id, graph}"),
        record_output(out_dir, ARTIFACTS["metrics"], "Aggregated evaluation metrics"),
    ]
    if metrics.get("per_sample"):
        with open(per_sample_path, "w") as f:
            json.dump(metrics["per_sample"], f, indent=2)
        outputs.append(record_output(out_dir, ARTIFACTS["per_sample"], "Per-sample metrics"))

    n = metrics.get("Total Evaluated", 0)
    finish_run(out_dir, {"metrics": agg, "num_evaluated": n}, outputs)

    print("\n" + json.dumps(agg, indent=2))
    print(f"\nRun summary: {os.path.join(out_dir, RUN_SUMMARY_FILENAME)}")


if __name__ == "__main__":
    main()
