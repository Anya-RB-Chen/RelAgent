"""
Unified run recording: one summary file per run for logging, reading, and monitoring.
"""
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


RUN_SUMMARY_FILENAME = "run_summary.json"
RUN_LOG_FILENAME = "run.log"

# Standard artifact names (all under output_dir)
ARTIFACTS = {
    "predictions": "predictions.json",
    "metrics": "metrics.json",
    "per_sample": "per_sample.json",
    "selected": "selected.json",
    "graph": "graph.json",
    "prompts": "prompts.json",
}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _serializable_config(args) -> Dict[str, Any]:
    """Build a minimal, serializable config from argparse args."""
    out = {}
    for k, v in vars(args).items():
        if v is None or (isinstance(v, (str, int, float, bool)) and not k.startswith("_")):
            out[k] = v
        elif isinstance(v, list):
            out[k] = v
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def start_run(
    output_dir: str,
    run_type: str,
    config: Dict[str, Any],
) -> None:
    """Create output_dir and write initial run_summary (status: running)."""
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "run_type": run_type,
        "status": "running",
        "timestamp": _ts(),
        "config": config,
        "summary": None,
        "outputs": [],
    }
    path = os.path.join(output_dir, RUN_SUMMARY_FILENAME)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(output_dir, f"Started {run_type}")


def finish_run(
    output_dir: str,
    summary_payload: Dict[str, Any],
    outputs: List[Dict[str, str]],
) -> None:
    """Update run_summary with status ok, summary, and list of outputs."""
    path = os.path.join(output_dir, RUN_SUMMARY_FILENAME)
    if not os.path.isfile(path):
        start_run(output_dir, "unknown", {})
    with open(path, "r") as f:
        data = json.load(f)
    data["status"] = "ok"
    data["summary"] = summary_payload
    data["outputs"] = outputs
    data["finished_at"] = _ts()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    _log(output_dir, "Finished successfully")


def fail_run(output_dir: str, message: str) -> None:
    """Mark run as failed in run_summary."""
    path = os.path.join(output_dir, RUN_SUMMARY_FILENAME)
    if os.path.isfile(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {"run_type": "unknown", "config": {}, "outputs": []}
    data["status"] = "error"
    data["error"] = message
    data["finished_at"] = _ts()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    _log(output_dir, f"Failed: {message}")


def _log(output_dir: str, message: str) -> None:
    """Append one line to run.log."""
    path = os.path.join(output_dir, RUN_LOG_FILENAME)
    with open(path, "a") as f:
        f.write(f"[{_ts()}] {message}\n")


def record_output(output_dir: str, rel_path: str, description: str) -> Dict[str, str]:
    """Return entry for outputs list. rel_path is relative to output_dir."""
    return {"path": rel_path, "description": description}
