"""
End-to-end pipeline: EE outputs -> localization -> relationship context -> REL prompts;
REL outputs -> selection -> parsed predictions.
"""
import json
import os
from tqdm import tqdm

from relagent.util import parse_ee_output, parse_rel_output
from relagent.template import EE_TEMPLATE, REL_TEMPLATE


def load_test_data(path: str) -> list:
    """Load molground test/val JSON. Each item: id, smiles, caption, ground_truth, optionally entity_extraction."""
    with open(path, "r") as f:
        return json.load(f)


def load_ee_outputs(path: str) -> list:
    """Load EE outputs: list of {id, outputs: [str, ...]}."""
    with open(path, "r") as f:
        return json.load(f)


def load_rel_outputs(path: str) -> list:
    """Load REL outputs: list of {id, entities?, relationships?, outputs: [str, ...]}."""
    with open(path, "r") as f:
        return json.load(f)


def _load_molgenie(molgenie_dir: str):
    """Load MolGenie pickles if present. Returns (all_mols, all_nodes_dict) or (None, None)."""
    if not molgenie_dir or not os.path.isdir(molgenie_dir):
        return None, None
    pkl_mols = os.path.join(molgenie_dir, "molgenie_all_mols.pkl")
    pkl_nodes = os.path.join(molgenie_dir, "molgenie_all_nodes_dict.pkl")
    if not os.path.isfile(pkl_mols) or not os.path.isfile(pkl_nodes):
        return None, None
    try:
        import pickle
        all_mols = pickle.load(open(pkl_mols, "rb"))
        all_nodes_dict = pickle.load(open(pkl_nodes, "rb"))
        return all_mols, all_nodes_dict
    except Exception:
        return None, None


def ee_to_rel_prompts(
    test_data: list,
    ee_outputs: list,
    rel_template: str = None,
    molgenie_dir: str = None,
    molonto_cache_dir: str = None,
) -> list:
    """
    From test_data and ee_outputs, run localization and build relationship context for each
    EE candidate; produce REL prompts. Returns list of dicts:
    {id, entities_list, relationships_list, prompts_list} so that prompts_list[i] corresponds
    to entities_list[i] and relationships_list[i].
    """
    from relagent.localizer import localize_entities
    from relagent.ontology import build_relationship_context, load_or_build_ontology

    rel_template = rel_template or REL_TEMPLATE
    molonto_cache_dir = molonto_cache_dir or "data/molonto"
    all_mols, all_nodes_dict = _load_molgenie(molgenie_dir or "data/molgenie")
    ee_by_id = {x["id"]: x for x in ee_outputs}

    result = []
    for test in tqdm(test_data, desc="Building REL prompts"):
        sample_id = test["id"]
        smiles = test["smiles"]
        caption = test["caption"]
        ee_record = ee_by_id.get(sample_id)
        if not ee_record or not ee_record.get("outputs"):
            result.append({"id": sample_id, "entities_list": [], "relationships_list": [], "prompts_list": []})
            continue

        ee_candidates = []
        for raw in ee_record["outputs"]:
            parsed = parse_ee_output(raw)
            ee_candidates.append(parsed)

        entities_list = []
        relationships_list = []
        relationships_dict = None
        if all_mols is not None and all_nodes_dict is not None:
            _, _, relationships_dict = load_or_build_ontology(
                smiles, sample_id, molonto_cache_dir, all_mols, all_nodes_dict
            )

        for ee_cand in ee_candidates:
            if ee_cand is None:
                entities_list.append(None)
                relationships_list.append(None)
                continue
            entities = localize_entities(smiles, ee_cand)
            entities_list.append(entities)
            ctx = build_relationship_context(
                smiles, entities, relationships_dict=relationships_dict
            )
            relationships_list.append(ctx)

        prompts_list = []
        for n, (entities, rel_ctx) in enumerate(zip(entities_list, relationships_list)):
            if entities is None or rel_ctx is None:
                prompts_list.append(None)
                continue
            entities_str = json.dumps(entities, indent=4)
            prompt = rel_template.format(
                smiles=smiles,
                caption=caption,
                entities=entities_str,
                relationships=rel_ctx,
            )
            prompts_list.append(prompt)

        result.append({
            "id": sample_id,
            "entities_list": entities_list,
            "relationships_list": relationships_list,
            "prompts_list": prompts_list,
        })
    return result


def select_outputs(rel_outputs: list, policy: str = "first") -> list:
    """
    Select one output per sample from REL outputs. policy: first, majority_voting, random.
    Returns list of {id, output: str}.
    """
    selected = []
    for rec in rel_outputs:
        sample_id = rec["id"]
        outputs = rec.get("outputs") or []
        if not outputs:
            selected.append({"id": sample_id, "output": None})
            continue
        if policy == "first":
            out = next((o for o in outputs if o), None) or outputs[0]
        elif policy == "majority_voting":
            out_count = {}
            for o in outputs:
                if o not in out_count:
                    out_count[o] = 0
                out_count[o] += 1
            out = max(out_count, key=out_count.get)
        elif policy == "random":
            import random
            out = random.choice([o for o in outputs if o]) or outputs[0]
        else:
            out = outputs[0]
        selected.append({"id": sample_id, "output": out})
    return selected


def parse_predictions(selected: list) -> list:
    """Parse each selected REL output string to {substructures, relationships}. Returns list of {id, graph}."""
    from relagent.util import parse_rel_output
    result = []
    for item in selected:
        g = parse_rel_output(item["output"]) if item.get("output") else None
        result.append({"id": item["id"], "graph": g})
    return result


def run_pipeline_from_ee_and_rel(
    test_path: str,
    ee_outputs_path: str,
    rel_outputs_path: str,
    output_dir: str = None,
    policy: str = "first",
    molgenie_dir: str = None,
) -> list:
    """
    Load test data, EE outputs, and REL outputs; select one output per sample; parse to graphs.
    Optionally save selected and parsed to output_dir.
    """
    test_data = load_test_data(test_path)
    ee_outputs = load_ee_outputs(ee_outputs_path)
    rel_outputs = load_rel_outputs(rel_outputs_path)
    selected = select_outputs(rel_outputs, policy=policy)
    predictions = parse_predictions(selected)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "selected_outputs.json"), "w") as f:
            json.dump(selected, f, indent=2)
        with open(os.path.join(output_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f, indent=2)
    return predictions


def run_end_to_end(
    smiles: str,
    caption: str,
    llm=None,
    llm_ee=None,
    llm_rel=None,
    n_ee_samples: int = 1,
    rel_selection_policy: str = "first",
    molgenie_dir: str = None,
    molonto_cache_dir: str = None,
    sample_id: str = "e2e",
) -> dict:
    """
    Run the full pipeline on a single input (SMILES, caption) and return the predicted graph.

    Args:
        smiles: SMILES string of the molecule.
        caption: Natural language caption.
        llm: RelAgentLLM instance (from relagent.llm) for both EE and REL generation (if llm_ee/llm_rel not provided).
        llm_ee: Optional separate RelAgentLLM instance for Entity Extraction. If None, uses `llm`.
        llm_rel: Optional separate RelAgentLLM instance for Relationship Reasoning. If None, uses `llm` or `llm_ee`.
        n_ee_samples: Number of EE samples (Best-of-N); one REL run per valid EE candidate.
        rel_selection_policy: How to pick final graph among REL outputs: first, majority_voting, random.
        molgenie_dir: Optional path to MolGenie pickles for ontology context.
        molonto_cache_dir: Optional cache dir for ontology artifacts.
        sample_id: Optional id for ontology cache keys.

    Returns:
        Graph dict with keys "substructures" and "relationships", or None if pipeline failed.
    """
    from relagent.localizer import localize_entities
    from relagent.ontology import build_relationship_context, load_or_build_ontology
    from relagent.util import parse_rel_output

    # Resolve LLM instances: llm_ee and llm_rel take precedence
    if llm_ee is None:
        llm_ee = llm
    if llm_rel is None:
        llm_rel = llm_ee
    
    if llm_ee is None:
        raise ValueError("Either llm or llm_ee must be provided")

    molonto_cache_dir = molonto_cache_dir or "data/molonto"
    all_mols, all_nodes_dict = _load_molgenie(molgenie_dir or "data/molgenie")
    relationships_dict = None
    if all_mols is not None and all_nodes_dict is not None:
        _, _, relationships_dict = load_or_build_ontology(
            smiles, sample_id, molonto_cache_dir, all_mols, all_nodes_dict
        )

    # 1) Entity Extraction
    ee_raw_list = llm_ee.generate_ee(smiles, caption, n=n_ee_samples)
    ee_candidates = [parse_ee_output(raw) for raw in ee_raw_list]

    # 2) Localization + relationship context per EE candidate
    entities_list = []
    relationships_list = []
    prompts_list = []
    for ee_cand in ee_candidates:
        if ee_cand is None:
            continue
        entities = localize_entities(smiles, ee_cand)
        if not entities or all(e.get("indices") is None or e.get("indices") == [] for e in entities):
            continue
        ctx = build_relationship_context(smiles, entities, relationships_dict=relationships_dict)
        entities_str = json.dumps(entities, indent=4)
        prompt = REL_TEMPLATE.format(
            smiles=smiles,
            caption=caption,
            entities=entities_str,
            relationships=ctx,
        )
        entities_list.append(entities)
        relationships_list.append(ctx)
        prompts_list.append(prompt)

    if not prompts_list:
        return None

    # 3) Relationship Reasoning
    rel_outputs = llm_rel.generate_rel(prompts_list, use_tqdm=len(prompts_list) > 1)

    # 4) Select one output
    if rel_selection_policy == "first":
        chosen = rel_outputs[0]
    elif rel_selection_policy == "majority_voting":
        out_count = {}
        for o in rel_outputs:
            out_count[o] = out_count.get(o, 0) + 1
        chosen = max(out_count, key=out_count.get)
    elif rel_selection_policy == "random":
        import random
        chosen = random.choice(rel_outputs)
    else:
        chosen = rel_outputs[0]

    # 5) Parse to graph
    graph = parse_rel_output(chosen)
    return graph
