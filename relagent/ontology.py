"""
Ontology-based relationship context: use MolGenie ontology when available.
"""
import os
import pickle
from relagent.MolSubOnto import (
    MolSubOntoInstance,
    get_all_instances_and_relationships,
    find_relationship,
)
from relagent.localizer import build_relationship_context_rdkit


def build_relationship_context(
    smiles: str,
    entities: list,
    molonto_instance=None,
    relationships_dict=None,
    molgenie_mols=None,
    molgenie_nodes=None,
    cache_dir=None,
    sample_id=None,
) -> str:
    """
    Build the relationship context string for the REL prompt.
    If relationships_dict (from ontology) is provided, use find_relationship.
    Otherwise fall back to RDKit-only relationship_from_indices via build_relationship_context_rdkit.
    """
    if relationships_dict is not None and entities:
        return _context_from_ontology(entities, relationships_dict)
    return build_relationship_context_rdkit(smiles, entities)


def _context_from_ontology(entities: list, relationships_dict: dict) -> str:
    """Build context string from precomputed ontology relationships_dict."""
    entity_indices2name = {}
    for entity in entities:
        name = entity.get("name", "")
        indices_list = entity.get("indices") or []
        if not indices_list:
            continue
        for idx_tuple in indices_list:
            index_str = ",".join(map(str, sorted(idx_tuple)))
            entity_indices2name[index_str] = name

    lines = []
    mentioned = set()
    keys = list(entity_indices2name.keys())
    for i, index_a in enumerate(keys):
        for index_b in keys[i + 1 :]:
            if (index_b, index_a) in mentioned or (index_a, index_b) in mentioned:
                continue
            mentioned.add((index_a, index_b))
            mentioned.add((index_b, index_a))
            rel_label = find_relationship(relationships_dict, index_a, index_b)
            if rel_label:
                name_a = entity_indices2name[index_a]
                name_b = entity_indices2name[index_b]
                lines.append(
                    f"Relationship between {name_a} ({index_a}) and {name_b} ({index_b}): {rel_label}"
                )
    return "\n".join(lines)


def load_or_build_ontology(smiles: str, sample_id: str, cache_dir: str, all_mols, all_nodes_dict):
    """
    Load MolSubOntoInstance from cache if present, else build and cache.
    Returns (molonto, instances_dict, relationships_dict) or (None, None, None) on failure.
    """
    if all_mols is None or all_nodes_dict is None:
        return None, None, None
    os.makedirs(cache_dir, exist_ok=True)
    graph_path = os.path.join(cache_dir, f"{sample_id}.pickle")
    ontology_path = os.path.join(cache_dir, f"{sample_id}.ttl")
    inst_path = os.path.join(cache_dir, f"{sample_id}_instances.pickle")
    rel_path = os.path.join(cache_dir, f"{sample_id}_relationships.pickle")

    if os.path.exists(inst_path) and os.path.exists(rel_path):
        try:
            instances_dict = pickle.load(open(inst_path, "rb"))
            relationships_dict = pickle.load(open(rel_path, "rb"))
            return None, instances_dict, relationships_dict
        except Exception:
            pass

    try:
        if os.path.exists(graph_path) and os.path.exists(ontology_path):
            molonto = MolSubOntoInstance(
                smiles, all_mols, all_nodes_dict,
                load_graph=True, graph_path=graph_path,
                load_ontology=True, ontology_path=ontology_path,
            )
        else:
            molonto = MolSubOntoInstance(smiles, all_mols, all_nodes_dict)
            molonto.save_graph(graph_path)
            molonto.save_ontology(ontology_path)
        instances_dict, relationships_dict = get_all_instances_and_relationships(molonto.ontology)
        pickle.dump(instances_dict, open(inst_path, "wb"))
        pickle.dump(relationships_dict, open(rel_path, "wb"))
        return molonto, instances_dict, relationships_dict
    except Exception:
        return None, None, None
