"""
Localization: map entity SMILES to atom indices in the molecule and compute pairwise relationships using RDKit.
"""
from rdkit import Chem

from relagent.MolSubOnto import search_index


def localize_entities(smiles: str, ee_list: list) -> list:
    """
    For each entity in ee_list (items with 'name' and 'smiles'), find all substructure matches
    in the molecule. Returns a list of entities with 'indices' added (list of match tuples).
    """
    if not ee_list:
        return []
    entities = []
    for e in ee_list:
        name = e.get("name") or e.get("Name")
        smi = e.get("smiles") or e.get("SMILES")
        if not name or not smi:
            entities.append({"name": name, "smiles": smi, "indices": None})
            continue
        matches = search_index(smiles, smi)
        if matches is None or matches == -1:
            entities.append({"name": name, "smiles": smi, "indices": None})
        else:
            # Normalize to list of lists for JSON serialization
            indices = [list(m) for m in matches]
            entities.append({"name": name, "smiles": smi, "indices": indices})
    return entities


def _check_bond_between(mol, indices_a, indices_b):
    for idx_a in indices_a:
        atom_a = mol.GetAtomWithIdx(idx_a)
        for bond in atom_a.GetBonds():
            idx_b = bond.GetOtherAtomIdx(idx_a)
            if idx_b in indices_b:
                return True
    return False


def _count_bonds_between(mol, indices_a, indices_b):
    count = 0
    for idx_a in indices_a:
        atom_a = mol.GetAtomWithIdx(idx_a)
        for bond in atom_a.GetBonds():
            idx_b = bond.GetOtherAtomIdx(idx_a)
            if idx_b in indices_b:
                count += 1
    return count


def _shared_atoms_adjacent(mol, shared_atoms):
    atoms = list(shared_atoms)
    if len(atoms) != 2:
        return False
    bond = mol.GetBondBetweenAtoms(atoms[0], atoms[1])
    return bond is not None


def _is_ring_fragment(mol, indices_set):
    """Return True if the subgraph induced by indices_set contains a ring."""
    if len(indices_set) < 3:
        return False
    indices_set = set(indices_set)
    bond_count = 0
    for idx in indices_set:
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(idx)
            if other in indices_set and other > idx:
                bond_count += 1
    # A connected cycle has n nodes and n edges; any connected graph with >= n edges has a ring
    n = len(indices_set)
    return bond_count >= n


def relationship_from_indices(mol, indices_a, indices_b):
    """
    Compute the relationship type between two substructures given by atom index sets.
    indices_a, indices_b: iterables of atom indices (e.g. from search_index).
    Returns one of: BondAttachment, AtomAttachment, Containment, Fusion, Others.
    """
    if mol is None:
        return "Others"
    set_a = set(indices_a)
    set_b = set(indices_b)
    if not set_a or not set_b:
        return "Others"

    # Containment
    if set_a != set_b:
        if set_a.issubset(set_b) or set_b.issubset(set_a):
            return "Containment"

    shared = set_a & set_b
    # AtomAttachment: share exactly one atom, no separate bond
    if len(shared) == 1:
        if not _check_bond_between(mol, set_a, set_b):
            return "AtomAttachment"
    elif len(shared) == 0:
        if _count_bonds_between(mol, set_a, set_b) == 1:
            return "BondAttachment"

    # Fusion: both rings, share exactly two adjacent atoms
    if _is_ring_fragment(mol, set_a) and _is_ring_fragment(mol, set_b):
        if len(shared) == 2 and _shared_atoms_adjacent(mol, shared):
            return "Fusion"

    return "Others"


def build_relationship_context_rdkit(smiles: str, entities: list) -> str:
    """
    Build the relationship context string from localized entities using RDKit only
    (no MolGenie ontology). Pairs entity instances by (indices_str, indices_str) and
    computes relationship via relationship_from_indices.
    """
    if not entities:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    # Collect (index_str -> name) for each entity instance
    index_to_name = {}
    for entity in entities:
        name = entity.get("name", "")
        indices_list = entity.get("indices") or []
        if not indices_list:
            continue
        for idx_tuple in indices_list:
            index_str = ",".join(map(str, sorted(idx_tuple)))
            index_to_name[index_str] = name

    lines = []
    seen = set()
    keys = list(index_to_name.keys())
    for i, index_a in enumerate(keys):
        for index_b in keys[i + 1 :]:
            if (index_b, index_a) in seen or (index_a, index_b) in seen:
                continue
            seen.add((index_a, index_b))
            indices_a = [int(x) for x in index_a.split(",")]
            indices_b = [int(x) for x in index_b.split(",")]
            rel = relationship_from_indices(mol, indices_a, indices_b)
            if rel:
                name_a = index_to_name[index_a]
                name_b = index_to_name[index_b]
                lines.append(
                    f"Relationship between {name_a} ({index_a}) and {name_b} ({index_b}): {rel}"
                )
    return "\n".join(lines)
