from rdkit import Chem
import networkx as nx
from rdflib import Graph as RDFGraph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD
import pickle
# import obonet

# url = "../data/mol_classes_ext_2024-12-06.obo"
# graph = obonet.read_obo(url)
# id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
# name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True)}

class MolSubOnto:

    def __init__(self, smiles, smarts_mols, all_nodes_dict, base_namespace="http://example.org/chemistry#"):
        """
        Initialize the MolSubOnto class with a SMILES string of the molecule.
        
        Args:
            smiles (str): SMILES representation of the molecule.
            smarts_mols (list): List of SMARTS patterns as RDKit Mol objects.
            all_nodes_dict (dict): Dictionary mapping indices to node information.
            base_namespace (str): Base namespace URI for the ontology.
        """
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.mol_subs = self.search_substructure(self.mol, smarts_mols, all_nodes_dict)
        self.graph = nx.Graph()
        self.build_graph()
        self.base_ns = Namespace(base_namespace)
        self.ontology = RDFGraph()
        self.ontology.bind("chem", self.base_ns)
        self.build_ontology()
    
    def search_substructure(self, mol, smarts_mols, all_nodes_dict):
        subs = []
        smiles = set()
        for i, smarts_mol in enumerate(smarts_mols):
            if not smarts_mol:
                continue
            matches = mol.GetSubstructMatches(query=smarts_mol)
            
            node = all_nodes_dict[i]
            if matches:
                smart_smiles = Chem.MolToSmiles(smarts_mol)
                if smart_smiles not in smiles:
                    smiles.add(smart_smiles)
                else:
                    continue
                
                subs.append({
                    'molgenie': node,
                    'indices': matches,
                    # 'descendants': descendants,  # Omit if not needed or not defined
                    'smiles': smart_smiles,
                    'smarts_mol': smarts_mol
                })
        return subs

    def build_graph(self):
        """
        Constructs the graph by adding nodes and edges based on relationships.
        """
        # Add nodes to the graph
        for sub in self.mol_subs:
            name = sub['molgenie']['name']
            self.graph.add_node(name, **sub)

        # Compute relationships and add edges
        for i, sub_a in enumerate(self.mol_subs):
            for j, sub_b in enumerate(self.mol_subs):
                if i < j:
                    relation = self.determine_relationship(sub_a, sub_b)
                    if relation:
                        name_a = sub_a['molgenie']['name']
                        name_b = sub_b['molgenie']['name']
                        self.graph.add_edge(name_a, name_b, relation=relation)

    def determine_relationship(self, sub_a, sub_b):
        """
        Determines the relationship between two substructures.

        Args:
            sub_a (dict): Information about substructure A.
            sub_b (dict): Information about substructure B.

        Returns:
            str or None: The relationship type or None if no relationship is found.
        """
        # Collect all atom indices for substructures A and B
        indices_a_sets = [set(match) for match in sub_a['indices']]
        indices_b_sets = [set(match) for match in sub_b['indices']]

        # Check for "Containment"
        for indices_a in indices_a_sets:
            for indices_b in indices_b_sets:
                if indices_a != indices_b:
                    if indices_a.issubset(indices_b):
                        return "Containment"
                    elif indices_b.issubset(indices_a):
                        return "Containment"

        # Check for "AtomAttachment" and "BondAttachment"
        for indices_a in indices_a_sets:
            for indices_b in indices_b_sets:
                shared_atoms = indices_a & indices_b
                if len(shared_atoms) == 1:
                    # Check if they are not connected via a separate bond
                    bond_exists = self.check_bond_between_substructures(self.mol, indices_a, indices_b)
                    if not bond_exists:
                        return "AtomAttachment"
                elif len(shared_atoms) == 0:
                    # Check for exactly one bond connecting atoms from each substructure
                    num_bonds = self.count_bonds_between_substructures(self.mol, indices_a, indices_b)
                    if num_bonds == 1:
                        return "BondAttachment"

        # Check for "Fusion"
        if self.is_ring(sub_a['smarts_mol']) and self.is_ring(sub_b['smarts_mol']):
            for indices_a in indices_a_sets:
                for indices_b in indices_b_sets:
                    shared_atoms = indices_a & indices_b
                    if len(shared_atoms) == 2:
                        # Check if shared atoms are adjacent and share a bond
                        if self.shared_atoms_adjacent(self.mol, shared_atoms):
                            return "Fusion"

        # If none of the above, return "Others"
        return "Others"

    def check_bond_between_substructures(self, mol, indices_a, indices_b):
        """
        Checks if there is a bond between any atom in indices_a and any atom in indices_b.

        Args:
            mol (rdkit.Chem.Mol): RDKit molecule object.
            indices_a (set): Atom indices of substructure A.
            indices_b (set): Atom indices of substructure B.

        Returns:
            bool: True if a bond exists between the substructures, False otherwise.
        """
        for idx_a in indices_a:
            atom_a = mol.GetAtomWithIdx(idx_a)
            for bond in atom_a.GetBonds():
                idx_b = bond.GetOtherAtomIdx(idx_a)
                if idx_b in indices_b:
                    return True
        return False

    def count_bonds_between_substructures(self, mol, indices_a, indices_b):
        """
        Counts the number of bonds between atoms in indices_a and indices_b.

        Args:
            mol (rdkit.Chem.Mol): RDKit molecule object.
            indices_a (set): Atom indices of substructure A.
            indices_b (set): Atom indices of substructure B.

        Returns:
            int: Number of bonds between the substructures.
        """
        count = 0
        for idx_a in indices_a:
            atom_a = mol.GetAtomWithIdx(idx_a)
            for bond in atom_a.GetBonds():
                idx_b = bond.GetOtherAtomIdx(idx_a)
                if idx_b in indices_b:
                    count += 1
        return count

    def is_ring(self, sub_mol):
        """
        Determines if a substructure is a ring.

        Args:
            sub_mol (rdkit.Chem.Mol): RDKit molecule object of the substructure.

        Returns:
            bool: True if the substructure is a ring, False otherwise.
        """
        ring_info = sub_mol.GetRingInfo()
        return ring_info.NumRings() > 0

    def shared_atoms_adjacent(self, mol, shared_atoms):
        """
        Checks if shared atoms are adjacent (connected by a bond).

        Args:
            mol (rdkit.Chem.Mol): RDKit molecule object.
            shared_atoms (set): Set of shared atom indices.

        Returns:
            bool: True if shared atoms are adjacent, False otherwise.
        """
        atom_indices = list(shared_atoms)
        if len(atom_indices) != 2:
            return False
        idx1, idx2 = atom_indices
        bond = mol.GetBondBetweenAtoms(idx1, idx2)
        return bond is not None

    def get_graph(self):
        """
        Returns the ontology graph.

        Returns:
            networkx.Graph: The ontology graph.
        """
        return self.graph

    def build_ontology(self):
        """
        Constructs the ontology using RDFLib based on the graph of substructures.
        """
        for node in self.graph.nodes(data=True):
            name = node[0]
            data = node[1]
            node_uri = self.base_ns[name.replace(" ", "_")]
            # Add node as an OWL Class
            self.ontology.add((node_uri, RDF.type, OWL.Class))
            # Optionally, add label
            self.ontology.add((node_uri, RDFS.label, Literal(name)))
            # Add SMILES annotation
            self.ontology.add((node_uri, self.base_ns['smiles'], Literal(data['smiles'])))
        
        # Define relationship properties
        relationships = ['BondAttachment', 'AtomAttachment', 'Containment', 'Fusion', 'Others']
        for rel in relationships:
            rel_uri = self.base_ns[rel]
            self.ontology.add((rel_uri, RDF.type, OWL.ObjectProperty))
            self.ontology.add((rel_uri, RDFS.label, Literal(rel)))
        
        # Add edges (relationships) to the ontology
        for edge in self.graph.edges(data=True):
            source_name = edge[0]
            target_name = edge[1]
            relation = edge[2]['relation']
            source_uri = self.base_ns[source_name.replace(" ", "_")]
            target_uri = self.base_ns[target_name.replace(" ", "_")]
            rel_uri = self.base_ns[relation]
            # Add relationship between nodes
            self.ontology.add((source_uri, rel_uri, target_uri))
    
    def save_ontology(self, filename, format='turtle'):
        """
        Serializes the ontology to a file.

        Args:
            filename (str): The output file name.
            format (str): The serialization format (e.g., 'turtle', 'xml', 'nt').
        """
        self.ontology.serialize(destination=filename, format=format)

    def query_ontology(self, query):
        """
        Executes a SPARQL query against the ontology.

        Args:
            query (str): SPARQL query string.

        Returns:
            list: Query results.
        """
        return self.ontology.query(query)

    def get_ontology(self):
        """
        Returns the RDFLib Graph representing the ontology.

        Returns:
            rdflib.Graph: The ontology graph.
        """
        return self.ontology
    

class MolSubOntoInstance:
    def __init__(self, smiles, smarts_mols, all_nodes_dict, base_namespace="http://example.org/chemistry#", include_others=False,
                 load_graph=False, graph_path=None,
                 load_ontology=False, ontology_path=None):
        """
        Initialize the MolSubOntoInstance class with a SMILES string of the molecule.

        Args:
            smiles (str): SMILES representation of the molecule.
            smarts_mols (list): List of SMARTS patterns as RDKit Mol objects.
            all_nodes_dict (dict): Dictionary mapping indices to node information.
            base_namespace (str): Base namespace URI for the ontology.
            include_others (bool): Whether to include "Others" in the ontology.
            build_graph (bool): Whether to build the graph.
            graph_path (str): The path to the graph file.
            load_ontology (bool): Whether to load the ontology from a file.
            ontology_path (str): The path to the ontology file.
        """
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.mol_subs = self.search_substructure(self.mol, smarts_mols, all_nodes_dict)
        self.graph = nx.Graph()
        if load_graph:
            self.graph = pickle.load(open(graph_path, 'rb'))
        else:
            self.build_graph()
        self.base_ns = Namespace(base_namespace)
        self.ontology = RDFGraph()
        self.ontology.bind("chem", self.base_ns)
        if load_ontology:
            self.ontology.parse(ontology_path, format='turtle')
        else:
            self.build_ontology(include_others)
    
    def get_graph(self):
        return self.graph
    
    def get_ontology(self):
        return self.ontology
    
    def save_graph(self, filename, format='pickle'):
        pickle.dump(self.graph, open(filename, 'wb'))
    
    def save_ontology(self, filename, format='turtle'):
        self.ontology.serialize(destination=filename, format=format)
    
    def query_ontology(self, query):
        return self.ontology.query(query)
    
    def search_substructure(self, mol, smarts_mols, all_nodes_dict):
        """
        Find all instances of substructures in the molecule and assign unique IDs.

        Args:
            mol (rdkit.Chem.Mol): The RDKit molecule object.
            smarts_mols (list): List of SMARTS patterns.
            all_nodes_dict (dict): Dictionary with substructure information.

        Returns:
            list: A list of dictionaries, each representing a substructure instance.
        """
        subs = []
        substructure_counts = {}  # To keep track of counts per substructure type
        for i, smarts_mol in enumerate(smarts_mols):
            if not smarts_mol:
                continue
            matches = mol.GetSubstructMatches(query=smarts_mol)
            
            node = all_nodes_dict[i]
            substructure_name = node['name']
            if matches:
                smart_smiles = Chem.MolToSmiles(smarts_mol)
                # Initialize count for this substructure if not present
                if substructure_name not in substructure_counts:
                    substructure_counts[substructure_name] = 0  # start from 0
                # For each match, create a separate instance
                for match in matches:
                    substructure_counts[substructure_name] += 1  # increment count
                    instance_id = f"{substructure_name.replace(' ', '_')}_{substructure_counts[substructure_name]}"
                    # sort match by index
                    match = sorted(match)
                    subs.append({
                        'molgenie': node,
                        'indices': match,  # single match
                        'smiles': smart_smiles,
                        'smarts_mol': smarts_mol,
                        'instance_id': instance_id
                    })
        return subs
    
    def build_graph(self):
        """
        Constructs the graph by adding nodes and edges based on relationships between instances.
        """
        # Add nodes to the graph
        for sub in self.mol_subs:
            instance_id = sub['instance_id']
            self.graph.add_node(instance_id, **sub)
    
        # Compute relationships and add edges
        for i, sub_a in enumerate(self.mol_subs):
            for j, sub_b in enumerate(self.mol_subs):
                if i < j:
                    relation = self.determine_relationship(sub_a, sub_b)
                    if relation:
                        instance_id_a = sub_a['instance_id']
                        instance_id_b = sub_b['instance_id']
                        self.graph.add_edge(instance_id_a, instance_id_b, relation=relation)
    
    def determine_relationship(self, sub_a, sub_b):
        """
        Determines the relationship between two substructure instances.

        Args:
            sub_a (dict): Information about substructure A.
            sub_b (dict): Information about substructure B.

        Returns:
            str or None: The relationship type or None if no relationship is found.
        """
        # Get atom indices sets for substructures A and B
        indices_a_set = set(sub_a['indices'])
        indices_b_set = set(sub_b['indices'])
        
        # Check for "Containment"
        if indices_a_set != indices_b_set:
            if indices_a_set.issubset(indices_b_set):
                return "Containment"
            elif indices_b_set.issubset(indices_a_set):
                return "Containment"
        
        # Check for "AtomAttachment" and "BondAttachment"
        shared_atoms = indices_a_set & indices_b_set
        if len(shared_atoms) == 1:
            # Check if they are not connected via a separate bond
            bond_exists = self.check_bond_between_substructures(self.mol, indices_a_set, indices_b_set)
            if not bond_exists:
                return "AtomAttachment"
        elif len(shared_atoms) == 0:
            # Check for exactly one bond connecting atoms from each substructure
            num_bonds = self.count_bonds_between_substructures(self.mol, indices_a_set, indices_b_set)
            if num_bonds == 1:
                return "BondAttachment"
        
        # Check for "Fusion"
        if self.is_ring(sub_a['smarts_mol']) and self.is_ring(sub_b['smarts_mol']):
            shared_atoms = indices_a_set & indices_b_set
            if len(shared_atoms) == 2:
                # Check if shared atoms are adjacent and share a bond
                if self.shared_atoms_adjacent(self.mol, shared_atoms):
                    return "Fusion"

        # If none of the above, return "Others"
        return None

    def check_bond_between_substructures(self, mol, indices_a, indices_b):
        """
        Checks if there is a bond between any atom in indices_a and any atom in indices_b.

        Args:
            mol (rdkit.Chem.Mol): RDKit molecule object.
            indices_a (set): Atom indices of substructure A.
            indices_b (set): Atom indices of substructure B.

        Returns:
            bool: True if a bond exists between the substructures, False otherwise.
        """
        for idx_a in indices_a:
            atom_a = mol.GetAtomWithIdx(idx_a)
            for bond in atom_a.GetBonds():
                idx_b = bond.GetOtherAtomIdx(idx_a)
                if idx_b in indices_b:
                    return True
        return False

    def count_bonds_between_substructures(self, mol, indices_a, indices_b):
        """
        Counts the number of bonds between atoms in indices_a and indices_b.

        Args:
            mol (rdkit.Chem.Mol): RDKit molecule object.
            indices_a (set): Atom indices of substructure A.
            indices_b (set): Atom indices of substructure B.

        Returns:
            int: Number of bonds between the substructures.
        """
        count = 0
        for idx_a in indices_a:
            atom_a = mol.GetAtomWithIdx(idx_a)
            for bond in atom_a.GetBonds():
                idx_b = bond.GetOtherAtomIdx(idx_a)
                if idx_b in indices_b:
                    count += 1
        return count

    def is_ring(self, sub_mol):
        """
        Determines if a substructure is a ring.

        Args:
            sub_mol (rdkit.Chem.Mol): RDKit molecule object of the substructure.

        Returns:
            bool: True if the substructure is a ring, False otherwise.
        """
        ring_info = sub_mol.GetRingInfo()
        return ring_info.NumRings() > 0

    def shared_atoms_adjacent(self, mol, shared_atoms):
        """
        Checks if shared atoms are adjacent (connected by a bond).

        Args:
            mol (rdkit.Chem.Mol): RDKit molecule object.
            shared_atoms (set): Set of shared atom indices.

        Returns:
            bool: True if shared atoms are adjacent, False otherwise.
        """
        atom_indices = list(shared_atoms)
        if len(atom_indices) != 2:
            return False
        idx1, idx2 = atom_indices
        bond = mol.GetBondBetweenAtoms(idx1, idx2)
        return bond is not None

    def build_ontology(self, include_others=False):
        """
        Constructs the ontology using RDFLib based on the graph of substructure instances.
        """
        # Keep track of defined classes (substructure types)
        defined_classes = set()
        for node in self.graph.nodes(data=True):
            instance_id = node[0]  # Node identifier (instance ID)
            data = node[1]
            substructure_name = data['molgenie']['name']
            substructure_class_uri = self.base_ns[substructure_name.replace(' ', '_')]
            instance_uri = self.base_ns[instance_id]
            
            # Define the substructure type as an OWL Class, if not already defined
            if substructure_name not in defined_classes:
                self.ontology.add((substructure_class_uri, RDF.type, OWL.Class))
                # Optionally, add label
                self.ontology.add((substructure_class_uri, RDFS.label, Literal(substructure_name)))
                defined_classes.add(substructure_name)
            
            # Add the instance as an individual of the substructure type
            self.ontology.add((instance_uri, RDF.type, substructure_class_uri))
            # Optionally, add label
            self.ontology.add((instance_uri, RDFS.label, Literal(instance_id)))
            # Add name
            self.ontology.add((instance_uri, self.base_ns['name'], Literal(substructure_name)))
            # Add annotations
            self.ontology.add((instance_uri, self.base_ns['smiles'], Literal(data['smiles'])))
            # You can also add the indices of the instance
            indices_str = ','.join(str(idx) for idx in data['indices'])
            indices_literal = Literal(indices_str)
            self.ontology.add((instance_uri, self.base_ns['indices'], indices_literal))
        
        # Define relationship properties
        relationships = ['BondAttachment', 'AtomAttachment', 'Containment', 'Fusion']
        if include_others:
            relationships.append('Others')
        for rel in relationships:
            rel_uri = self.base_ns[rel]
            self.ontology.add((rel_uri, RDF.type, OWL.ObjectProperty))
            self.ontology.add((rel_uri, RDFS.label, Literal(rel)))
        
        # Add edges (relationships) between instances
        for edge in self.graph.edges(data=True):
            source_id = edge[0]
            target_id = edge[1]
            relation = edge[2]['relation']
            source_uri = self.base_ns[source_id]
            target_uri = self.base_ns[target_id]
            rel_uri = self.base_ns[relation]
            # Add relationship between instances
            self.ontology.add((source_uri, rel_uri, target_uri))

def search_index(smiles: str, smarts: str):
    """
    Search for substructure matches in a molecule using a SMARTS pattern.

    Args:
        smiles (str): SMILES string of the molecule to search in.
        smarts (str): SMARTS string of the substructure to search for.

    Returns:
        list: List of tuples, each containing the indices of the matched substructure.
               Returns None if the molecule or pattern is None, or if an exception occurs.
               Returns -1 if no matches are found.
    """
    try:
        # Convert SMILES and SMARTS to RDKit molecules
        mol = Chem.MolFromSmiles(smiles)
        pattern = Chem.MolFromSmiles(smarts)
        
        if mol is None or pattern is None:
            return None
            
        # Find matches
        matches = mol.GetSubstructMatches(pattern)
        return matches
        
    except Exception:
        return -1


def get_all_instances_and_relationships(ontology):
    """
    Retrieves all instances and relationships from the ontology.

    Args:
        ontology (rdflib.Graph): The ontology graph.

    Returns:
        instances_dict (dict): Mapping from indices string to instance data.
        relationships_dict (dict): Mapping from (indices_a_str, indices_b_str) to relationship label.
    """
    query = """
    PREFIX chem: <http://example.org/chemistry#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?instance ?indices ?name ?smiles ?instanceLabel
    WHERE {
      ?instance chem:indices ?indices ;
                rdfs:label ?instanceLabel ;
                chem:smiles ?smiles ;
                chem:name ?name .
    }
    """

    instances = {}
    for row in ontology.query(query):
        indices_str = str(row['indices'])
        instance_data = {
            'instance': row['instance'],
            'indices': indices_str,
            'name': str(row['name']),
            'smiles': str(row['smiles']),
            'label': str(row['instanceLabel'])
        }
        instances[indices_str] = instance_data

    # Retrieve all relationships between instances
    relationship_query = """
    PREFIX chem: <http://example.org/chemistry#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?instanceA ?indicesA ?instanceB ?indicesB ?relationshipLabel
    WHERE {
      ?instanceA chem:indices ?indicesA .
      ?instanceB chem:indices ?indicesB .
      ?instanceA ?relationship ?instanceB .
      ?relationship rdfs:label ?relationshipLabel .
    }
    """

    relationships = {}
    for row in ontology.query(relationship_query):
        indices_a_str = str(row['indicesA'])
        indices_b_str = str(row['indicesB'])
        relationship_label = str(row['relationshipLabel'])

        key = (indices_a_str, indices_b_str)
        relationships[key] = relationship_label

    return instances, relationships

def find_relationship(relationships_dict, indices_a, indices_b):
    rel_label1 = relationships_dict.get((indices_a, indices_b))
    rel_label2 = relationships_dict.get((indices_b, indices_a))
    rel_label = rel_label1 if rel_label1 else rel_label2
    return rel_label