EE_TEMPLATE = """You are a chemistry expert. Given a caption and SMILES of a molecule, please extract its mentioned chemical entities from the caption where these entities exists in the given molecule, and output the chemical name with its SMILES. 
You should output the entities and their SMILESs into JSON format. If one entity is mentioned in the caption multiple times, you should output only one time.

## Example
SMILES: CC1=CC(CCCCCCCCCCCCCC)=C(C2=CC3=C(S2)C=C(C4=C(CCCCCCCCCCCCCC)C=C(C)S4)S3)S1
Caption: The basic structure consists of a central thienothiophene core with two thiophene rings.
Answer: 
```json
[
    {{
        "name": "thiophene", 
        "SMILES": "C1=CSC=C1"
    }},
    {{
        "name": "thienothiophene", 
        "SMILES": "C1=CSC2=C1SC=C2"
    }}
]
```

Remember to only output in JSON format and nothing else.
Please extract entities and SMILES from the following caption and SMILES.
SMILES: {smiles}
Caption: {caption}
Answer: 
"""


REL_TEMPLATE = """You are a chemistry expert. Follow the instructions and provide the answer in the format of JSON.

# Task Description
You will be given: a **SMILES** string and a **caption** of a molecule. Also, a list of **entities** extracted from the caption, entities' **indexes** regarding the SMILES (starting from index 0), and the relationships between the entities are provided for your reference.
Please determine the instance-level pairwise *relationships* for the chemical name entities mentioned in the caption. Note that the entities might be included in the molecule multiple times, and each of them has a unique ID.

Here are the detailed instructions:

1. You need to extract instance-level pairwise relationships through all pairs of entities, by analyzing their location indexes in the SMILES and deduce the topology relationship between each pair of the unique entities.
2. The instance-level pairwise relationships should be determined based on their positions in the molecule, and it should being chosen from one of the following: `\"BondAttachment\"`, `\"AtomAttachment\"`, `\"Containment\"`, `\"Fusion\"`, `\"Others\"`.
    - `\"BondAttachment\"`: There is exactly one bond connecting an atom in substructure_a and an atom in substructure_b, and they do not share any atom.
    - `\"AtomAttachment\"`: substructure_a and substructure_b share exactly one atom, and are not connected via a separate bond.
    - `\"Containment\"`: Every atom in substructure_a is also in substructure_b, and substructure_a ≠ substructure_b.
    - `\"Fusion\"`: substructure_a and substructure_b are both rings and share exactly two adjacent atoms and the bond between them.
    - `\"Others\"`: Any other type of relationship.
3. Please return a **JSON object** with two keys:
    - `\"substructures\"`: a list of all substructure instances mentioned in the caption, each with:
        - `name`: the official chemical name of the substructure.
        - `instance_id`: the unique identifier of the substructure.
        - `atom_indices`: the list of atom indices in the SMILES string.
    - `\"relationships\"`: a list of all unique instance pairs and their relationship type as the context information. Not every instance pair is necessarily related to each other, so you need to determine the relationship between each pair of instances.

# Example

### Input
SMILES: CC1=CC(CCCCCCCCCCCCCC)=C(C2=CC3=C(S2)C=C(C4=C(CCCCCCCCCCCCCC)C=C(C)S4)S3)S1
Caption: The basic structure consists of a central thienothiophene core with two thiophene rings.
Entities (SMILES, list of indexes):
```json
[
    {{
        "name": "thiophene",
        "SMILES": "C1=CSC=C1",
        "indexes": [(1, 2, 3, 18, 47), (26, 27, 42, 43, 45)]
    }},
    {{
        "name": "thienothiophene",
        "SMILES": "C1=CSC2=C1SC=C2",
        "indexes": [(19, 20, 21, 22, 23, 24, 25, 46)]
    }}
]
```
Relationships:
- Relationship between thienothiophene (19,20,21,22,23,24,25,46) and thiophene (1,2,3,18,47): BondAttachment
- Relationship between thienothiophene (19,20,21,22,23,24,25,46) and thiophene (26,27,42,43,45): BondAttachment

### Answer
```json
{{
    "substructures": [
        {{
            "name": "thiophene",
            "instance_id": "THIOP1",
            "atom_indices": [1, 2, 3, 18, 47]
        }},
        {{
            "name": "thiophene",
            "instance_id": "THIOP2",
            "atom_indices": [26, 27, 42, 43, 45]
        }},
        {{
            "name": "thienothiophene",
            "instance_id": "THIEN1",
            "atom_indices": [19, 20, 21, 22, 23, 24, 25, 46]
        }}
    ],
    "relationships": [
        {{
            "substructure_a": "THIOP1",
            "substructure_b": "THIOP2",
            "relationship": "Others"
        }},
        {{
            "substructure_a": "THIOP1",
            "substructure_b": "THIEN1",
            "relationship": "BondAttachment"
        }},
        {{
            "substructure_a": "THIOP2", 
            "substructure_b": "THIEN1",
            "relationship": "BondAttachment"
        }}
    ]
}}
```

# Your Task
Please follow the instructions, think step by step, and return the JSON object.

### Input
SMILES: {smiles}
Caption: {caption}
Entities (SMILES, list of indexes):
```json
{entities}
```
Relationships:
{relationships}

### Answer
"""