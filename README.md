# RelAgent: A multi-agent solution for molecular relationship grounding

Understanding the spatial relationships between a molecule's substructures is crucial for scientific domains like drug discovery, yet a significant gap exists between abstract natural language descriptions and formal chemical structures. This semantic-structural gap causes existing models, including large language models, to fail at reliably grounding textual concepts to specific molecular components, hindering their use in complex tasks like interpreting patents or executing precise molecular edits. To bridge this gap, we introduce the task of molecular grounding and propose RelAgent, a cooperative multi-agent framework. RelAgent mimics human-like reasoning by decomposing the task into specialized agents for entity extraction, localization, and relationship inference, enhanced by an ontology-guided verification mechanism. This approach significantly improves performance, demonstrating a robust path towards truly structure-aware molecular intelligence.

![](figure/baseline.png)

## Code structure

The pipeline follows the paper: **Entity Extraction (EE)** → **Localization** → **Ontology-guided relationship context** → **Relationship Reasoning (REL)** → **Verifier/selection** → **Evaluation**.

| Path | Role |
|------|------|
| `relagent/util.py` | JSON extraction from LLM text, EE/REL output parsing |
| `relagent/template.py` | EE and REL prompt templates |
| `relagent/llm.py` | vLLM-backed `RelAgentLLM`: `generate_ee()`, `generate_rel()` using templates |
| `relagent/localizer.py` | RDKit-based localization (`search_index`), entity→indices, RDKit-only relationship context |
| `relagent/MolSubOnto.py` | MolGenie-based molecular ontology (instances, RDF, SPARQL); `search_index` |
| `relagent/ontology.py` | Build relationship context from ontology or fallback to RDKit |
| `relagent/pipeline.py` | Load data/EE/REL outputs, build REL prompts from EE, select policy, parse predictions; `run_end_to_end(smiles, caption, llm)` for single-input E2E |
| `relagent/evaluation.py` | Substructure and relationship metrics (P/R/F1) vs ground truth |
| `relagent/config.py` | Default paths for data, molonto cache, outputs |
| `run_baseline.py` | CLI: evaluate from EE + REL outputs; generate REL prompts; or **run E2E** (SMILES + caption → graph) with vLLM |
| `scripts/generate_synthetic_outputs.py` | Create EE/REL outputs from test ground truth (sanity check) |
| `data/molground/` | `molground_test.json`, `molground_val.json` (id, smiles, caption, ground_truth) |

Optional: `data/molgenie/` with `molgenie_all_mols.pkl` and `molgenie_all_nodes_dict.pkl` for ontology-based relationship context; otherwise the pipeline uses RDKit-only relationship context.

## Setup

```bash
pip install -r requirements.txt
```

Requires: `rdkit`, `networkx`, `rdflib`, `tqdm`, `vllm` (see `requirements.txt`). For evaluation-only (no prompt building), `rdkit` is not required. End-to-end runs require `vllm` and a GPU.

## Running the baseline

0. **End-to-end: one input (SMILES + caption) → full graph** (uses vLLM):

   ```bash
   python run_baseline.py --run_e2e --smiles "CC1=CC=CC=C1" --caption "Benzene ring" --model meta-llama/Llama-3.2-1B-Instruct
   ```

   Optional: `--n_ee_samples N`, `--e2e_policy first|majority_voting|random`, `--e2e_out path/to/graph.json`, `--molgenie_dir data/molgenie`.

1. **Evaluate with precomputed EE and REL outputs** (e.g. from your LLM runs):

   ```bash
   python run_baseline.py --test_data data/molground/molground_test.json \
     --ee_outputs path/to/ee_outputs.json \
     --rel_outputs path/to/rel_outputs.json \
     --output_dir outputs/run1 --policy majority_voting
   ```

   EE format: JSON list of `{"id": "<sample_id>", "outputs": ["<json string>", ...]}`.  
   REL format: JSON list of `{"id": "<sample_id>", "entities": [...], "relationships": [...], "outputs": ["<markdown json>", ...]}`.

2. **Generate REL prompts from EE outputs** (then run your REL model on the saved prompts):

   ```bash
   python run_baseline.py --test_data data/molground/molground_test.json \
     --ee_outputs path/to/ee_outputs.json \
     --build_prompts_only --prompts_out outputs/rel_prompts.json
   ```

3. **Optional MolGenie ontology** (if you have the pickles):

   ```bash
   python run_baseline.py ... --molgenie_dir data/molgenie
   ```

4. **Synthetic run** (ground truth as REL output, for sanity check):

   ```bash
   python scripts/generate_synthetic_outputs.py --max_samples 10 --out_dir outputs/synthetic
   python run_baseline.py --rel_outputs outputs/synthetic/rel_outputs.json --output_dir outputs/synthetic/eval
   ```

