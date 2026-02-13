import json
import re


def extract_json_from_text(text: str):
    """
    Extract JSON from markdown-wrapped or raw text. Returns parsed object or None.
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if "```json" in text and "```" in text:
        try:
            chunk = text.split("```json")[1].split("```")[0].strip()
            return json.loads(chunk)
        except Exception:
            pass
    # Try raw JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find first {...} or [...]
    for start, end in [("{", "}"), ("[", "]")]:
        i = text.find(start)
        if i == -1:
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == start:
                depth += 1
            elif text[j] == end:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i : j + 1])
                    except Exception:
                        break
    return None


def parse_ee_output(output_text: str) -> list:
    """Parse one EE LLM output to list of {name, smiles}."""
    obj = extract_json_from_text(output_text)
    if obj is None:
        return None
    if isinstance(obj, list):
        return [{"name": x.get("name") or x.get("Name"), "smiles": x.get("smiles") or x.get("SMILES")} for x in obj]
    return None


def parse_rel_output(output_text: str) -> dict:
    """
    Parse one REL LLM output to ground-truth-like dict with 'substructures' and 'relationships'.
    Returns None if parsing fails.
    """
    obj = extract_json_from_text(output_text)
    if obj is None or not isinstance(obj, dict):
        return None
    subs = obj.get("substructures") or []
    rels = obj.get("relationships") or []
    return {"substructures": subs, "relationships": rels}