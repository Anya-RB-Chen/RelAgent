from rdkit import Chem
import json


def extract_json_from_text(text: str) -> dict:
    json_data = None
    if "```json" in text and "```" in text:
        json_data = text.split("```json")[1].split("```")[0]
        try:
            json_data = json.loads(json_data)
            return json_data
        except:
            return None
    else:
        try:
            json_data = json.loads(text)
            return json_data
        except:
            return None