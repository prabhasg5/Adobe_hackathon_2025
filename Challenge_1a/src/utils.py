import json
from sentence_transformers import util

def cosine_sim(a, b):
    return util.pytorch_cos_sim(a, b)

def is_heading_candidate(line):
    text = line["text"].strip()
    if len(text.split()) > 15:
        return False
    return True

def save_output_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)











        