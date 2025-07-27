import os
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Choose a fast + accurate < 1GB CPU model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def compute_similarity_scores(query: str, section_map: dict) -> list:
    """
    Rank sections by similarity of (persona + task) vs. section headings and content.
    Input:
        - query: concatenated persona + job description string
        - section_map: dict like {filename: [{"title": str, "content": str, "page_number": int}]}
    Output:
        - ranked_sections: list of dicts sorted by score [{document, section_title, page_number, score}]
    """
    query_emb = model.encode(query, convert_to_tensor=True)
    results = []

    for doc_name, sections in section_map.items():
        for sec in sections:
            sec_text = sec['title'] + ". " + sec['content']
            sec_emb = model.encode(sec_text, convert_to_tensor=True)
            sim_score = util.pytorch_cos_sim(query_emb, sec_emb).item()

            results.append({
                "document": doc_name,
                "section_title": sec['title'],
                "page_number": sec['page_number'],
                "score": sim_score
            })

    # Rank by similarity
    results.sort(key=lambda x: x['score'], reverse=True)
    return results