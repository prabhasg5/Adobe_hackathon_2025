# src/heading_ranker.py

from sentence_transformers import SentenceTransformer, util
from src.utils import is_heading_candidate
from sklearn.cluster import KMeans
import torch

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def classify_headings(lines):
    # Filter heading candidates (short lines)
    candidates = [line for line in lines if is_heading_candidate(line)]
    if not candidates:
        return []

    # Compute embeddings
    texts = [c["text"] for c in candidates]
    embeddings = model.encode(texts, convert_to_tensor=True)

    # Cluster headings into 3 levels (H1, H2, H3)
    num_clusters = 3 if len(candidates) >= 3 else len(candidates)
    if num_clusters < 1:
        return []

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings.cpu().numpy())

    # Rank clusters by average font size (assumes larger font = higher heading)
    cluster_to_fontsize = {}
    for i, label in enumerate(cluster_labels):
        cluster_to_fontsize.setdefault(label, []).append(candidates[i]["font_size"])
    cluster_avg_size = {
        label: sum(sizes) / len(sizes) for label, sizes in cluster_to_fontsize.items()
    }
    sorted_clusters = sorted(cluster_avg_size.items(), key=lambda x: -x[1])

    # Map cluster index to heading level
    level_map = {}
    for i, (cluster_idx, _) in enumerate(sorted_clusters):
        level_map[cluster_idx] = f"H{i+1}"

    # Build outline with heading levels
    outline = []
    for i, candidate in enumerate(candidates):
        outline.append({
            "level": level_map[cluster_labels[i]],
            "text": candidate["text"],
            "page": candidate["page"]
        })

    return outline