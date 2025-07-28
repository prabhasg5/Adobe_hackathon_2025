from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import numpy as np

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
def cluster_lines_by_semantics(lines, n_clusters=3):
    texts = [line["text"] for line in lines]
    fonts = [line["font_size"] for line in lines]
    pages = [line["page"] for line in lines]

    embeddings = model.encode(texts, convert_to_tensor=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings.cpu().numpy())
    cluster_fonts = {i: [] for i in range(n_clusters)}
    for label, font in zip(labels, fonts):
        cluster_fonts[label].append(font)
    
    cluster_avg_size = {
        k: sum(v) / len(v) for k, v in cluster_fonts.items()
    }
    cluster_order = {
        cluster: f"H{i+1}"
        for i, (cluster, _) in enumerate(
            sorted(cluster_avg_size.items(), key=lambda x: -x[1])
        )
    }

    result = []
    for text, font, page, label in zip(texts, fonts, pages, labels):
        result.append({
            "level": cluster_order[label],
            "text": text.strip(),
            "page": page
        })

    return result
