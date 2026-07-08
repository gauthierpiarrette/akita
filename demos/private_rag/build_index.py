"""
Private RAG prototype — index builder.

Embeds a real corpus (20 Newsgroups, 12,288 documents) with a real
sentence-transformer (all-MiniLM-L6-v2, 384-dim) and saves the plaintext
index the server will hold. The corpus is the *server's* asset here; it is
the client's queries that are sensitive and stay encrypted end to end.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)

N_DOCS = 12288  # 3 chunks of 4096 (CKKS slot count at poly degree 8192)


def main():
    raw = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    docs = [d.strip() for d in raw.data if len(d.strip()) > 200][:N_DOCS]
    assert len(docs) == N_DOCS, f"only {len(docs)} docs available"

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(
        docs, batch_size=256, normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float64)

    np.save(DATA / "embeddings.npy", emb)
    (DATA / "texts.json").write_text(json.dumps(docs))
    print(f"indexed {len(docs)} docs, embeddings {emb.shape} -> {DATA}")


if __name__ == "__main__":
    main()
