"""
Private RAG prototype — the client (data owner).

Owns the secret key. Embeds natural-language queries locally, encrypts
them, sends ciphertexts to the server, decrypts the returned scores, and
ranks locally. The server never sees the query, the scores, or the
ranking.

Measures the full round trip against the same server's plaintext search,
and verifies the encrypted top-10 matches the plaintext top-10 exactly.
"""

import base64
import json
import statistics
import time
from pathlib import Path

import numpy as np
import requests
import tenseal as ts
from sentence_transformers import SentenceTransformer

SERVER = "http://127.0.0.1:8631"
RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
PAD = 512

QUERIES = [
    "problems with my graphics card drivers on windows",
    "space shuttle launches and NASA missions",
    "gun control legislation and the second amendment",
    "hockey playoffs and the Stanley Cup",
    "encryption, privacy and the government's clipper chip",
    "advice on motorcycle riding and safety gear",
    "medical treatment and clinical studies for chronic disease",
    "arab israeli conflict and the peace process",
]


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Keygen + one-time enrollment: server gets the PUBLIC context only.
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    pub = ctx.serialize(save_secret_key=False)
    info = requests.post(f"{SERVER}/setup", data=pub).json()
    print(f"enrolled: {info['docs']} docs, dim {info['dim']}, "
          f"public context {len(pub)/1e6:.1f} MB (one-time)\n")

    per_query, agreements = [], []
    for q in QUERIES:
        t0 = time.perf_counter()
        vec = model.encode([q], normalize_embeddings=True)[0].astype(np.float64)
        embed_s = time.perf_counter() - t0

        qp = np.zeros(PAD)
        qp[: len(vec)] = vec
        t0 = time.perf_counter()
        ct = ts.ckks_vector(ctx, qp.tolist()).serialize()
        encrypt_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        resp = requests.post(f"{SERVER}/search", data=ct).json()
        roundtrip_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = np.concatenate([
            ts.ckks_vector_from(ctx, base64.b64decode(c))
              .decrypt(ctx.secret_key())
            for c in resp["chunks"]
        ])
        decrypt_s = time.perf_counter() - t0
        top10 = np.argsort(scores)[::-1][:10]

        plain = requests.post(
            f"{SERVER}/plaintext_search", json={"vector": vec.tolist()}
        ).json()
        agree = len(set(top10.tolist()) & set(plain["top"]))
        agreements.append(agree)

        total = embed_s + encrypt_s + roundtrip_s + decrypt_s
        per_query.append({
            "query": q,
            "total_s": total,
            "embed_s": embed_s,
            "encrypt_s": encrypt_s,
            "roundtrip_s": roundtrip_s,
            "server_compute_s": resp["server_compute_s"],
            "decrypt_s": decrypt_s,
            "plaintext_server_s": plain["server_compute_s"],
            "top10_agreement": agree,
            "upload_KB": len(ct) / 1024,
            "download_KB": sum(len(c) for c in resp["chunks"]) * 0.75 / 1024,
        })

        best = requests.get(f"{SERVER}/doc/{int(top10[0])}").json()["text"]
        snippet = " ".join(best.split())[:110]
        print(f"[{total:5.2f}s | top10 {agree}/10] {q}\n"
              f"          -> {snippet}...\n")

    med = lambda k: statistics.median(r[k] for r in per_query)
    summary = {
        "corpus": f"{info['docs']} real docs (20 Newsgroups), "
                  f"MiniLM-L6-v2 {info['dim']}-dim embeddings",
        "median_end_to_end_s": round(med("total_s"), 2),
        "median_breakdown_s": {
            "embed_query_local": round(med("embed_s"), 3),
            "encrypt": round(med("encrypt_s"), 3),
            "http_roundtrip_incl_server": round(med("roundtrip_s"), 3),
            "server_encrypted_compute": round(med("server_compute_s"), 3),
            "decrypt_and_rank_local": round(med("decrypt_s"), 3),
        },
        "median_plaintext_server_s": round(med("plaintext_server_s"), 5),
        "server_overhead_vs_plaintext": round(
            med("server_compute_s") / med("plaintext_server_s")
        ),
        "upload_KB_per_query": round(med("upload_KB")),
        "download_KB_per_query": round(med("download_KB")),
        "public_context_MB_one_time": round(len(pub) / 1e6, 1),
        "top10_exact_across_queries": f"{sum(agreements)}/{10 * len(QUERIES)}",
        "queries": per_query,
    }
    (RESULTS / "demo4_private_rag.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "queries"}, indent=2))


if __name__ == "__main__":
    main()
