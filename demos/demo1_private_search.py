"""
Demo 1 — Private semantic search: the server ranks documents for a query
it can never see.

Threat model demonstrated here:
  - The CLIENT owns the secret key. It encrypts its query embedding.
  - The SERVER holds only a *public* context (no secret key) and the
    plaintext document embeddings (its own corpus — not sensitive to it).
  - The server computes  scores = q_enc @ D  entirely under encryption
    and returns encrypted scores. It learns neither the query nor which
    documents matched.
  - The client decrypts locally and takes the top-k.

This is the exact workload shape Apple ships in production (Live Caller
ID Lookup, Enhanced Visual Search): one wide, shallow linear pass over a
corpus with an encrypted query — no bootstrapping, multiplicative depth 1.
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # single-thread numpy
os.environ.setdefault("OMP_NUM_THREADS", "1")         # for a fair baseline

import numpy as np
import tenseal as ts

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

DIM = 128            # embedding dimension
CHUNK = 4096         # docs per ciphertext op (= slot count at poly 8192)
N_DOCS = 16384       # total corpus size for this run
TOP_K = 10
CLOUD_USD_PER_CORE_HOUR = 0.05  # ~on-demand c-family vCPU price


def make_corpus(n_docs: int, dim: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    docs = rng.standard_normal((n_docs, dim))
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    query = rng.standard_normal(dim)
    query /= np.linalg.norm(query)
    return docs, query


def main():
    docs, query = make_corpus(N_DOCS, DIM)

    # ---- CLIENT: keygen + encrypt query ------------------------------
    # n_threads=1 pins SEAL to a single core so the overhead-vs-plaintext
    # comparison is apples-to-apples; throughput scales ~linearly with
    # cores (the corpus shards are independent).
    t0 = time.perf_counter()
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
        n_threads=1,
    )
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    keygen_s = time.perf_counter() - t0

    # What the server receives: a public context only.
    server_ctx_bytes = ctx.serialize(save_secret_key=False)
    server_ctx = ts.context_from(server_ctx_bytes, n_threads=1)
    assert not server_ctx.is_private(), "server context must not hold the secret key"

    t0 = time.perf_counter()
    q_enc_bytes = ts.ckks_vector(ctx, query.tolist()).serialize()
    encrypt_s = time.perf_counter() - t0

    # ---- SERVER: encrypted matvec over the corpus --------------------
    q_enc = ts.ckks_vector_from(server_ctx, q_enc_bytes)
    enc_score_chunks = []
    t0 = time.perf_counter()
    for start in range(0, N_DOCS, CHUNK):
        block = docs[start : start + CHUNK]          # (chunk, DIM)
        enc_score_chunks.append(q_enc.matmul(block.T.tolist()))
    server_s = time.perf_counter() - t0

    # ---- CLIENT: decrypt scores, rank locally ------------------------
    t0 = time.perf_counter()
    scores_enc = np.concatenate(
        [np.array(c.decrypt(ctx.secret_key())) for c in enc_score_chunks]
    )
    decrypt_s = time.perf_counter() - t0

    # ---- Plaintext baseline (best of 10, warm) + correctness ---------
    plain_s = float("inf")
    for _ in range(10):
        t0 = time.perf_counter()
        scores_plain = docs @ query
        plain_s = min(plain_s, max(time.perf_counter() - t0, 1e-9))

    top_enc = set(np.argsort(scores_enc)[::-1][:TOP_K].tolist())
    top_plain = set(np.argsort(scores_plain)[::-1][:TOP_K].tolist())
    max_err = float(np.max(np.abs(scores_enc - scores_plain)))

    out = {
        "workload": f"encrypted query vs {N_DOCS} docs, dim {DIM}",
        "keygen_s": round(keygen_s, 3),
        "encrypt_s": round(encrypt_s, 4),
        "server_compute_s": round(server_s, 3),
        "decrypt_s": round(decrypt_s, 4),
        "plaintext_baseline_s": round(plain_s, 6),
        "overhead_multiplier_vs_plaintext": round(server_s / plain_s),
        "docs_per_second_per_core": round(N_DOCS / server_s),
        "usd_per_1M_docs_searched": round(
            (server_s / N_DOCS) * 1e6 / 3600 * CLOUD_USD_PER_CORE_HOUR, 4
        ),
        "query_ciphertext_KB": round(len(q_enc_bytes) / 1024, 1),
        "server_context_MB_one_time": round(len(server_ctx_bytes) / 1024 / 1024, 1),
        "max_abs_score_error": max_err,
        "top10_agreement": f"{len(top_enc & top_plain)}/{TOP_K}",
    }
    (RESULTS / "demo1_private_search.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
