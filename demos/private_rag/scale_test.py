"""
Private search scale test: how do latency and cost grow with corpus size?

Runs the same encrypted-ranking kernel as the prototype (CKKS, 384-dim
queries padded to 512, 4,096-doc chunks) over a 102,400-doc corpus —
8.3x the prototype — and derives the economics of a 1M-doc deployment.

Embeddings are synthetic here: the crypto kernel is oblivious to what the
vectors mean, and result-exactness vs plaintext was already established
on real embeddings in the prototype. This test isolates throughput.
"""

import json
import time
from pathlib import Path

import numpy as np
import tenseal as ts

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

DIM, PAD, CHUNK = 384, 512, 4096
N_DOCS = 102_400  # 25 chunks
N_QUERIES = 3
CLOUD_USD_PER_CORE_HOUR = 0.05
LAPTOP_CORES = 10


def main():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()

    rng = np.random.default_rng(1)

    print(f"pre-encoding {N_DOCS} docs into {N_DOCS // CHUNK} chunk matrices...")
    t0 = time.perf_counter()
    matrices = []
    for _ in range(N_DOCS // CHUNK):
        block = rng.standard_normal((CHUNK, DIM))
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        padded = np.zeros((PAD, CHUNK))
        padded[:DIM] = block.T
        matrices.append(padded.tolist())
    setup_s = time.perf_counter() - t0
    print(f"setup {setup_s:.1f}s")

    latencies = []
    for i in range(N_QUERIES):
        q = rng.standard_normal(DIM)
        q /= np.linalg.norm(q)
        qp = np.zeros(PAD)
        qp[:DIM] = q
        enc = ts.ckks_vector(ctx, qp.tolist())

        t0 = time.perf_counter()
        _ = [enc.matmul(m) for m in matrices]
        dt = time.perf_counter() - t0
        latencies.append(dt)
        print(f"query {i + 1}: {dt:.2f}s ({N_DOCS / dt:,.0f} docs/s on {LAPTOP_CORES} cores)")

    lat = float(np.median(latencies))
    core_s_per_query = lat * LAPTOP_CORES  # upper bound: all cores busy
    usd_per_query = core_s_per_query / 3600 * CLOUD_USD_PER_CORE_HOUR
    docs_per_core_s = N_DOCS / core_s_per_query

    out = {
        "corpus_docs": N_DOCS,
        "chunks": N_DOCS // CHUNK,
        "median_query_latency_s_laptop": round(lat, 2),
        "docs_per_second_whole_laptop": round(N_DOCS / lat),
        "docs_per_core_second_upper_bound_cost": round(docs_per_core_s),
        "usd_per_query_100k_docs": round(usd_per_query, 5),
        "derived_1M_docs": {
            "usd_per_query": round(usd_per_query * 1e6 / N_DOCS, 4),
            "latency_s_on_64_core_server": round(
                (1e6 / N_DOCS) * core_s_per_query / 64, 1
            ),
            "note": "chunks are independent; latency divides by cores/machines",
        },
        "setup_s_one_time": round(setup_s, 1),
    }
    (RESULTS / "demo5_scale_test.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
