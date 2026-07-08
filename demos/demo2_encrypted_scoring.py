"""
Demo 2 — Encrypted batch scoring: a vendor scores records it can never see.

The inverse trust direction from demo 1:
  - The DATA OWNER (bank, hospital) encrypts a batch of records
    column-wise: ciphertext j holds feature j for B records at once.
  - The VENDOR holds a proprietary model (weights in plaintext, its IP)
    and scores the whole batch under encryption:
        score = sum_j  w_j * ct_j  + b
    Only scalar multiplications and additions — no rotations, no
    bootstrapping, multiplicative depth 1.
  - The data owner decrypts B scores and applies the sigmoid locally.

The packing is the entire trick: one ciphertext operation processes
4,096 records simultaneously, so the per-record overhead collapses.
This is logistic regression (credit risk, readmission risk, fraud
propensity) — the workhorse of regulated-industry ML.
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

N_FEATURES = 32
BATCH = 4096                     # records packed per ciphertext (slots)
CLOUD_USD_PER_CORE_HOUR = 0.05


def main():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((BATCH, N_FEATURES))      # sensitive records
    w = rng.standard_normal(N_FEATURES)               # vendor's model
    b = 0.3

    # ---- DATA OWNER: keygen + column-packed encryption ----------------
    t0 = time.perf_counter()
    # n_threads=1 pins SEAL to one core for an honest per-core comparison.
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 60],
        n_threads=1,
    )
    ctx.global_scale = 2**40
    keygen_s = time.perf_counter() - t0

    server_ctx = ts.context_from(ctx.serialize(save_secret_key=False), n_threads=1)
    assert not server_ctx.is_private()

    t0 = time.perf_counter()
    col_bytes = [
        ts.ckks_vector(ctx, X[:, j].tolist()).serialize()
        for j in range(N_FEATURES)
    ]
    encrypt_s = time.perf_counter() - t0
    upload_mb = sum(len(c) for c in col_bytes) / 1024 / 1024

    # ---- VENDOR: encrypted linear scoring ------------------------------
    cols = [ts.ckks_vector_from(server_ctx, c) for c in col_bytes]
    t0 = time.perf_counter()
    acc = cols[0] * float(w[0])
    for j in range(1, N_FEATURES):
        acc += cols[j] * float(w[j])
    acc += b
    server_s = time.perf_counter() - t0

    # ---- DATA OWNER: decrypt + sigmoid locally -------------------------
    t0 = time.perf_counter()
    logits = np.array(acc.decrypt(ctx.secret_key()))
    probs = 1.0 / (1.0 + np.exp(-logits))
    decrypt_s = time.perf_counter() - t0

    # ---- Plaintext baseline (best of 10, warm) + correctness -----------
    plain_s = float("inf")
    for _ in range(10):
        t0 = time.perf_counter()
        ref_logits = X @ w + b
        plain_s = min(plain_s, max(time.perf_counter() - t0, 1e-9))
    ref_probs = 1.0 / (1.0 + np.exp(-ref_logits))

    per_record_us = server_s / BATCH * 1e6
    out = {
        "workload": f"logistic scoring, {N_FEATURES} features, batch {BATCH}",
        "keygen_s": round(keygen_s, 3),
        "encrypt_s": round(encrypt_s, 3),
        "upload_MB_per_batch": round(upload_mb, 1),
        "server_compute_s_per_batch": round(server_s, 4),
        "server_us_per_record": round(per_record_us, 1),
        "decrypt_s": round(decrypt_s, 4),
        "plaintext_baseline_s": round(plain_s, 6),
        "overhead_multiplier_vs_plaintext": round(server_s / plain_s),
        "usd_per_1M_records_scored": round(
            server_s / BATCH * 1e6 / 3600 * CLOUD_USD_PER_CORE_HOUR, 6
        ),
        "max_abs_prob_error": float(np.max(np.abs(probs - ref_probs))),
    }
    (RESULTS / "demo2_encrypted_scoring.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
