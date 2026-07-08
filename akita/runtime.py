"""
Execute a validated Plan on TenSEAL and report measured-vs-estimated.

The runtime enforces the plan's trust invariants: computation happens
against a context stripped of the secret key, and decryption uses the
key holder's context only.
"""

from __future__ import annotations

import time

import numpy as np
import tenseal as ts

from .planner import Plan, SLOTS, POLY_DEGREE


def _contexts(plan: Plan, n_threads: int | None):
    kwargs = {"n_threads": n_threads} if n_threads else {}
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_DEGREE,
        coeff_mod_bit_sizes=plan.coeff_mod_bit_sizes,
        **kwargs,
    )
    ctx.global_scale = 2**40
    if plan.needs_galois_keys:
        ctx.generate_galois_keys()
    server = ts.context_from(ctx.serialize(save_secret_key=False), **kwargs)
    if server.is_private():
        raise RuntimeError("server context must hold no secret key")
    return ctx, server


def run_matvec(plan: Plan, query: np.ndarray, matrix: np.ndarray,
               n_threads: int | None = None) -> dict:
    """query: (dim,); matrix: (n_items, dim). Returns scores + timings."""
    assert plan.spec.workload == "matvec_ranking"
    dim, n = plan.spec.dim, plan.spec.n_items
    assert query.shape == (dim,) and matrix.shape == (n, dim)

    client, server = _contexts(plan, n_threads)

    padded = np.zeros(plan.padded_dim)
    padded[:dim] = query
    t0 = time.perf_counter()
    ct = ts.ckks_vector(client, padded.tolist()).serialize()
    encrypt_s = time.perf_counter() - t0

    enc = ts.ckks_vector_from(server, ct)
    upload_kb = len(ct) / 1024
    results, download_kb = [], 0.0
    t0 = time.perf_counter()
    for s in range(0, n, SLOTS):
        block = matrix[s : s + SLOTS]
        padded_block = np.zeros((plan.padded_dim, block.shape[0]))
        padded_block[:dim] = block.T
        out = enc.matmul(padded_block.tolist()).serialize()
        download_kb += len(out) / 1024
        results.append(out)
    compute_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = np.concatenate([
        ts.ckks_vector_from(client, r).decrypt(client.secret_key())
        for r in results
    ])[:n]
    decrypt_s = time.perf_counter() - t0

    return {
        "scores": scores,
        "measured": {
            "encrypt_s": encrypt_s,
            "compute_s": compute_s,
            "decrypt_s": decrypt_s,
            "upload_kb": upload_kb,
            "download_kb": download_kb,
        },
        "estimated": {
            "compute_wall_s": plan.est_wall_seconds
            if n_threads is None else plan.est_core_seconds,
            "upload_kb": plan.est_upload_kb,
            "download_kb": plan.est_download_kb,
        },
    }


def run_column_scoring(plan: Plan, X: np.ndarray, w: np.ndarray, b: float,
                       n_threads: int | None = None) -> dict:
    """X: (batch, features); w: (features,). Returns logits + timings."""
    assert plan.spec.workload == "column_scoring"
    n, d = plan.spec.n_items, plan.spec.dim
    assert X.shape == (n, d) and w.shape == (d,)
    assert n <= SLOTS, "v0 runtime: one batch per call"

    client, server = _contexts(plan, n_threads)

    t0 = time.perf_counter()
    cols = [ts.ckks_vector(client, X[:, j].tolist()).serialize() for j in range(d)]
    encrypt_s = time.perf_counter() - t0
    upload_kb = sum(len(c) for c in cols) / 1024

    enc_cols = [ts.ckks_vector_from(server, c) for c in cols]
    t0 = time.perf_counter()
    acc = enc_cols[0] * float(w[0])
    for j in range(1, d):
        acc += enc_cols[j] * float(w[j])
    acc += float(b)
    compute_s = time.perf_counter() - t0
    out = acc.serialize()

    t0 = time.perf_counter()
    logits = np.array(
        ts.ckks_vector_from(client, out).decrypt(client.secret_key())
    )[:n]
    decrypt_s = time.perf_counter() - t0

    return {
        "logits": logits,
        "measured": {
            "encrypt_s": encrypt_s,
            "compute_s": compute_s,
            "decrypt_s": decrypt_s,
            "upload_kb": upload_kb,
            "download_kb": len(out) / 1024,
        },
        "estimated": {
            "compute_wall_s": plan.est_wall_seconds
            if n_threads is None else plan.est_core_seconds,
            "upload_kb": plan.est_upload_kb,
            "download_kb": plan.est_download_kb,
        },
    }
