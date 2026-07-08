"""
Demo 3 — Exact decisions on encrypted data (TFHE, via Zama's Concrete).

CKKS (demos 1-2) does fast approximate linear algebra but cannot branch:
no comparisons, no thresholds, no exact logic. TFHE is its complement —
it computes *exact* functions of encrypted integers via programmable
bootstrapping, at a much higher per-operation price.

Workload: an anti-money-laundering style rule evaluated by a compliance
service on a transaction it can never see:

    flag = (amount > 9000) OR (risk_score > 200 AND country_risk >= 4)

The transaction fields stay encrypted end to end; the service returns an
encrypted 0/1 flag. This is the piece that makes encrypted *pipelines*
complete: CKKS for the wide linear math, TFHE for the exact decisions —
scheme selection per stage, which is the actual architecture of every
serious FHE deployment today.
"""

import json
import time
from pathlib import Path

import numpy as np
from concrete import fhe

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def aml_rule(amount, risk_score, country_risk):
    large_cash = amount > 9000
    risky_profile = (risk_score > 200) & (country_risk >= 4)
    return large_cash | risky_profile


def main():
    compiler = fhe.Compiler(
        aml_rule,
        {"amount": "encrypted", "risk_score": "encrypted", "country_risk": "encrypted"},
    )
    rng = np.random.default_rng(3)
    inputset = [
        (int(rng.integers(0, 20000)), int(rng.integers(0, 256)), int(rng.integers(0, 8)))
        for _ in range(200)
    ]

    t0 = time.perf_counter()
    circuit = compiler.compile(inputset)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    circuit.keygen()
    keygen_s = time.perf_counter() - t0

    cases = [
        (12000, 10, 1),   # large cash          -> flag
        (500, 250, 5),    # risky profile       -> flag
        (500, 250, 2),    # risky score, ok geo -> no flag
        (8999, 10, 1),    # clean               -> no flag
    ]
    run_times, all_correct = [], True
    for amount, score, geo in cases:
        enc = circuit.encrypt(amount, score, geo)
        t0 = time.perf_counter()
        enc_flag = circuit.run(enc)
        run_times.append(time.perf_counter() - t0)
        flag = circuit.decrypt(enc_flag)
        expected = int(aml_rule(np.int64(amount), np.int64(score), np.int64(geo)))
        all_correct &= (int(flag) == expected)

    # Plaintext baseline: the same rule as interpreted Python.
    reps = 1_000_000
    t0 = time.perf_counter()
    for _ in range(reps):
        aml_rule(12000, 10, 1)
    plain_s = (time.perf_counter() - t0) / reps

    eval_s = float(np.mean(run_times))
    out = {
        "workload": "AML rule on encrypted transaction (2 comparisons + AND/OR)",
        "compile_s": round(compile_s, 2),
        "keygen_s": round(keygen_s, 2),
        "eval_s_per_transaction": round(eval_s, 3),
        "plaintext_baseline_ns": round(plain_s * 1e9),
        "overhead_multiplier_vs_plaintext": round(eval_s / plain_s),
        "all_decisions_exact": bool(all_correct),
        "note": "TFHE cost is per-decision, ~8 orders above plaintext on a CPU "
                "core; use it only at the decision points, CKKS everywhere else.",
    }
    (RESULTS / "demo3_exact_rules.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
