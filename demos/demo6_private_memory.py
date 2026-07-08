"""
Demo 6 — Akita Private Memory: an AI's memory of you, readable by no one.

A simulated year of one person's sensitive notes (health, legal, money,
work, family) goes into a Memory whose operator provably cannot read it:
text is AES-sealed, embeddings are CKKS-encrypted, keys never leave the
"device". Natural-language recall works anyway — and the demo prints
what the operator actually sees, which is the product pitch in one
screenful of hex.
"""

import json
import time
from pathlib import Path

import numpy as np

from akita import Memory, MemoryServer, MiniLMEmbedder, PipelineSpec, plan

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

THEMES = {
    "health": [
        "dr {n} increased the {drug} dose to {mg}mg after the last episode",
        "migraine again after the night shift, lasted {h} hours",
        "physio says the {joint} is healing, cleared for light running",
        "blood work came back, {marker} slightly elevated, retest in {m} months",
    ],
    "legal": [
        "lawyer {n} says the settlement offer of {amt}k expires on the {d}th",
        "signed the NDA with {co}, cannot discuss the acquisition until Q{q}",
        "custody mediation moved to the {d}th, need the school records",
        "the {co} contract has a non-compete clause, 18 months",
    ],
    "money": [
        "wire of {amt}k to the {place} account flagged, called the bank",
        "accountant estimates {amt}k in capital gains tax this year",
        "mortgage refinance approved at {pct} percent",
        "sold the {co} shares, proceeds {amt}k, half to the index fund",
    ],
    "work": [
        "recruiter from {co} called about the {role} role, {amt}k base",
        "told {n} about the reorg before the announcement, keep quiet",
        "performance review: exceeds, but comp frozen this cycle",
        "the {co} offer includes equity, {amt}k over four years",
    ],
    "family": [
        "{n}'s surgery rescheduled to the {d}th, taking the week off",
        "mom's memory is getting worse, called the clinic about {drug}",
        "{n} failed the midterm, meeting the teacher on the {d}th",
        "dad finally agreed to the assisted living tour on the {d}th",
    ],
}
FILL = {
    "n": ["chen", "moreau", "okafor", "silva", "novak"],
    "drug": ["propranolol", "metformin", "sertraline", "donepezil"],
    "mg": ["40", "80", "120"], "h": ["6", "12", "36"],
    "joint": ["knee", "shoulder", "achilles"],
    "marker": ["a1c", "ldl", "cortisol"], "m": ["3", "6"],
    "amt": ["45", "120", "300", "85"], "d": ["9", "14", "23"],
    "co": ["vertex", "meridian", "halcyon", "northgate"],
    "q": ["2", "3"], "pct": ["4.1", "5.3"],
    "place": ["singapore", "zurich", "dubai"],
    "role": ["quant", "staff engineer", "head of data"],
}

QUERIES = [
    "what medication changes has my doctor made?",
    "when does the settlement offer expire?",
    "what happened with the flagged international wire transfer?",
    "what do I know about the job offers and compensation?",
    "what's going on with my mother's health?",
]


def make_notes(n=400, seed=42):
    rng = np.random.default_rng(seed)
    notes = []
    themes = list(THEMES)
    while len(notes) < n:
        t = themes[len(notes) % len(themes)]
        tpl = THEMES[t][rng.integers(len(THEMES[t]))]
        note = tpl.format(**{k: v[rng.integers(len(v))] for k, v in FILL.items()})
        notes.append(note)
    return notes


def main():
    notes = make_notes()
    embedder = MiniLMEmbedder()
    server = MemoryServer()

    t0 = time.perf_counter()
    mem = Memory.open("demo_user", "correct horse battery staple",
                      server, embedder, store_dir=".akita_keys")
    open_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    mem.add(notes)
    add_s = time.perf_counter() - t0

    p = plan(PipelineSpec("memory_search", dim=embedder.dim, n_items=len(notes)))

    print(f"stored {len(notes)} sensitive notes in {add_s:.1f}s "
          f"(embed+encrypt+upload)\n")

    searches = []
    for q in QUERIES:
        t0 = time.perf_counter()
        results = mem.search(q, k=3)
        e2e = time.perf_counter() - t0
        server_s = server._users["demo_user"]["last_search_s"]
        searches.append({"query": q, "e2e_s": round(e2e, 3),
                         "server_s": round(server_s, 4),
                         "top": results[0]["text"]})
        print(f"[{e2e:.2f}s | server {server_s*1000:.0f}ms] {q}")
        for r in results:
            print(f"    {r['score']:+.3f}  {r['text']}")
        print()

    # ---- what the operator sees ---------------------------------------
    audit = mem.audit()
    a_blob = next(iter(server._users["demo_user"]["blobs"].values()))
    print("=== the operator's view of this user ===")
    print(f"  secret key held server-side : {audit['context_is_private']}")
    print(f"  a stored memory (hex)       : {a_blob[:36].hex()}...")
    print(f"  an embedding segment (hex)  : {audit['sample_ciphertext_hex']}...")
    print(f"  storage for {audit['blobs']} blobs + {audit['segments']} segments: "
          f"{audit['storage_bytes']/1e6:.1f} MB")

    # ---- delete + crypto-shredding proof -------------------------------
    victim = mem.search("settlement offer expiry", k=1)[0]
    mem.delete(victim["id"])
    still = [r["id"] for r in mem.search("settlement offer expiry", k=10)]
    mem.compact()

    out = {
        "notes": len(notes),
        "embedder": "all-MiniLM-L6-v2 (384-dim, runs client-side)",
        "open_s_keygen": round(open_s, 2),
        "add_s_total": round(add_s, 1),
        "add_ms_per_note": round(add_s / len(notes) * 1000, 1),
        "median_search_e2e_s": round(
            float(np.median([s["e2e_s"] for s in searches])), 3),
        "median_server_compute_s": round(
            float(np.median([s["server_s"] for s in searches])), 4),
        "planner_estimate_server_s": round(p.est_core_seconds, 4),
        "public_context_mb_one_time": round(
            len(server._users["demo_user"]["ctx_bytes"]) / 1e6, 2),
        "server_storage_mb": round(audit["storage_bytes"] / 1e6, 1),
        "kb_per_memory_at_rest": round(
            audit["storage_bytes"] / len(notes) / 1024, 1),
        "delete_effective_immediately": victim["id"] not in still,
        "searches": searches,
    }
    (RESULTS / "demo6_private_memory.json").write_text(json.dumps(out, indent=2))
    print("\n" + json.dumps({k: v for k, v in out.items() if k != "searches"},
                            indent=2))


if __name__ == "__main__":
    main()
