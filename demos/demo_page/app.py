"""
Akita Private Memory — the demo page.

One command, one browser tab, zero mocks: this serves a real Memory
(real MiniLM embeddings, real CKKS ciphertexts, real AES blobs) and
renders both sides of the trust boundary at once — what the key holder
experiences, and the hex the operator is left with.

    .venv/bin/python demos/demo_page/app.py
    -> http://127.0.0.1:8642
"""

import shutil
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from akita import Memory, MemoryServer, MiniLMEmbedder

HERE = Path(__file__).resolve().parent
KEYDIR = HERE / ".demo_keys"

SEED_NOTES = [
    "dr chen increased the propranolol dose to 80mg after the last episode",
    "migraine again after the night shift, lasted 12 hours",
    "physio says the knee is healing, cleared for light running",
    "blood work came back, a1c slightly elevated, retest in 3 months",
    "lawyer silva says the settlement offer of 120k expires on the 14th",
    "signed the NDA with meridian, cannot discuss the acquisition until Q3",
    "custody mediation moved to the 23rd, need the school records",
    "wire of 45k to the singapore account flagged, called the bank",
    "accountant estimates 85k in capital gains tax this year",
    "mortgage refinance approved at 4.1 percent",
    "recruiter from halcyon called about the staff engineer role, 300k base",
    "performance review: exceeds, but comp frozen this cycle",
    "the northgate offer includes equity, 120k over four years",
    "mom's memory is getting worse, called the clinic about donepezil",
    "sister's surgery was rescheduled to the 14th, taking the week off",
    "dad finally agreed to the assisted living tour on the 9th",
    "dentist found a cracked molar, crown needed",
    "pharmacist warned about mixing ibuprofen with the new meds",
    "therapist said the panic attacks are tied to the deadline cycles",
    "hr confirmed the parental leave starts in march",
]

app = Flask(__name__)

# fresh demo state on every start
shutil.rmtree(KEYDIR, ignore_errors=True)
server = MemoryServer()
mem = Memory.open("demo_user", "demo-passphrase", server,
                  MiniLMEmbedder(), store_dir=KEYDIR)
mem.add(SEED_NOTES)


def operator_view() -> dict:
    """Everything the server can see about this user. All of it."""
    u = server._users["demo_user"]
    audit = server.audit("demo_user")
    blobs = [
        {"id": bid, "bytes": len(b), "hex": b[:18].hex()}
        for bid, b in list(u["blobs"].items())
    ]
    segments = [
        {"segment": i, "kb": round(len(s) / 1024), "hex": s[:18].hex()}
        for i, s in enumerate(u["segments"])
    ]
    return {
        "secret_key_held": audit["context_is_private"],
        "context_sha256": audit["context_sha256"][:16],
        "public_context_mb": round(len(u["ctx_bytes"]) / 1e6, 2),
        "storage_mb": round(audit["storage_bytes"] / 1e6, 1),
        "blob_fetches_observed": audit["blob_fetches_observed"],
        "blobs": blobs,
        "segments": segments,
    }


@app.get("/")
def index():
    return send_file(HERE / "index.html")


@app.get("/api/operator")
def api_operator():
    return jsonify(operator_view())


@app.post("/api/add")
def api_add():
    text = request.get_json()["text"].strip()
    if not text:
        return jsonify({"error": "empty"}), 400
    t0 = time.perf_counter()
    ids = mem.add([text])
    return jsonify({
        "id": ids[0],
        "add_ms": round((time.perf_counter() - t0) * 1000),
        "operator": operator_view(),
    })


@app.post("/api/search")
def api_search():
    query = request.get_json()["query"].strip()
    trace: dict = {}
    t0 = time.perf_counter()
    results = mem.search(query, k=3, decoys=2, trace=trace)
    e2e_ms = round((time.perf_counter() - t0) * 1000)
    return jsonify({
        "results": [{"id": r["id"], "score": round(r["score"], 3),
                     "text": r["text"]} for r in results],
        "e2e_ms": e2e_ms,
        "trace": {
            "embed_ms": round(trace["embed_s"] * 1000),
            "encrypt_ms": round(trace["encrypt_s"] * 1000),
            "server_ms": round(trace["server_roundtrip_s"] * 1000),
            "decrypt_ms": round(trace["decrypt_and_rank_s"] * 1000),
            "query_ct_kb": trace["query_ct_kb"],
            "query_ct_hex": trace["query_ct_hex"],
            "download_kb": trace["download_kb"],
            "segments": trace["segments"],
            "fetched_ids": trace["fetched_ids"],
        },
        "operator": operator_view(),
    })


if __name__ == "__main__":
    print("Akita Private Memory demo -> http://127.0.0.1:8642")
    app.run(host="127.0.0.1", port=8642, threaded=False)
