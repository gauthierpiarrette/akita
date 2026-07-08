"""
Private RAG prototype — the untrusted server.

Holds the plaintext document index (its own asset) and a PUBLIC CKKS
context uploaded by the client at /setup. It can rank documents for an
encrypted query but can never decrypt the query or the scores.

Endpoints:
  POST /setup             client's public context (binary body)
  POST /search            one encrypted query vector (binary body) ->
                          JSON {chunks: [b64 ciphertexts], server_compute_s}
  POST /plaintext_search  JSON {vector: [...]} -> top-k (benchmark baseline)
  GET  /doc/<i>           document text (production would use PIR here;
                          the *scores* leak nothing, the fetch is the part
                          that still needs a private-retrieval stage)
"""

import base64
import json
import time
from pathlib import Path

import numpy as np
import tenseal as ts
from flask import Flask, request, jsonify

DATA = Path(__file__).resolve().parent / "data"
CHUNK = 4096
PAD = 512  # query dim padded to a power of two dividing the slot count

app = Flask(__name__)

EMB = np.load(DATA / "embeddings.npy")          # (N, 384) plaintext index
TEXTS = json.loads((DATA / "texts.json").read_text())
N, DIM = EMB.shape

# Pre-transposed, zero-padded chunk matrices for ct-pt matmul.
MATRICES = []
for s in range(0, N, CHUNK):
    block = EMB[s : s + CHUNK]                  # (CHUNK, DIM)
    padded = np.zeros((PAD, block.shape[0]))
    padded[:DIM] = block.T
    MATRICES.append(padded.tolist())

STATE = {"ctx": None}


@app.post("/setup")
def setup():
    ctx = ts.context_from(request.get_data())
    if ctx.is_private():
        return jsonify(
            {"error": "refusing a context that contains a secret key"}
        ), 400
    STATE["ctx"] = ctx
    return jsonify({"ok": True, "docs": N, "dim": DIM})


@app.post("/search")
def search():
    if STATE["ctx"] is None:
        return jsonify({"error": "no public context: call /setup first"}), 400
    enc_query = ts.ckks_vector_from(STATE["ctx"], request.get_data())
    t0 = time.perf_counter()
    enc_chunks = [enc_query.matmul(m) for m in MATRICES]
    compute_s = time.perf_counter() - t0
    return jsonify(
        {
            "chunks": [
                base64.b64encode(c.serialize()).decode() for c in enc_chunks
            ],
            "server_compute_s": compute_s,
        }
    )


@app.post("/plaintext_search")
def plaintext_search():
    q = np.array(request.get_json()["vector"])
    t0 = time.perf_counter()
    scores = EMB @ q
    top = np.argsort(scores)[::-1][:10]
    compute_s = time.perf_counter() - t0
    return jsonify(
        {
            "top": top.tolist(),
            "scores": scores[top].tolist(),
            "server_compute_s": compute_s,
        }
    )


@app.get("/doc/<int:i>")
def doc(i: int):
    return jsonify({"text": TEXTS[i][:400]})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8631, threaded=False)
