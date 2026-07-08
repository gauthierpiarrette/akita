# Akita — computing on data no one is allowed to see

![tests](https://github.com/gauthierpiarrette/akita/actions/workflows/tests.yml/badge.svg)

The "100,000× overhead" of fully homomorphic encryption is a choice, not a
constant. This repo demonstrates it four ways:

1. **Akita Private Memory** (`akita/memory.py`) — a working SDK giving an
   AI app per-user memory that the operator provably cannot read, with
   sub-second semantic recall over encrypted embeddings.
2. **A planner** (`akita/planner.py`) that validates and costs encrypted
   computation *before* it runs — and catches a class of silent-corruption
   bugs that today's tooling doesn't.
3. **A working private search engine** (`demos/private_rag/`) — real
   embeddings, real client/server split, results exactly identical to
   plaintext, on a corpus the server ranks for queries it can never read.
4. **Measured micro-benchmarks** (`demos/`, `results/`) showing the
   overhead spans five orders of magnitude *on the same laptop* depending
   only on workload shape.

All numbers measured on an Apple M1 Pro; every one traces to a script and
a JSON file in `results/`. The technical reasoning and sources are in
[`docs/why.md`](docs/why.md).

**Status**: v0.1 prototype. Real cryptography, honest measurements,
**not audited** — don't protect production data with it yet (see
[`SECURITY.md`](SECURITY.md)).

## The planner: know the cost before you encrypt

```python
from akita import PipelineSpec, plan

print(plan(PipelineSpec("matvec_ranking", dim=384, n_items=1_000_000)).explain())
```
```
Akita plan — matvec_ranking
  security   CKKS, N=8192, modulus 200 bits (<= 218 for 128-bit), depth 1
  layout     dim 384 -> padded 512 (validated: divides 4096 slots), 245 chunk(s) x 4096 items
  est compute  639.74 core-s (127.95 s wall on 10 cores)
  est traffic  326 KB up / 59,976 KB down (+ 33.8 MB one-time public context)
  est cost     $0.008885 per query at full utilization ($0.017771 on a dedicated 10-core box) @ $0.05/core-hr
  warning    CKKS results must be revealed only to the secret-key holder. ...
```

Why this matters — two facts we hit building this repo:

- **Silent corruption is one layout mistake away.** A 384-dim encrypted
  matmul in TenSEAL returns *wrong scores with no error* because 384
  doesn't divide the slot count (`tests/test_planner.py::test_the_trap_is_real`
  reproduces it). The planner makes that layout unrepresentable.
- **The cost model holds.** The planner predicted 13.1 s for a 100k-doc
  encrypted search this laptop measured at 14.0 s — within 7%
  (`test_cost_model_within_tolerance_of_scale_test`).

v0 plans the two workload shapes measured here (`matvec_ranking`,
`column_scoring`), refuses what shouldn't run under CKKS (comparisons →
routed to TFHE or the client), and executes plans on TenSEAL via
`run_matvec` / `run_column_scoring`. Cost constants are calibrated on this
machine; expect ±30% elsewhere until re-calibrated.

## Akita Private Memory: an AI's memory no operator can read

```python
from akita import Memory, MemoryServer, MiniLMEmbedder

mem = Memory.open("user1", passphrase, server, MiniLMEmbedder())
mem.add(["dr chen increased the propranolol dose to 80mg", ...])
mem.search("what medication changes has my doctor made?", k=3)
```

The trust boundary: text is embedded **on the key holder's side**, sealed
with AES-GCM, and its embeddings are CKKS-encrypted before anything
leaves. The server (`MemoryServer`) stores blobs it cannot open and
answers searches with one ciphertext×ciphertext multiply per packed
segment — no rotations, so enrollment ships a **1.9 MB** public context,
not 34 MB. Keys derive from a passphrase (scrypt) and never leave the
keystore; `wipe()` is crypto-shredding.

Measured (demo 6, 400 sensitive notes, real MiniLM embeddings, M1 Pro):

| Metric | Value |
|---|---|
| Store 400 notes (embed + encrypt + upload) | 0.8–2.4 s (~2–6 ms/note) |
| Semantic search, end to end | **0.24 s median** |
| Server compute per search (encrypted corpus) | 145 ms — planner predicted 145 ms |
| Storage at rest | 45.5 KB/memory (18.6 MB total) |
| Delete | tombstone effective immediately; `compact()` re-packs physically |
| Operator's view | no secret key (asserted at enrollment + auditable), hex-opaque blobs and segments |

Honest v0 costs, stated plainly: the search download scales with corpus
size (~230 KB per 8-memory segment — server-side aggregation is the
roadmap fix); fetch patterns are padded with decoy reads pending a PIR
stage; and key loss means memory loss — that's what "the operator cannot
read it" costs. Run it: `.venv/bin/python demos/demo6_private_memory.py`.

**See it**: `.venv/bin/python demos/demo_page/app.py` →
http://127.0.0.1:8642 — a live split-screen of both sides of the trust
boundary: your device (ask anything, ~95 ms warm end-to-end) next to the
operator's view (hex blobs, ciphertext segments, `secret key: NO`, and
which fetches it saw — winners and decoys, indistinguishable). Real
cryptography, no mocks.

## The working prototype: private semantic search over HTTP

A real client/server system on real data: 12,288 documents (20 Newsgroups)
embedded with MiniLM-L6-v2 (384-dim). The client owns the secret key,
embeds and encrypts its query locally, and sends only a ciphertext; the
server — holding a public context it verifiably cannot decrypt with —
ranks the entire corpus and returns encrypted scores.

Measured across 8 natural-language queries:

| Metric | Value |
|---|---|
| Top-10 results vs plaintext search | **80/80 exactly identical** |
| Median end-to-end latency | **2.26 s** (1.68 s server compute, 10 threads) |
| Traffic per query | 326 KB up / 690 KB down |
| One-time enrollment (public keys) | 35.5 MB |
| Server overhead vs plaintext matvec | ~1,459× — and the answer still costs ~$0.0002 |

At scale (measured, synthetic embeddings, same kernel): 102,400 docs in
14.0 s on this laptop ≈ **$0.002/query**; a derived 1M-doc corpus is
~$0.02/query but ~21 s on a 64-core server — honest evidence that linear
scan stops being interactive around 10⁵–10⁶ docs. Production systems
shard into clusters and search one shard (exactly Apple's Wally design),
which is a planner problem, not a cryptography problem.

**Costing conventions** (so the numbers above can't be misread): scale-test
and prototype figures price a *dedicated* machine for the query's wall
time — the conservative upper bound. The planner also reports the *full
utilization* basis (core-seconds priced directly), which is what batch
serving achieves. The two bracket real deployments and differ by ~2× on
this machine; the planner prints both.

```bash
.venv/bin/python demos/private_rag/build_index.py   # embeds the corpus (~2 min)
.venv/bin/python demos/private_rag/server.py &      # untrusted server
.venv/bin/python demos/private_rag/client.py        # data owner
```

## The reframe in one paragraph

The famous "100,000× overhead" of homomorphic encryption is a category
error twice over. First, it prices one encrypted op against one plaintext
op — but modern lattice schemes are SIMD machines (one ciphertext operation
processes thousands of values), so the *amortized* overhead is a function
of workload shape, not a constant: we measure ~330× for packed
rotation-free linear algebra, ~6,600× for rotation-heavy linear algebra,
and ~10⁸× only at exact comparisons. Second, overhead multipliers are the
wrong unit entirely — what decides viability is **dollars per answer vs
value per answer**, and for retrieval, scoring, matching, and aggregation
over sensitive data, the encrypted cost per answer is already far below the
value of the answer.

## Measured micro-benchmarks (single M1 Pro core, pinned to 1 thread)

| Demo | Workload | Server compute | Overhead vs plaintext | Cost |
|---|---|---|---|---|
| 1. Private search | encrypted query ranked against 16,384 docs (dim 128) | 2.6 s (0.54 s on 10 cores) | ~6,600× | $0.0022 / 1M docs |
| 2. Encrypted scoring | logistic model on 4,096 encrypted records (32 feat) | 9.4 ms | ~330× | $0.00003 / 1M records |
| 3. Exact rules (TFHE) | AML rule (2 comparisons + AND/OR) on encrypted txn | 11.3 s / decision | ~10⁸× | route around it |

Correctness: demo 1 top-10 ranking identical to plaintext (max score error
5×10⁻⁸); demo 2 max probability error 4×10⁻⁸; demo 3 exact by construction.
In every demo the server holds a **public context only** — the scripts
assert it cannot decrypt.

**Baseline caveat**: overheads are measured against single-thread numpy.
A production SIMD/ANN baseline (FAISS-class) is several times faster than
naive numpy, so multipliers vs *best* plaintext are ~2–5× higher than the
table shows. The dollars-per-answer conclusion is unchanged — that's the
point of pricing answers instead of multipliers.

The five-order spread between the demos is the load-bearing observation:
**the overhead multiplier is set by workload shape — packing layout,
rotation count, and where the nonlinearities go — not by the cryptography.**
Moving a stage from the expensive class to the cheap class is worth more
than the last five years of hardware progress. That routing/packing layer
is where the planner (and eventually a real compiler) lives.

## Security notes & limitations — read before relying on this

- **Parameters**: all CKKS contexts use N=8192 with ≤200-bit total
  coefficient modulus — within the 218-bit cap for 128-bit security in the
  homomorphicencryption.org standard. TFHE (Concrete) uses its defaults
  (p_fail ≤ 2⁻¹²⁸-class, post-2024 hardened).
- **IND-CPA-D / result disclosure**: CKKS is approximate. Decrypted
  results are safe to use *by the key holder*; revealing them to other
  parties (including the server) enables key-recovery-style attacks
  unless noise flooding is applied (Li–Micciancio, and the 2024 CCS line
  of work). Every plan the planner emits carries this warning. The demos
  here never disclose results beyond the key holder.
- **Access patterns**: in the prototype, fetching the top-k documents by
  ID reveals *which* documents matched (never the query or the scores).
  Closing this requires a PIR fetch stage — production-proven elsewhere
  (Apple), not yet implemented here.
- **Model extraction** (scoring workloads): decrypted scores leak model
  information over many queries; rate-limit as you would any ML API.
- **Integrity**: FHE guarantees privacy, not that the server ran the
  right function. Verifiable FHE is still research.
- **This is a prototype**: TenSEAL/SEAL and Concrete are real libraries,
  but this repo's glue code is unaudited. Do not deploy as-is.

## Layout

- `akita/` — the planner + TenSEAL runtime (`pip install -e .`)
- `tests/` — including the silent-corruption reproduction and the
  cost-model validation (`python -m pytest tests/ -q`)
- `demos/demo1_private_search.py` — CKKS encrypted retrieval, measured
- `demos/demo2_encrypted_scoring.py` — CKKS column-packed scoring, measured
- `demos/demo3_exact_rules.py` — TFHE exact compliance logic, measured
- `demos/private_rag/` — the end-to-end private search engine (build
  index, server, client, scale test)
- `demos/demo_page/` — the Private Memory demo page (both sides of the
  trust boundary, live, in a browser)
- `demos/demo6_private_memory.py` — Private Memory measured end to end
- `results/` — every number in this README, as JSON
- `docs/why.md` — the technical reasoning, with sources

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[demos,dev]"
```

or with plain pip: `python3.12 -m venv .venv && .venv/bin/pip install -e ".[demos,dev]"`.

Exact versions every number in this README was produced with are pinned
in `requirements.lock.txt` (Python 3.12.13, macOS 14.6, Apple M1 Pro).
Requires Python 3.10–3.12 (TenSEAL has no 3.13+ wheels yet).

## Contributing, security, roadmap

Contributions that don't require cryptography expertise are the most
valuable right now — adapters, embedders, clients, and benchmarks on
your hardware: see [`CONTRIBUTING.md`](CONTRIBUTING.md). Vulnerabilities:
privately, per [`SECURITY.md`](SECURITY.md). Where this is going (and
what it deliberately isn't): [`ROADMAP.md`](ROADMAP.md). License:
Apache-2.0.
