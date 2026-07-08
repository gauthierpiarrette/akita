# Why encrypted computation is more viable than its reputation

*The technical reasoning behind this repo, with sources. All local
measurements: Apple M1 Pro, single core unless noted, exact versions in
`requirements.lock.txt`.*

## The unit-of-analysis problem

The famous "100,000× overhead" of fully homomorphic encryption prices
one encrypted operation against one plaintext operation. Two things are
off about that unit:

1. Modern lattice schemes are SIMD machines — one ciphertext operation
   processes thousands of packed values — so *amortized* overhead
   depends on how well the workload packs, not on a fixed constant.
2. Nobody buys operations. Workloads produce *answers* (a ranked list,
   a score, a flag), and the practical question is whether the cost per
   answer is below the value of the answer.

## Where the overhead actually comes from

| Source | Mechanism | Fundamental? |
|---|---|---|
| Ciphertext expansion | a scalar becomes a degree-2¹³–2¹⁶ polynomial | Mostly no — packed, expansion is ~40–120×. A layout problem. |
| Arithmetic blowup | every op runs NTTs, O(N log N) modular mults | Partly — amortized over slots ~10²×; rotations add ~10× and rotation count is a layout choice. |
| Noise budget | each multiply consumes modulus; deep circuits bootstrap | Workload-shaped — shallow circuits never bootstrap; depth comes from nonlinearities, which can often be routed elsewhere. |
| Memory bandwidth | ~1 op/byte intensity, keys in the 100s of MB | The real floor — why CPUs sit at 10⁴–10⁵× and GPUs near 10³×, and what FHE ASICs attack (Intel's Heracles demo at ISSCC 2026: 1,074–5,547× over a 24-core Xeon on FHE kernels). |

Three of the four knobs live in software. That is the claim this repo
tests empirically.

## What we measured (same laptop, same 128-bit security)

- Packed, rotation-free linear scoring: **~330×** (2.3 µs/record)
- Rotation-heavy encrypted matvec: **~6,600×** (160 µs/doc)
- Exact comparison logic (TFHE): **~10⁸×** (11.3 s/decision vs 59 ns)
- End to end: private semantic search over 12,288 real documents —
  top-10 identical to plaintext, 2.26 s median over HTTP; and per-user
  encrypted memory (query *and* corpus encrypted) — ~95 ms warm recall
  over 400 notes.

A spread of more than five orders of magnitude, determined by workload
shape alone. Independent anchors are consistent: encrypted ResNet-20 at
~10³× on GPU (Cheddar, arXiv:2407.13055), encrypted transformers around
10⁴× (Llama-3-8B fully under CKKS at 18–22 s/token on 8 GPUs,
arXiv:2601.18511), TFHE bootstrapping under 1 ms on H100 (Zama, vendor
benchmark with published parameters).

In dollars per answer, the cheap class is already economically
invisible: our measured costs are $0.0022 per million documents
searched and $0.00003 per million records scored. Production evidence
that this class is real: Apple ships BFV-based private lookups to
iPhones at ~3,000 QPS (Wally, arXiv:2406.06761); Microsoft ships
homomorphic password checking in Edge.

## What this does not show

- **Chat-speed encrypted LLM inference** — still ~10³–10⁴× on GPUs;
  plausibly waits for dedicated silicon.
- **Encrypted training** — no practical path known.
- **Arbitrary SQL / joins / branching** — comparisons are the expensive
  class; systems that work route around them (client-side decisions,
  sparse TFHE stages) rather than through them.
- **Security shortcuts** — CKKS decryptions must stay with the key
  holder unless noise flooding is applied (IND-CPA-D, Li–Micciancio and
  the 2024 CCS line); parameters here follow the
  homomorphicencryption.org 128-bit budgets; access patterns leak
  unless a PIR fetch stage is added. See the README's security notes.

## Key sources

Wally (arXiv:2406.06761) · Cheddar (arXiv:2407.13055) · encrypted
Llama-3-8B (arXiv:2601.18511) · NEXUS (eprint 2024/136) · Intel
Heracles at ISSCC 2026 (IEEE Spectrum coverage) · Zama TFHE-rs
benchmarks (docs.zama.org) · Apple: machinelearning.apple.com/research/
homomorphic-encryption · homomorphicencryption.org security standard.
