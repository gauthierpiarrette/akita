"""
The Akita planner: decide *whether and how* to run a workload under
encryption before spending a single ciphertext operation.

Everything an FHE deployment gets wrong is decided before execution:
packing layout, padding, parameter budget, rotation count. Today those
decisions are made by hand, and the tooling fails silently when they're
wrong (see tests/test_planner.py::test_the_trap_is_real — an unpadded
384-dim matmul in TenSEAL returns garbage with no error). The planner
makes them explicit, validated, and costed.

v0 scope — the two workload shapes measured in this repo:

  matvec_ranking    encrypted vector × plaintext matrix
                    (retrieval, semantic search, PIR-style ranking)
  column_scoring    plaintext weights × encrypted feature columns
                    (batch scoring: credit, fraud, risk)

Cost model constants are calibrated on an Apple M1 Pro (see demos/ and
results/); treat estimates as ±30% until re-calibrated on your host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------- constants

POLY_DEGREE = 8192
SLOTS = POLY_DEGREE // 2

# Security: 128-bit per the homomorphicencryption.org standard, which caps
# the total coefficient modulus at N=8192 to 218 bits.
MAX_MODULUS_BITS_128 = 218

# Calibrated on M1 Pro (single core), from results/*.json:
MATVEC_S_PER_DIM_PER_CHUNK = 0.0051   # demo1: 0.65 s/chunk at padded dim 128
COLSCORE_S_PER_FEATURE = 0.00029      # demo2: 9.4 ms per 32-feature batch
MEMSEARCH_S_PER_SEGMENT = 0.0029      # ct-by-ct mult incl (de)serialization
KB_PER_PRIME = 81.6                   # serialized ciphertext KB per RNS prime
PUBLIC_CONTEXT_MB_GALOIS = 33.8       # context incl. galois keys (matvec)
PUBLIC_CONTEXT_MB_PLAIN = 1.1         # context without galois (column scoring)
PUBLIC_CONTEXT_MB_RELIN = 1.9         # context with relin keys (memory_search)
PARALLEL_EFFICIENCY = 0.5             # measured 4.9x speedup on 10 cores

IND_CPAD_WARNING = (
    "CKKS results must be revealed only to the secret-key holder. "
    "Disclosing decrypted results to other parties (including the server) "
    "requires noise-flooding countermeasures (IND-CPA-D, Li-Micciancio)."
)


class PlanError(ValueError):
    """A workload that must not be run as specified."""


# ------------------------------------------------------------------- specs

@dataclass
class PipelineSpec:
    """What you want to compute — declared, not implemented.

    workload: 'matvec_ranking' or 'column_scoring'
    dim:      vector dimension (embedding size / feature count)
    n_items:  corpus size (matvec) or batch size (column_scoring)
    """

    workload: str
    dim: int
    n_items: int
    cores: int = 10
    usd_per_core_hour: float = 0.05


@dataclass
class Plan:
    """A validated, costed execution plan."""

    spec: PipelineSpec
    padded_dim: int
    chunks: int
    coeff_mod_bit_sizes: list[int] = field(default_factory=list)
    needs_galois_keys: bool = False
    est_core_seconds: float = 0.0
    est_wall_seconds: float = 0.0
    est_upload_kb: float = 0.0
    est_download_kb: float = 0.0
    # Two costing bases, stated explicitly so they can't be confused:
    # utilization = core-seconds priced directly (batch serving, full machine
    # utilization); dedicated = the whole machine for the query's wall time
    # (latency-sensitive single-query serving, the conservative upper bound).
    est_usd_per_query: float = 0.0
    est_usd_per_query_dedicated: float = 0.0
    one_time_context_mb: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def explain(self) -> str:
        s = self.spec
        lines = [
            f"Akita plan — {s.workload}",
            f"  security   CKKS, N={POLY_DEGREE}, "
            f"modulus {sum(self.coeff_mod_bit_sizes)} bits "
            f"(<= {MAX_MODULUS_BITS_128} for 128-bit), depth 1",
            f"  layout     dim {s.dim} -> padded {self.padded_dim} "
            f"(validated: divides {SLOTS} slots), "
            f"{self.chunks} chunk(s) x {SLOTS} items",
            f"  est compute  {self.est_core_seconds:.2f} core-s "
            f"({self.est_wall_seconds:.2f} s wall on {s.cores} cores)",
            f"  est traffic  {self.est_upload_kb:,.0f} KB up / "
            f"{self.est_download_kb:,.0f} KB down "
            f"(+ {self.one_time_context_mb} MB one-time public context)",
            f"  est cost     ${self.est_usd_per_query:.6f} per query at full "
            f"utilization (${self.est_usd_per_query_dedicated:.6f} on a "
            f"dedicated {s.cores}-core box) @ ${s.usd_per_core_hour}/core-hr",
        ]
        for w in self.warnings:
            lines.append(f"  warning    {w}")
        return "\n".join(lines)


# ----------------------------------------------------------------- planner

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def plan(spec: PipelineSpec) -> Plan:
    if spec.dim < 1 or spec.n_items < 1:
        raise PlanError("dim and n_items must be positive")

    if spec.workload == "matvec_ranking":
        return _plan_matvec(spec)
    if spec.workload == "column_scoring":
        return _plan_column_scoring(spec)
    if spec.workload == "memory_search":
        return _plan_memory_search(spec)
    if spec.workload in ("comparison", "threshold", "argmax", "branch"):
        raise PlanError(
            f"'{spec.workload}' needs exact logic, which CKKS cannot do and "
            "TFHE does at ~10^6-10^8x overhead (measured: 11.3 s/decision on "
            "one CPU core). Route it: decrypt-at-the-edge if the client may "
            "see the value, or a sparse TFHE decision stage if not."
        )
    raise PlanError(f"unknown workload '{spec.workload}'")


def _check_modulus(bits: list[int]) -> None:
    if sum(bits) > MAX_MODULUS_BITS_128:
        raise PlanError(
            f"coefficient modulus {sum(bits)} bits exceeds the "
            f"{MAX_MODULUS_BITS_128}-bit budget for 128-bit security at "
            f"N={POLY_DEGREE}"
        )


def _plan_matvec(spec: PipelineSpec) -> Plan:
    padded = _next_pow2(spec.dim)
    if padded > SLOTS:
        raise PlanError(
            f"dim {spec.dim} (padded {padded}) exceeds {SLOTS} slots at "
            f"N={POLY_DEGREE}; v0 does not split query vectors. Reduce the "
            "embedding dimension (PCA/Matryoshka) or use larger parameters."
        )
    # The rule that makes silent corruption impossible: the packed vector
    # length must divide the slot count exactly (see the trap test).
    assert SLOTS % padded == 0

    bits = [60, 40, 40, 60]
    _check_modulus(bits)

    chunks = math.ceil(spec.n_items / SLOTS)
    core_s = MATVEC_S_PER_DIM_PER_CHUNK * padded * chunks
    wall_s = core_s / max(spec.cores * PARALLEL_EFFICIENCY, 1.0)
    upload = KB_PER_PRIME * len(bits)                 # one fresh query ct
    download = KB_PER_PRIME * (len(bits) - 1) * chunks  # 1 level consumed

    return Plan(
        spec=spec,
        padded_dim=padded,
        chunks=chunks,
        coeff_mod_bit_sizes=bits,
        needs_galois_keys=True,
        est_core_seconds=core_s,
        est_wall_seconds=wall_s,
        est_upload_kb=upload,
        est_download_kb=download,
        est_usd_per_query=core_s / 3600 * spec.usd_per_core_hour,
        est_usd_per_query_dedicated=wall_s * spec.cores / 3600
        * spec.usd_per_core_hour,
        one_time_context_mb=PUBLIC_CONTEXT_MB_GALOIS,
        warnings=[IND_CPAD_WARNING],
    )


def _plan_memory_search(spec: PipelineSpec) -> Plan:
    """Encrypted query against an ENCRYPTED corpus (Akita Private Memory).

    v0 kernel: one ct-by-ct elementwise multiply per packed segment on
    the server (no rotations, so no galois keys); the key holder decrypts
    the products and block-sums locally. Honest cost: the download scales
    with corpus size — the roadmap fix is server-side aggregation.
    """
    padded = _next_pow2(spec.dim)
    if padded > SLOTS:
        raise PlanError(
            f"dim {spec.dim} (padded {padded}) exceeds {SLOTS} slots at "
            f"N={POLY_DEGREE}; reduce the embedding dimension."
        )
    assert SLOTS % padded == 0

    bits = [60, 40, 40, 60]
    _check_modulus(bits)

    chunks_per_ct = SLOTS // padded
    segments = math.ceil(spec.n_items / chunks_per_ct)
    core_s = MEMSEARCH_S_PER_SEGMENT * segments
    wall_s = core_s / max(spec.cores * PARALLEL_EFFICIENCY, 1.0)
    upload = KB_PER_PRIME * len(bits)                     # one query ct
    download = KB_PER_PRIME * (len(bits) - 1) * segments  # product cts

    return Plan(
        spec=spec,
        padded_dim=padded,
        chunks=segments,
        coeff_mod_bit_sizes=bits,
        needs_galois_keys=False,
        est_core_seconds=core_s,
        est_wall_seconds=wall_s,
        est_upload_kb=upload,
        est_download_kb=download,
        est_usd_per_query=core_s / 3600 * spec.usd_per_core_hour,
        est_usd_per_query_dedicated=wall_s * spec.cores / 3600
        * spec.usd_per_core_hour,
        one_time_context_mb=PUBLIC_CONTEXT_MB_RELIN,
        warnings=[
            IND_CPAD_WARNING,
            f"download is {KB_PER_PRIME * (len(bits) - 1) * segments:,.0f} KB "
            "and scales with corpus size (v0 kernel returns per-segment "
            "products); server-side aggregation is the roadmap fix",
        ],
    )


def _plan_column_scoring(spec: PipelineSpec) -> Plan:
    if spec.dim > 4096:
        raise PlanError(
            f"{spec.dim} feature columns means {spec.dim} ciphertexts per "
            "batch; above ~4096 the upload dominates. Reduce features or "
            "pack multiple features per ciphertext (v1)."
        )
    bits = [60, 40, 60]
    _check_modulus(bits)

    batches = math.ceil(spec.n_items / SLOTS)
    core_s = COLSCORE_S_PER_FEATURE * spec.dim * batches
    wall_s = core_s / max(spec.cores * PARALLEL_EFFICIENCY, 1.0)
    upload = KB_PER_PRIME * len(bits) * spec.dim * batches
    download = KB_PER_PRIME * (len(bits) - 1) * batches

    return Plan(
        spec=spec,
        padded_dim=spec.dim,
        chunks=batches,
        coeff_mod_bit_sizes=bits,
        needs_galois_keys=False,
        est_core_seconds=core_s,
        est_wall_seconds=wall_s,
        est_upload_kb=upload,
        est_download_kb=download,
        est_usd_per_query=core_s / 3600 * spec.usd_per_core_hour,
        est_usd_per_query_dedicated=wall_s * spec.cores / 3600
        * spec.usd_per_core_hour,
        one_time_context_mb=PUBLIC_CONTEXT_MB_PLAIN,
        warnings=[
            IND_CPAD_WARNING,
            "returned scores can leak the model over many queries; "
            "rate-limit as you would any ML API",
        ],
    )
