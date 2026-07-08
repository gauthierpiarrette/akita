import numpy as np
import pytest
import tenseal as ts

from akita import PipelineSpec, PlanError, plan, run_column_scoring, run_matvec


def test_the_trap_is_real():
    """The bug the planner exists to prevent: an unpadded 384-dim matmul in
    TenSEAL silently returns garbage — no exception, no warning. This is
    not a contrived case; it bit this repo's own prototype on day one."""
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    rng = np.random.default_rng(0)
    q = rng.standard_normal(384)
    M = rng.standard_normal((384, 4096))  # full slot width, as in production
    out = np.array(ts.ckks_vector(ctx, q.tolist()).matmul(M.tolist()).decrypt())
    assert np.max(np.abs(out - q @ M)) > 0.1  # silently, badly wrong


def test_planner_autopads_the_trap_away():
    p = plan(PipelineSpec("matvec_ranking", dim=384, n_items=4096))
    assert p.padded_dim == 512
    assert 4096 % p.padded_dim == 0


def test_matvec_end_to_end_exact():
    rng = np.random.default_rng(1)
    n, dim = 4096, 384
    matrix = rng.standard_normal((n, dim))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = rng.standard_normal(dim)
    query /= np.linalg.norm(query)

    p = plan(PipelineSpec("matvec_ranking", dim=dim, n_items=n))
    result = run_matvec(p, query, matrix)
    ref = matrix @ query
    assert np.max(np.abs(result["scores"] - ref)) < 1e-4
    top = np.argsort(result["scores"])[::-1][:10]
    assert set(top.tolist()) == set(np.argsort(ref)[::-1][:10].tolist())


def test_column_scoring_end_to_end_exact():
    rng = np.random.default_rng(2)
    n, d = 4096, 32
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    p = plan(PipelineSpec("column_scoring", dim=d, n_items=n))
    result = run_column_scoring(p, X, w, b=0.3)
    assert np.max(np.abs(result["logits"] - (X @ w + 0.3))) < 1e-4


def test_oversized_dim_rejected():
    with pytest.raises(PlanError, match="slots"):
        plan(PipelineSpec("matvec_ranking", dim=5000, n_items=1000))


def test_comparison_routed_away():
    with pytest.raises(PlanError, match="TFHE"):
        plan(PipelineSpec("comparison", dim=1, n_items=1))


def test_every_plan_carries_the_disclosure_warning():
    for wl, dim in (("matvec_ranking", 384), ("column_scoring", 32)):
        p = plan(PipelineSpec(wl, dim=dim, n_items=4096))
        assert any("IND-CPA-D" in w for w in p.warnings)


def test_cost_model_within_tolerance_of_scale_test():
    """The 100k-doc scale test measured 14.0 s wall on 10 cores
    (results/demo5_scale_test.json). The model must stay within 30%."""
    p = plan(PipelineSpec("matvec_ranking", dim=384, n_items=102_400, cores=10))
    assert p.chunks == 25
    assert abs(p.est_wall_seconds - 14.0) / 14.0 < 0.30
