from akita import PipelineSpec, plan


def test_memory_search_plan_shape():
    p = plan(PipelineSpec("memory_search", dim=384, n_items=1000))
    assert p.padded_dim == 512
    assert p.chunks == 125            # ceil(1000 / 8 chunks per ct)
    assert p.needs_galois_keys is False
    assert p.one_time_context_mb < 5  # relin only, not 34 MB of galois
    assert any("scales with corpus" in w for w in p.warnings)
    # 125 segments x 1.9 ms measured kernel ~ 0.24 core-seconds
    assert 0.1 < p.est_core_seconds < 0.5
