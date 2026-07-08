# Contributing to Akita

Thanks for looking. This project is young; useful contributions don't
require cryptography expertise — most of the valuable surface is
integrations, benchmarks, and clients.

## Dev setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[demos,dev]"
.venv/bin/python -m pytest tests/ -q       # should be all green
```

(Plain `python3.12 -m venv` + `pip install -e ".[demos,dev]"` works too.)

## Ground rules

- **Every performance claim ships with the script and JSON that produced
  it** (see `results/`). No unmeasured numbers in docs or docstrings.
- **Correctness is verified against the plaintext pipeline** — see
  `tests/test_memory.py::test_search_matches_plaintext_pipeline_exactly`
  for the pattern.
- Security-relevant changes (parameters, key handling, protocol) need
  an explicit rationale in the PR and get extra scrutiny.
- Match the existing code style; keep modules small.

## Contributions we'd especially welcome

- **Benchmarks on your hardware** — run the demos, open a PR adding your
  `results/*.json` with CPU model noted. Cross-hardware cost surfaces
  make the planner better for everyone.
- **Framework adapters** — LangChain / LlamaIndex retriever wrappers.
- **Embedder backends** — ONNX/quantized clients, other models.
- **Client ports** — WASM / Swift / Kotlin (the embedder + AES + CKKS
  client side is what needs to run on end-user devices).
- **OpenFHE backend** — see `ROADMAP.md` for why and when.
- Docs, examples, and anything that makes the five-minute path smoother.

Open an issue before large changes so we don't duplicate work.
