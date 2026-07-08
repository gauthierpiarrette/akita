# Roadmap

Ordered by intent, not promise. Items move when users pull them.

## Near term

- **CI** (GitHub Actions: pytest on 3.10–3.12) and tagged releases.
- **LangChain / LlamaIndex retriever adapters** — one import away from
  existing RAG stacks.
- **Demo assets** — recorded demo-page walkthrough in the README.

## Kernel work

- **Server-side aggregation for memory search.** The v0 kernel returns
  one product ciphertext per 8-memory segment (~230 KB each), so
  download scales with corpus size. Rotation-based aggregation
  server-side cuts this ~8–64×. Blocked on backend rotations (below).
- **PIR fetch stage** — close the access-pattern leak (currently
  mitigated with decoy fetches).
- **Batch/multi-query packing** for higher search throughput.

## Backend

- **OpenFHE migration.** TenSEAL sits on Microsoft SEAL, which is in
  maintenance mode; TenSEAL also exposes no rotation API (the reason
  for the v0 kernel design) and has a silent-corruption footgun we
  guard in the planner. OpenFHE is actively maintained, has full
  rotations, and tracked the 2024 IND-CPA-D hardening. Caveats: its
  PyPI wheel is currently Linux-x86_64-only (macOS needs a source
  build), and SEAL/OpenFHE ciphertext formats are incompatible, so the
  migration is all-at-once, with every benchmark re-measured. Trigger
  conditions: aggregation kernel needed at scale, a SEAL-side security
  event, or a funded/community milestone.

## Clients

- **WASM client** (browser) and **Swift/Kotlin clients** (mobile) — the
  embedder + AES + CKKS encryption must run on end-user devices for the
  trust model to hold end to end.
- Key backup/recovery flows (passphrase-wrapped export, platform
  keystores).

## Trust

- **Independent security audit** — targeted once grant funding or
  sponsorship covers it; until then the prototype banner stays.
- Reproducibility: cross-hardware benchmark submissions welcome (see
  CONTRIBUTING).

## Explicitly not planned

- Chat-speed encrypted LLM inference claims (revisit when FHE ASICs are
  purchasable).
- Encrypted training.
- General encrypted SQL (joins/GROUP BY belong to the expensive class;
  we route around comparisons, not through them).
