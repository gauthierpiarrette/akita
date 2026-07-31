# Changelog

## 0.2.0 — 2026-07-31

Security hardening release, prompted by an internal audit. The public API
is unchanged for typical use; two behaviors are stricter (noted below).

- **No `assert`s on the trust path.** Every security check is an explicit
  exception and survives `python -O`: the server refuses enrollment of a
  context containing a secret key, and refuses re-enrolling an existing
  user with a different context (anti-hijack).
- **Client-verified audit.** `Memory.audit()` now re-fetches the
  server-held context and verifies client-side that it contains no
  secret key and still hashes to what was enrolled — the report no
  longer relies on the server's word.
- **Tamper evidence.** Every stored blob is AEAD-bound to its blob id
  and user; the encrypted index carries a version counter pinned in a
  local state file. Substituted, reordered, missing, or rolled-back
  server data raises `IntegrityError` (new, exported). Caveat: a total
  server wipe is indistinguishable from crypto-shredding; rollback
  detection covers partial rollbacks.
- **Keystore hardening.** scrypt work factor raised to n=2^17 (r=8, p=1,
  per OWASP); KDF parameters now stored in a versioned keystore, so
  pre-0.2 keystores upgrade transparently on first successful open. Key
  and state files are written owner-only (0600), the store directory
  0700.
- **CSPRNG decoys.** Decoy fetch selection uses `secrets.SystemRandom`.
- **CI supply chain.** GitHub Actions pinned to commit SHAs,
  least-privilege `permissions:`, release-environment gating on publish,
  Dependabot enabled.

Behavior changes: a wrong passphrase now raises `ValueError` (was
`cryptography`'s `InvalidTag`), and re-enrolling an existing user with a
different context now raises `ValueError` instead of silently
overwriting.

## 0.1.0 — 2026-07-08

Initial public release: encrypted per-user memory (`Memory`,
`MemoryServer`) over CKKS + AES-GCM, the cost planner (`plan`,
`PipelineSpec`), embedders, measured demos, and the private-RAG demo
server.
