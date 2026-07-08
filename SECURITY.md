# Security Policy

## Status: prototype — not audited

Akita is a research prototype. It has **not** undergone an independent
security audit. Do not use it to protect production data yet. Known
limitations are documented in the README ("Security notes &
limitations") and are part of the public record on purpose.

What is already in place:

- CKKS parameters within the homomorphicencryption.org 128-bit budget
  (N=8192, total modulus ≤ 200 bits); TFHE via Concrete defaults
  (post-2024 hardened).
- The server refuses enrollment of any context containing a secret key
  (explicit exception — no `assert`s on the trust path), and
  `Memory.audit()` re-fetches the server-held context and verifies
  client-side that it cannot decrypt and still hashes to what was
  enrolled.
- Tamper evidence: every stored blob is AEAD-bound to its blob id, and
  the encrypted index carries a version counter pinned on the client —
  substituted, reordered, or rolled-back server data fails closed
  (`IntegrityError`). Caveat: a total server wipe is indistinguishable
  from crypto-shredding; rollback detection covers partial rollbacks.
- Keystore: scrypt (n=2^17, r=8, p=1) wraps the CKKS secret context and
  the AES key; KDF parameters are stored in the keystore (older files
  upgrade transparently), and key files are written owner-only (0600).
- Decrypted CKKS results are never disclosed beyond the key holder in
  any shipped flow (IND-CPA-D discipline).

Known residual leakage (documented, not hidden): storage/traffic
metadata (including blob sizes, which track note lengths), and blob
fetch patterns (mitigated with decoys; a PIR fetch stage is on the
roadmap).

## Reporting a vulnerability

Please report privately — do not open a public issue:

- GitHub: use **"Report a vulnerability"** (Security tab → private
  advisory) on this repository, or
- Email: gauthierpiarrette@gmail.com with subject `[AKITA SECURITY]`.

You'll get an acknowledgment within 72 hours. Please include a
reproduction if you can. Credit given in the fix notes unless you
prefer otherwise.
