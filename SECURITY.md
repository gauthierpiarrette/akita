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
- The server-side context is asserted public at enrollment and
  verifiable via `Memory.audit()`.
- Decrypted CKKS results are never disclosed beyond the key holder in
  any shipped flow (IND-CPA-D discipline).

Known residual leakage (documented, not hidden): storage/traffic
metadata, and blob fetch patterns (mitigated with decoys; a PIR fetch
stage is on the roadmap).

## Reporting a vulnerability

Please report privately — do not open a public issue:

- GitHub: use **"Report a vulnerability"** (Security tab → private
  advisory) on this repository, or
- Email: gauthierpiarrette@gmail.com with subject `[AKITA SECURITY]`.

You'll get an acknowledgment within 72 hours. Please include a
reproduction if you can. Credit given in the fix notes unless you
prefer otherwise.
