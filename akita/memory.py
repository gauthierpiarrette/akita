"""
Akita Private Memory v0: per-user encrypted memory that the operator
cannot read.

Trust model
-----------
- The KEY HOLDER (end user's device) runs `Memory`: it embeds text
  locally, holds the CKKS secret context and the AES key (both derived
  from / wrapped by a passphrase), and is the only party that ever sees
  plaintext.
- The SERVER runs `MemoryServer`: it stores AES-encrypted text blobs it
  cannot open, CKKS-encrypted embedding segments it cannot decrypt, and
  answers searches by multiplying ciphertexts it cannot read. Its entire
  API is bytes-in/bytes-out; enrollment refuses any context that holds a
  secret key.

Tamper evidence
---------------
The server is modeled as honest-but-curious, but the client fails closed
on detectable tampering: every AES-GCM blob is bound to its blob id (a
substituted or reordered blob fails authentication), the encrypted index
carries a version counter pinned in a local state file (a rolled-back
index is refused), and `Memory.audit()` fetches the server-held context
and verifies client-side that it cannot decrypt and still matches the
enrolled hash. Caveat: a server that wipes a user entirely is
indistinguishable from crypto-shredding; rollback detection covers
partial rollbacks, not total loss.

The v0 kernel (measured in this repo)
-------------------------------------
Embeddings are zero-padded to a power-of-two block and packed
CHUNKS_PER_CT per ciphertext. The client sends the query replicated
across every block; the server computes ONE elementwise ct-by-ct multiply
per segment (~1.9 ms/segment on one M1 Pro core — no rotations, so no
galois keys and a ~1.5 MB public context instead of ~34 MB) and returns
the encrypted products. The client decrypts and block-sums locally.
Honest cost: download scales with corpus size (~230 KB/segment); the
roadmap fix is server-side rotation aggregation.

Residual leakage (stated, not hidden): the server sees counts, sizes,
timing, and which blob ids are fetched after a search (mitigated with
decoy fetches; the real fix is a PIR fetch stage).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from pathlib import Path

import numpy as np
import tenseal as ts
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

POLY_DEGREE = 8192
SLOTS = POLY_DEGREE // 2
COEFF_MOD_BITS = [60, 40, 40, 60]
GLOBAL_SCALE = 2**40

KEYSTORE_VERSION = 2
SCRYPT_KDF = {"n": 2**17, "r": 8, "p": 1}     # OWASP-recommended work factor
LEGACY_V1_KDF = {"n": 2**14, "r": 8, "p": 1}  # pre-v0.2 keystores


class IntegrityError(RuntimeError):
    """The server returned data that fails authentication, is stale, or
    is missing. The client fails closed rather than use it."""


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _unb64(s: str) -> bytes:
    return base64.b64decode(s)


def _write_private(path: Path, data: bytes) -> None:
    """Write a key-material file readable by the owner only."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)  # files created before v0.2 may be looser


# --------------------------------------------------------------- server

class MemoryServer:
    """The blind side. Receives only public/encrypted bytes."""

    def __init__(self):
        self._users: dict[str, dict] = {}

    def _user(self, user_id: str) -> dict:
        if user_id not in self._users:
            raise KeyError(f"unknown user {user_id!r}")
        return self._users[user_id]

    def has_user(self, user_id: str) -> bool:
        return user_id in self._users

    def enroll(self, user_id: str, public_context: bytes) -> None:
        ctx = ts.context_from(public_context)
        if ctx.is_private():
            raise ValueError(
                "refusing enrollment: context contains a secret key"
            )
        if user_id in self._users:
            if self._users[user_id]["ctx_bytes"] == public_context:
                return  # idempotent re-enrollment with the same context
            raise ValueError(
                f"user {user_id!r} is already enrolled with a different "
                "context; wipe() first to replace it"
            )
        self._users[user_id] = {
            "ctx_bytes": public_context,
            "ctx": ctx,
            "segments": [],
            "blobs": {},
            "fetches": 0,
        }

    def get_context(self, user_id: str) -> bytes:
        """The stored public context, so the client can verify it."""
        return self._user(user_id)["ctx_bytes"]

    def put_blob(self, user_id: str, blob_id: str, blob: bytes) -> None:
        self._user(user_id)["blobs"][blob_id] = blob

    def get_blobs(self, user_id: str, blob_ids: list[str]) -> dict[str, bytes]:
        u = self._user(user_id)
        u["fetches"] += len(blob_ids)
        return {i: u["blobs"][i] for i in blob_ids if i in u["blobs"]}

    def delete_blobs(self, user_id: str, blob_ids: list[str]) -> None:
        u = self._user(user_id)
        for i in blob_ids:
            u["blobs"].pop(i, None)

    def append_segment(self, user_id: str, ct: bytes) -> int:
        u = self._user(user_id)
        u["segments"].append(ct)
        return len(u["segments"]) - 1

    def get_segments(self, user_id: str) -> list[bytes]:
        return list(self._user(user_id)["segments"])

    def replace_segments(self, user_id: str, cts: list[bytes]) -> None:
        self._user(user_id)["segments"] = list(cts)

    def search(self, user_id: str, query_ct: bytes) -> list[bytes]:
        """One ct-by-ct multiply per segment. The server never sees the
        query, the products, or which memories score high."""
        import time
        u = self._user(user_id)
        q = ts.ckks_vector_from(u["ctx"], query_ct)
        out = []
        t0 = time.perf_counter()
        for seg_bytes in u["segments"]:
            seg = ts.ckks_vector_from(u["ctx"], seg_bytes)
            out.append((seg * q).serialize())
        u["last_search_s"] = time.perf_counter() - t0
        return out

    def audit(self, user_id: str) -> dict:
        u = self._user(user_id)
        ctx = ts.context_from(u["ctx_bytes"])  # fresh load, no cache tricks
        return {
            "context_is_private": ctx.is_private(),
            "context_sha256": hashlib.sha256(u["ctx_bytes"]).hexdigest(),
            "segments": len(u["segments"]),
            "blobs": len(u["blobs"]),
            "storage_bytes": sum(len(b) for b in u["blobs"].values())
            + sum(len(s) for s in u["segments"])
            + len(u["ctx_bytes"]),
            "sample_ciphertext_hex": (
                u["segments"][0][:24].hex() if u["segments"] else None
            ),
            "blob_fetches_observed": u["fetches"],
        }

    def wipe(self, user_id: str) -> None:
        self._users.pop(user_id, None)


# --------------------------------------------------------------- client

class Memory:
    """The key holder's side. The only party that ever sees plaintext."""

    INDEX_BLOB = "__index__"
    STATE_LABEL = "__state__"

    def __init__(self, user_id, server, embedder, ctx, aes_key, store_path,
                 state: dict | None = None):
        self.user_id = user_id
        self._server = server
        self._embedder = embedder
        self._ctx = ctx
        self._aes = AESGCM(aes_key)
        self._store_path = store_path
        self._state_path = store_path.with_suffix(".state")
        self._block = _next_pow2(embedder.dim)
        if self._block > SLOTS or SLOTS % self._block:
            raise ValueError(
                f"embedding dim {embedder.dim} (padded {self._block}) "
                f"does not pack into {SLOTS} slots"
            )
        self._chunks_per_ct = SLOTS // self._block
        if state is not None:
            self._state = state
            self._save_state()
        else:
            self._state = self._load_state()
        self._index = self._load_index()

    # ---- keystore -----------------------------------------------------

    @classmethod
    def open(cls, user_id: str, passphrase: str, server: MemoryServer,
             embedder, store_dir: str | Path = ".akita_keys") -> "Memory":
        store = Path(store_dir)
        store.mkdir(parents=True, exist_ok=True)
        os.chmod(store, 0o700)
        path = store / f"{user_id}.keys"

        if path.exists():
            data = json.loads(path.read_text())
            version = data.get("version", 1)
            kdf = data.get("kdf", LEGACY_V1_KDF)
            master = cls._kdf(passphrase, _unb64(data["salt"]),
                              n=kdf["n"], r=kdf["r"], p=kdf["p"])
            wrapped = _unb64(data["wrapped"])
            try:
                payload = json.loads(
                    AESGCM(master).decrypt(wrapped[:12], wrapped[12:],
                                           user_id.encode())
                )
            except InvalidTag:
                raise ValueError(
                    f"wrong passphrase (or corrupted keystore) for "
                    f"{user_id!r}"
                ) from None
            ctx = ts.context_from(_unb64(payload["ckks_secret"]))
            aes_key = _unb64(payload["aes_key"])
            if version < KEYSTORE_VERSION or kdf != SCRYPT_KDF:
                # transparent upgrade: rewrap under the current KDF
                cls._write_keystore(path, user_id, passphrase, payload)

            pub = ctx.serialize(save_secret_key=False)
            state = None
            if not server.has_user(user_id):
                # existing keystore, new/blank server: re-enroll the
                # public context and pin a fresh history (the server
                # starts empty for this user)
                server.enroll(user_id, pub)
                state = {"context_sha256": hashlib.sha256(pub).hexdigest(),
                         "index_version": 0}
            elif not path.with_suffix(".state").exists():
                # pre-v0.2 keystore: start pinning from here
                state = {"context_sha256": hashlib.sha256(pub).hexdigest(),
                         "index_version": 0}
            return cls(user_id, server, embedder, ctx, aes_key, path, state)

        ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                         poly_modulus_degree=POLY_DEGREE,
                         coeff_mod_bit_sizes=COEFF_MOD_BITS)
        ctx.global_scale = GLOBAL_SCALE
        ctx.generate_relin_keys()   # ct-by-ct mult; no galois needed
        aes_key = secrets.token_bytes(32)
        cls._write_keystore(path, user_id, passphrase, {
            "ckks_secret": _b64(ctx.serialize(save_secret_key=True)),
            "aes_key": _b64(aes_key),
        })
        pub = ctx.serialize(save_secret_key=False)
        server.enroll(user_id, pub)
        mem = cls(user_id, server, embedder, ctx, aes_key, path,
                  state={"context_sha256": hashlib.sha256(pub).hexdigest(),
                         "index_version": 0})
        mem._save_index()
        return mem

    @staticmethod
    def _kdf(passphrase: str, salt: bytes, n: int, r: int, p: int) -> bytes:
        return Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(
            passphrase.encode()
        )

    @classmethod
    def _write_keystore(cls, path: Path, user_id: str, passphrase: str,
                        payload: dict) -> None:
        salt = secrets.token_bytes(16)
        master = cls._kdf(passphrase, salt, **SCRYPT_KDF)
        nonce = secrets.token_bytes(12)
        wrapped = nonce + AESGCM(master).encrypt(
            nonce, json.dumps(payload).encode(), user_id.encode()
        )
        _write_private(path, json.dumps({
            "version": KEYSTORE_VERSION,
            "kdf": dict(SCRYPT_KDF),
            "salt": _b64(salt),
            "wrapped": _b64(wrapped),
        }).encode())

    # ---- local pinned state (rollback counter + context hash) ---------

    def _load_state(self) -> dict:
        if not self._state_path.exists():
            return {"context_sha256": None, "index_version": 0}
        return json.loads(
            self._aes_decrypt(self._state_path.read_bytes(),
                              self.STATE_LABEL)
        )

    def _save_state(self) -> None:
        _write_private(
            self._state_path,
            self._aes_encrypt(json.dumps(self._state), self.STATE_LABEL),
        )

    # ---- encrypted index (just another blob the server can't read) ----

    def _load_index(self) -> dict:
        blobs = self._server.get_blobs(self.user_id, [self.INDEX_BLOB])
        if self.INDEX_BLOB not in blobs:
            if self._state["index_version"] > 0:
                raise IntegrityError(
                    "server holds no index but one existed (version "
                    f"{self._state['index_version']}): rollback or data loss"
                )
            return {"next_id": 0, "entries": {}, "tombstones": [],
                    "version": 0}
        index = json.loads(
            self._aes_decrypt(blobs[self.INDEX_BLOB], self.INDEX_BLOB)
        )
        version = int(index.get("version", 0))
        if version < self._state["index_version"]:
            raise IntegrityError(
                f"server returned index version {version}, older than the "
                f"last seen {self._state['index_version']}: rollback"
            )
        if version > self._state["index_version"]:
            # e.g. a crash between the server write and the local pin
            self._state["index_version"] = version
            self._save_state()
        return index

    def _save_index(self) -> None:
        self._index["version"] = int(self._index.get("version", 0)) + 1
        self._server.put_blob(self.user_id, self.INDEX_BLOB,
                              self._aes_encrypt(json.dumps(self._index),
                                                self.INDEX_BLOB))
        self._state["index_version"] = self._index["version"]
        self._save_state()

    def _aes_encrypt(self, plaintext: str, label: str) -> bytes:
        """AES-GCM with the blob identity in the associated data, so the
        server cannot swap one blob for another undetected."""
        nonce = secrets.token_bytes(12)
        aad = f"{self.user_id}:{label}".encode()
        return nonce + self._aes.encrypt(nonce, plaintext.encode(), aad)

    def _aes_decrypt(self, blob: bytes, label: str) -> str:
        aad = f"{self.user_id}:{label}".encode()
        try:
            return self._aes.decrypt(blob[:12], blob[12:], aad).decode()
        except InvalidTag:
            raise IntegrityError(
                f"authentication failed for {label!r}: wrong key, or the "
                "server returned tampered or substituted data"
            ) from None

    # ---- core API ------------------------------------------------------

    def add(self, texts: list[str], metadata: list[dict] | None = None) -> list[int]:
        """Embed locally, encrypt locally, ship only ciphertext."""
        metadata = metadata or [{} for _ in texts]
        emb = self._embedder.encode(texts)                 # (n, dim)
        padded = np.zeros((len(texts), self._block))
        padded[:, : emb.shape[1]] = emb

        ids = []
        for start in range(0, len(texts), self._chunks_per_ct):
            group = padded[start : start + self._chunks_per_ct]
            flat = np.zeros(SLOTS)
            flat[: group.size] = group.flatten()
            seg_idx = self._server.append_segment(
                self.user_id,
                ts.ckks_vector(self._ctx, flat.tolist()).serialize(),
            )
            for slot, local in enumerate(range(start, start + len(group))):
                mid = self._index["next_id"]
                self._index["next_id"] += 1
                self._index["entries"][str(mid)] = {
                    "segment": seg_idx, "slot": slot,
                    "meta": metadata[local],
                }
                self._server.put_blob(self.user_id, str(mid),
                                      self._aes_encrypt(texts[local],
                                                        str(mid)))
                ids.append(mid)
        self._save_index()
        return ids

    def search(self, query: str, k: int = 5, decoys: int = 2,
               trace: dict | None = None) -> list[dict]:
        """If `trace` is a dict, it is filled with real timings and
        ciphertext facts from this search (for dashboards/demos)."""
        import time as _time

        t0 = _time.perf_counter()
        q = self._embedder.encode([query])[0]
        embed_s = _time.perf_counter() - t0

        padded = np.zeros(self._block)
        padded[: len(q)] = q
        t0 = _time.perf_counter()
        q_ct = ts.ckks_vector(
            self._ctx, np.tile(padded, self._chunks_per_ct).tolist()
        ).serialize()
        encrypt_s = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        products = self._server.search(self.user_id, q_ct)
        server_roundtrip_s = _time.perf_counter() - t0
        t_decrypt = _time.perf_counter()

        dead = set(self._index["tombstones"])
        scored = [
            (mid, e) for mid, e in self._index["entries"].items()
            if mid not in dead and e["segment"] < len(products)
        ]
        # decrypt each product once, block-sum locally
        sums = {}
        needed = {e["segment"] for _, e in scored}
        for seg in needed:
            vec = np.array(
                ts.ckks_vector_from(self._ctx, products[seg])
                .decrypt(self._ctx.secret_key())
            )
            sums[seg] = vec.reshape(self._chunks_per_ct, self._block).sum(axis=1)
        ranked = sorted(
            ((mid, e, float(sums[e["segment"]][e["slot"]])) for mid, e in scored),
            key=lambda t: t[2], reverse=True,
        )[:k]

        # fetch winners plus decoys so the access pattern is padded
        want = [mid for mid, _, _ in ranked]
        pool = [m for m in self._index["entries"]
                if m not in want and m not in dead]
        rng = secrets.SystemRandom()
        extra = rng.sample(pool, min(decoys, len(pool))) if pool else []
        fetch_ids = want + extra
        rng.shuffle(fetch_ids)
        blobs = self._server.get_blobs(self.user_id, fetch_ids)

        if trace is not None:
            trace.update({
                "embed_s": embed_s,
                "encrypt_s": encrypt_s,
                "server_roundtrip_s": server_roundtrip_s,
                "decrypt_and_rank_s": _time.perf_counter() - t_decrypt,
                "query_ct_kb": round(len(q_ct) / 1024, 1),
                "query_ct_hex": q_ct[:24].hex(),
                "download_kb": round(sum(len(p) for p in products) / 1024, 1),
                "segments": len(products),
                "fetched_ids": fetch_ids,   # winners + decoys, shuffled —
                                            # exactly what the server saw
            })

        results = []
        for mid, e, score in ranked:
            if mid not in blobs:
                raise IntegrityError(
                    f"server did not return blob {mid!r} it should hold"
                )
            results.append({
                "id": int(mid), "score": score,
                "text": self._aes_decrypt(blobs[mid], mid),
                "metadata": e["meta"],
            })
        return results

    def delete(self, memory_id: int) -> None:
        """Tombstone now (excluded from every future search); the packed
        ciphertext slot is physically removed at the next compact()."""
        mid = str(memory_id)
        if mid in self._index["entries"] and mid not in self._index["tombstones"]:
            self._index["tombstones"].append(mid)
            self._server.delete_blobs(self.user_id, [mid])
            self._save_index()

    def compact(self) -> None:
        """Client-side re-pack: decrypt segments (key holder's right),
        drop tombstoned slots, re-encrypt densely, replace server state."""
        segments = self._server.get_segments(self.user_id)
        dead = set(self._index["tombstones"])
        live = []
        for mid, e in sorted(self._index["entries"].items(),
                             key=lambda kv: int(kv[0])):
            if mid in dead:
                continue
            vec = np.array(
                ts.ckks_vector_from(self._ctx, segments[e["segment"]])
                .decrypt(self._ctx.secret_key())
            ).reshape(self._chunks_per_ct, self._block)[e["slot"]]
            live.append((mid, e["meta"], vec))

        new_segments, new_entries = [], {}
        for start in range(0, len(live), self._chunks_per_ct):
            group = live[start : start + self._chunks_per_ct]
            flat = np.zeros(SLOTS)
            for slot, (mid, meta, vec) in enumerate(group):
                flat[slot * self._block : slot * self._block + self._block] = vec
                new_entries[mid] = {"segment": len(new_segments),
                                    "slot": slot, "meta": meta}
            new_segments.append(
                ts.ckks_vector(self._ctx, flat.tolist()).serialize()
            )
        self._server.replace_segments(self.user_id, new_segments)
        self._index["entries"] = new_entries
        self._index["tombstones"] = []
        self._save_index()

    def audit(self) -> dict:
        """Server-side facts plus genuine client-side verification: fetch
        the context bytes the server holds, load them locally, and check
        that they cannot decrypt and still hash to what was enrolled."""
        report = self._server.audit(self.user_id)
        ctx_bytes = self._server.get_context(self.user_id)
        held = ts.context_from(ctx_bytes)
        report["client_verified_no_secret_key"] = not held.is_private()
        report["client_verified_context_unchanged"] = (
            hashlib.sha256(ctx_bytes).hexdigest()
            == self._state["context_sha256"]
        )
        return report

    def wipe(self) -> None:
        """Crypto-shredding: delete the keys and the server holds noise."""
        self._server.wipe(self.user_id)
        self._store_path.unlink(missing_ok=True)
        self._state_path.unlink(missing_ok=True)
