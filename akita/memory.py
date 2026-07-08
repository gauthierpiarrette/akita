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
  API is bytes-in/bytes-out; enrollment asserts the context it receives
  holds no secret key.

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
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

POLY_DEGREE = 8192
SLOTS = POLY_DEGREE // 2
COEFF_MOD_BITS = [60, 40, 40, 60]
GLOBAL_SCALE = 2**40


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _unb64(s: str) -> bytes:
    return base64.b64decode(s)


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
        assert not ctx.is_private(), \
            "refusing enrollment: context contains a secret key"
        self._users[user_id] = {
            "ctx_bytes": public_context,
            "ctx": ctx,
            "segments": [],
            "blobs": {},
            "fetches": 0,
        }

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

    def __init__(self, user_id, server, embedder, ctx, aes_key, store_path):
        self.user_id = user_id
        self._server = server
        self._embedder = embedder
        self._ctx = ctx
        self._aes = AESGCM(aes_key)
        self._store_path = store_path
        self._block = _next_pow2(embedder.dim)
        assert SLOTS % self._block == 0
        self._chunks_per_ct = SLOTS // self._block
        self._index = self._load_index()

    # ---- keystore -----------------------------------------------------

    @classmethod
    def open(cls, user_id: str, passphrase: str, server: MemoryServer,
             embedder, store_dir: str | Path = ".akita_keys") -> "Memory":
        store = Path(store_dir)
        store.mkdir(parents=True, exist_ok=True)
        path = store / f"{user_id}.keys"

        if path.exists():
            data = json.loads(path.read_text())
            master = cls._kdf(passphrase, _unb64(data["salt"]))
            wrapped = _unb64(data["wrapped"])
            secrets_json = json.loads(
                AESGCM(master).decrypt(wrapped[:12], wrapped[12:],
                                       user_id.encode())
            )
            ctx = ts.context_from(_unb64(secrets_json["ckks_secret"]))
            aes_key = _unb64(secrets_json["aes_key"])
            if not server.has_user(user_id):
                # existing keystore, new/blank server: re-enroll the
                # public context (the server starts empty for this user)
                server.enroll(user_id, ctx.serialize(save_secret_key=False))
            mem = cls(user_id, server, embedder, ctx, aes_key, path)
        else:
            salt = secrets.token_bytes(16)
            master = cls._kdf(passphrase, salt)
            ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                             poly_modulus_degree=POLY_DEGREE,
                             coeff_mod_bit_sizes=COEFF_MOD_BITS)
            ctx.global_scale = GLOBAL_SCALE
            ctx.generate_relin_keys()   # ct-by-ct mult; no galois needed
            aes_key = secrets.token_bytes(32)
            secrets_json = json.dumps({
                "ckks_secret": _b64(ctx.serialize(save_secret_key=True)),
                "aes_key": _b64(aes_key),
            }).encode()
            nonce = secrets.token_bytes(12)
            wrapped = nonce + AESGCM(master).encrypt(nonce, secrets_json,
                                                     user_id.encode())
            path.write_text(json.dumps(
                {"salt": _b64(salt), "wrapped": _b64(wrapped)}
            ))
            server.enroll(user_id, ctx.serialize(save_secret_key=False))
            mem = cls(user_id, server, embedder, ctx, aes_key, path)
            mem._save_index()
        return mem

    @staticmethod
    def _kdf(passphrase: str, salt: bytes) -> bytes:
        return Scrypt(salt=salt, length=32, n=2**14, r=8, p=1).derive(
            passphrase.encode()
        )

    # ---- encrypted index (just another blob the server can't read) ----

    def _load_index(self) -> dict:
        blobs = self._server.get_blobs(self.user_id, [self.INDEX_BLOB])
        if self.INDEX_BLOB not in blobs:
            return {"next_id": 0, "entries": {}, "tombstones": []}
        return json.loads(self._aes_decrypt(blobs[self.INDEX_BLOB]))

    def _save_index(self) -> None:
        self._server.put_blob(self.user_id, self.INDEX_BLOB,
                              self._aes_encrypt(json.dumps(self._index)))

    def _aes_encrypt(self, plaintext: str) -> bytes:
        nonce = secrets.token_bytes(12)
        return nonce + self._aes.encrypt(nonce, plaintext.encode(),
                                         self.user_id.encode())

    def _aes_decrypt(self, blob: bytes) -> str:
        return self._aes.decrypt(blob[:12], blob[12:],
                                 self.user_id.encode()).decode()

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
                                      self._aes_encrypt(texts[local]))
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
        rng = np.random.default_rng()
        extra = list(rng.choice(pool, size=min(decoys, len(pool)),
                                replace=False)) if pool else []
        fetch_ids = [str(x) for x in want + extra]
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

        return [
            {"id": int(mid), "score": score,
             "text": self._aes_decrypt(blobs[mid]),
             "metadata": e["meta"]}
            for mid, e, score in ranked
        ]

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
        """Server-side facts plus client-side verification that the
        material the server holds cannot decrypt anything."""
        report = self._server.audit(self.user_id)
        report["client_verified_no_secret_key"] = not report["context_is_private"]
        return report

    def wipe(self) -> None:
        """Crypto-shredding: delete the keys and the server holds noise."""
        self._server.wipe(self.user_id)
        self._store_path.unlink(missing_ok=True)
