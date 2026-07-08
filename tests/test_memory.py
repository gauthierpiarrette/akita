import base64
import json
import os

import numpy as np
import pytest
import tenseal as ts
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from akita import HashingEmbedder, IntegrityError, Memory, MemoryServer

NOTES = [
    "therapist said my migraines get worse after night shifts",
    "lawyer confirmed the settlement offer expires next friday",
    "doctor increased the propranolol dose to 80mg",
    "landlord agreed to fix the heating before december",
    "bank flagged the wire transfer to the singapore account",
    "sister's surgery was rescheduled to the 14th",
    "recruiter from the hedge fund called about the quant role",
    "dentist found a cracked molar, crown needed",
    "accountant says the capital gains bill will be huge this year",
    "coach thinks the knee can handle the marathon in april",
    "pharmacist warned about mixing ibuprofen with the new meds",
    "hr confirmed the parental leave starts in march",
]


@pytest.fixture
def mem(tmp_path):
    server = MemoryServer()
    m = Memory.open("user1", "correct horse battery", server,
                    HashingEmbedder(384), store_dir=tmp_path)
    m.add(NOTES)
    return m


def test_search_matches_plaintext_pipeline_exactly(mem):
    emb = HashingEmbedder(384)
    query = "what did the doctor say about my migraine medication"
    ref_scores = emb.encode(NOTES) @ emb.encode([query])[0]
    ref_top = np.argsort(ref_scores)[::-1][:5]

    results = mem.search(query, k=5)
    assert [r["id"] for r in results] == ref_top.tolist()
    for r in results:
        assert abs(r["score"] - ref_scores[r["id"]]) < 1e-4
        assert r["text"] == NOTES[r["id"]]


def test_server_is_blind(mem):
    report = mem.audit()
    assert report["context_is_private"] is False
    assert report["client_verified_no_secret_key"] is True
    assert report["segments"] >= 1 and report["blobs"] >= len(NOTES)

    server = mem._server
    for blob in server._users["user1"]["blobs"].values():
        for word in ("migraines", "settlement", "propranolol"):
            assert word.encode() not in blob
    for seg in server._users["user1"]["segments"]:
        assert b"migraines" not in seg


def test_delete_tombstones_then_compact_removes(mem):
    victim = mem.search("settlement offer from the lawyer", k=1)[0]
    mem.delete(victim["id"])

    after = mem.search("settlement offer from the lawyer", k=len(NOTES) - 1)
    assert victim["id"] not in [r["id"] for r in after]

    segs_before = mem.audit()["segments"]
    mem.compact()
    assert mem.audit()["segments"] <= segs_before
    # still exact after physical re-pack
    again = mem.search("wire transfer bank account", k=3)
    assert "singapore" in again[0]["text"]


def test_keystore_reopen_and_wrong_passphrase(tmp_path):
    server = MemoryServer()
    m = Memory.open("u2", "hunter2hunter2", server,
                    HashingEmbedder(384), store_dir=tmp_path)
    m.add(["the safe deposit box code is 4417"])
    del m

    m2 = Memory.open("u2", "hunter2hunter2", server,
                     HashingEmbedder(384), store_dir=tmp_path)
    hit = m2.search("safe deposit code", k=1)[0]
    assert "4417" in hit["text"]

    with pytest.raises(ValueError, match="passphrase"):
        Memory.open("u2", "wrong password", server,
                    HashingEmbedder(384), store_dir=tmp_path)


def test_search_fetches_decoys(mem):
    before = mem._server.audit("user1")["blob_fetches_observed"]
    mem.search("knee marathon training", k=2, decoys=3)
    fetched = mem._server.audit("user1")["blob_fetches_observed"] - before
    assert fetched == 5  # 2 winners + 3 decoys, indistinguishable to server


def test_old_keystore_fresh_server_reenrolls(tmp_path):
    """A surviving keystore must work against a brand-new server: the
    client re-enrolls its public context and starts an empty corpus."""
    m = Memory.open("u4", "pw", MemoryServer(), HashingEmbedder(384),
                    store_dir=tmp_path)
    m.add(["first life"])
    fresh_server = MemoryServer()
    m2 = Memory.open("u4", "pw", fresh_server, HashingEmbedder(384),
                     store_dir=tmp_path)
    assert fresh_server.has_user("u4")
    assert m2.search("first life", k=1) == []   # new server holds nothing
    m2.add(["second life"])
    assert "second" in m2.search("second life", k=1)[0]["text"]


def test_wipe_is_crypto_shredding(tmp_path):
    server = MemoryServer()
    m = Memory.open("u3", "pass", server, HashingEmbedder(384),
                    store_dir=tmp_path)
    m.add(["ephemeral secret"])
    m.wipe()
    with pytest.raises(KeyError):
        server.audit("u3")
    assert not (tmp_path / "u3.keys").exists()
    assert not (tmp_path / "u3.state").exists()


def _public_context_bytes() -> bytes:
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2**40
    return ctx.serialize(save_secret_key=False)


def test_enroll_refuses_context_with_secret_key():
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2**40
    with pytest.raises(ValueError, match="secret key"):
        MemoryServer().enroll("mallory", ctx.serialize(save_secret_key=True))


def test_enroll_cannot_hijack_existing_user(mem):
    """Re-enrolling an existing user with a different context would let
    anyone clobber their data; the server must refuse."""
    with pytest.raises(ValueError, match="already enrolled"):
        mem._server.enroll("user1", _public_context_bytes())


def test_audit_verifies_context_client_side(mem):
    report = mem.audit()
    assert report["client_verified_no_secret_key"] is True
    assert report["client_verified_context_unchanged"] is True
    # a server that swaps in a different (still public) context is caught
    mem._server._users["user1"]["ctx_bytes"] = _public_context_bytes()
    report = mem.audit()
    assert report["client_verified_context_unchanged"] is False


def test_blob_swap_is_detected(mem):
    """A server that answers a fetch for blob A with blob B's ciphertext
    must fail authentication, not silently return the wrong memory."""
    u = mem._server._users["user1"]
    u["blobs"]["0"], u["blobs"]["1"] = u["blobs"]["1"], u["blobs"]["0"]
    with pytest.raises(IntegrityError, match="authentication failed"):
        mem.search("anything at all", k=len(NOTES), decoys=0)


def test_index_rollback_is_detected(tmp_path):
    server = MemoryServer()
    m = Memory.open("u5", "pw", server, HashingEmbedder(384),
                    store_dir=tmp_path)
    m.add(["version one"])
    stale = server._users["u5"]["blobs"][Memory.INDEX_BLOB]
    m.add(["version two"])
    server._users["u5"]["blobs"][Memory.INDEX_BLOB] = stale
    with pytest.raises(IntegrityError, match="rollback"):
        Memory.open("u5", "pw", server, HashingEmbedder(384),
                    store_dir=tmp_path)


def test_key_files_are_owner_only(tmp_path):
    Memory.open("u6", "pw", MemoryServer(), HashingEmbedder(384),
                store_dir=tmp_path)
    assert tmp_path.stat().st_mode & 0o077 == 0
    for name in ("u6.keys", "u6.state"):
        assert (tmp_path / name).stat().st_mode & 0o077 == 0


def test_legacy_v1_keystore_upgrades_transparently(tmp_path):
    """A pre-v0.2 keystore (no version field, scrypt n=2^14) must still
    open with the right passphrase and be rewritten in the new format."""
    user, pw = "u7", "old-passphrase"
    salt = os.urandom(16)
    master = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1).derive(
        pw.encode())
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2**40
    ctx.generate_relin_keys()
    payload = json.dumps({
        "ckks_secret": base64.b64encode(
            ctx.serialize(save_secret_key=True)).decode(),
        "aes_key": base64.b64encode(os.urandom(32)).decode(),
    }).encode()
    nonce = os.urandom(12)
    wrapped = nonce + AESGCM(master).encrypt(nonce, payload, user.encode())
    (tmp_path / f"{user}.keys").write_text(json.dumps({
        "salt": base64.b64encode(salt).decode(),
        "wrapped": base64.b64encode(wrapped).decode(),
    }))

    m = Memory.open(user, pw, MemoryServer(), HashingEmbedder(384),
                    store_dir=tmp_path)
    m.add(["survived the upgrade"])
    assert "survived" in m.search("upgrade", k=1)[0]["text"]
    upgraded = json.loads((tmp_path / f"{user}.keys").read_text())
    assert upgraded["version"] == 2
    assert upgraded["kdf"]["n"] == 2**17
