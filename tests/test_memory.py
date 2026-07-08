import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

from akita import HashingEmbedder, Memory, MemoryServer

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

    with pytest.raises(InvalidTag):
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
