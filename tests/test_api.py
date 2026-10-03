"""API smoke tests (FastAPI TestClient)."""
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_index_and_scenarios():
    assert client.get("/").status_code == 200
    names = [s["name"] for s in client.get("/api/scenarios").json()]
    assert "intercept_resend_Z" in names and len(names) >= 15


def test_run_and_ledger():
    r = client.post("/api/run", json={"scenario": "replay_same_block", "seed": 1}).json()
    assert r["report"]["attack_class"] == "REPLAY ATTACK"
    led = client.get("/api/ledger").json()
    assert led["chain_valid"] and led["length"] >= 2
    proof = client.get("/api/ledger/0/proof").json()
    assert "anchored" in proof


def test_bad_scenario():
    assert client.post("/api/run", json={"scenario": "nope"}).status_code == 404


def test_websocket_stream():
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"scenario": "honest", "seed": 2})
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m["type"])
            if m["type"] == "result":
                assert m["session"]["report"]["attack_class"] == "NONE (HONEST)"
                break
        assert msgs[0] == "start" and "llr" in msgs
