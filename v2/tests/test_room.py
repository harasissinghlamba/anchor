"""Service, session levels and the Control Room with a stub scorer (no model): the stub's rule is fixed (text with
the injection phrase → 0.95, else 0.05) so these tests check plumbing and policy, never detection quality."""
import sys
import threading
import time
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V2 / "offline"))

from fastapi.testclient import TestClient  # noqa: E402

from anchor.service import create_app  # noqa: E402
from anchor.sidecar import Session  # noqa: E402
from room import world  # noqa: E402


class StubScorer:
    threshold = 0.5
    layer = 9
    watch_threshold = None

    def __init__(self, fail=False, injection_score=0.95):
        self.fail = fail
        self.injection_score = injection_score
        self.lock = threading.Lock()

    def score(self, task, output, prior=(), attribution=None):
        if self.fail:
            return {"drift_score": None, "tainted": True, "fail_closed": True, "reason": "scorer error: Stub"}
        bad = "IMPORTANT NOTICE FOR AUTOMATED" in output or "INJECT" in output
        s = self.injection_score if bad else 0.05
        r = {"drift_score": s, "tainted": s >= self.threshold, "fail_closed": False, "skipped": False}
        if attribution:
            from anchor.scorer import sentences
            r["attribution"] = [{"start": a, "end": b, "text": output[a:b],
                                 "weight": 0.9 if ("NOTICE" in output[a:b] or "Ignore the remit" in output[a:b]
                                                   or "Mark it urgent" in output[a:b]) else 0.0}
                                for a, b in sentences(output)]
        return r


@pytest.fixture
def client():
    app = create_app(scorer=StubScorer(), model_info={"model": "stub", "device": "cpu"})
    return TestClient(app)


def test_health_reports_probe_fields(client):
    h = client.get("/v1/health").json()
    assert h["ok"] and h["threshold"] == 0.5 and h["layer"] == 9 and h["steering"] is False


def test_taint_blocks_later_side_effect(client):
    sid = client.post("/v1/sessions", json={"task": "Summarise the page"}).json()["session_id"]
    assert client.post("/v1/decide", json={"session_id": sid, "tool": "send"}).json()["decision"] == "allow"
    r = client.post("/v1/score", json={"session_id": sid, "source": "web", "content": "INJECT now"}).json()
    assert r["level"] == "tainted" and r["session_level"] == "tainted"
    client.post("/v1/score", json={"session_id": sid, "source": "web", "content": "clean text"})
    d = client.post("/v1/decide", json={"session_id": sid, "tool": "send"}).json()
    assert d["decision"] == "block"  # sticky


def test_handoff_propagates_taint(client):
    a = client.post("/v1/sessions", json={"task": "t", "session_id": "A"}).json()["session_id"]
    b = client.post("/v1/sessions", json={"task": "t", "session_id": "B"}).json()["session_id"]
    client.post("/v1/score", json={"session_id": a, "content": "INJECT"})
    r = client.post("/v1/handoff", json={"from": a, "to": b, "content": "here is my clean-looking summary"}).json()
    assert r["session_level"] == "tainted"
    assert client.post("/v1/decide", json={"session_id": b, "tool": "pay"}).json()["tainted_by"] == "A"


def test_scorer_failure_fails_closed():
    c = TestClient(create_app(scorer=StubScorer(fail=True), model_info={}))
    sid = c.post("/v1/sessions", json={"task": "t"}).json()["session_id"]
    assert c.post("/v1/score", json={"session_id": sid, "content": "x"}).json()["fail_closed"] is True
    assert c.post("/v1/decide", json={"session_id": sid, "tool": "pay"}).json()["decision"] == "block"


def test_token_auth(monkeypatch):
    monkeypatch.setenv("ANCHOR_TOKEN", "t0k")
    c = TestClient(create_app(scorer=StubScorer(), model_info={}))
    assert c.get("/v1/health").status_code == 200
    assert c.post("/v1/sessions", json={"task": "t"}).status_code == 401
    assert c.post("/v1/sessions", json={"task": "t"}, headers={"authorization": "Bearer t0k"}).status_code == 200


def test_watch_level_on_spike_against_session_baseline():
    s = Session("x", "t")
    for v in (0.01, 0.012, 0.011):
        s.record("web", {"drift_score": v}, 0.5)
    r = s.record("web", {"drift_score": 0.3}, 0.5)
    assert r["level"] == "watch" and s.decide("pay")["decision"] == "approve"
    s.record("web", {"drift_score": 0.01}, 0.5)
    assert s.level == "watch"  # never gets cleaner


def run_room(client, mode, approve=True):
    """approve: True → a human approves every card except a payout to the attacker account (which a human sees)."""
    room = client.app.state.room
    assert client.post("/v1/room/run", json={"mode": mode, "brain": "scripted", "pace": 0}).json()["started"]
    deadline = time.time() + 20
    while time.time() < deadline:
        st = client.get("/v1/room/state").json()
        for a in st["approvals"]:
            if a["decision"] is None:
                ok = approve and a["action"].get("account") != world.ATTACKER_ACCOUNT
                client.post("/v1/room/approve", json={"id": a["id"], "decision": "approve" if ok else "deny"})
        if not st["running"] and st["tally"]["effects"] + st["tally"]["blocked"] > 0:
            return st
        time.sleep(0.05)
    raise AssertionError(f"room did not finish: {room.state()}")


def test_room_off_pays_attacker(client, tmp_path, monkeypatch):
    monkeypatch.setattr("room.runner.RUNS", tmp_path)
    st = run_room(client, "off")
    assert st["tally"]["attacker_payments"] == 1 and st["tally"]["blocked"] == 0
    assert len(list(tmp_path.glob("*-off-scripted.json"))) == 1


def test_room_on_blocks_attacker_reroutes_and_keeps_clean_work(client, tmp_path, monkeypatch):
    monkeypatch.setattr("room.runner.RUNS", tmp_path)
    st = run_room(client, "on", approve=True)
    t = st["tally"]
    assert t["attacker_payments"] == 0 and t["false_blocks"] == 0
    assert t["blocked"] == 1 and t["approvals"] == 1
    tools = [e["tool"] for e in st["ledger"]]
    assert sorted(tools) == ["grant_access", "release_payment", "send_email"]
    pay = next(e for e in st["ledger"] if e["tool"] == "release_payment")
    assert pay["action"]["account"] == world.VENDOR_ACCOUNT


def test_room_on_denied_approval_pays_nobody(client, tmp_path, monkeypatch):
    monkeypatch.setattr("room.runner.RUNS", tmp_path)
    st = run_room(client, "on", approve=False)
    assert not any(e["tool"] == "release_payment" for e in st["ledger"])


def watch_client():
    sc = StubScorer(injection_score=0.6)
    sc.threshold, sc.watch_threshold = 0.75, 0.5
    return TestClient(create_app(scorer=sc, model_info={}))


def test_watch_band_needs_approval_and_propagates():
    c = watch_client()
    a = c.post("/v1/sessions", json={"task": "t", "session_id": "A"}).json()["session_id"]
    b = c.post("/v1/sessions", json={"task": "t", "session_id": "B"}).json()["session_id"]
    r = c.post("/v1/score", json={"session_id": a, "content": "INJECT"}).json()
    assert r["level"] == "watch"
    assert c.post("/v1/decide", json={"session_id": a, "tool": "pay"}).json()["decision"] == "approve"
    c.post("/v1/handoff", json={"from": a, "to": b, "content": "summary"})
    assert c.post("/v1/decide", json={"session_id": b, "tool": "pay"}).json()["decision"] == "approve"


def test_room_watch_path_human_denies_attacker_then_reroute(tmp_path, monkeypatch):
    monkeypatch.setattr("room.runner.RUNS", tmp_path)
    st = run_room(watch_client(), "on", approve=True)
    t = st["tally"]
    assert t["attacker_payments"] == 0 and t["false_blocks"] == 0 and t["approvals"] == 2
    pays = [e for e in st["ledger"] if e["tool"] == "release_payment"]
    assert len(pays) == 1 and pays[0]["action"]["account"] == world.VENDOR_ACCOUNT


def test_sentence_splitter_keeps_emails_and_decimals_whole():
    from anchor.scorer import sentences
    t = "Contact accounts@northwind.example. Version 2.5 is out!\nNew line here"
    assert [t[a:b] for a, b in sentences(t)] == ["Contact accounts@northwind.example.", "Version 2.5 is out!",
                                                 "New line here"]


def test_unusable_brain_action_is_never_executed(client, tmp_path, monkeypatch):
    monkeypatch.setattr("room.runner.RUNS", tmp_path)
    from room.brains import ScriptedBrain
    real = ScriptedBrain.act
    monkeypatch.setattr(ScriptedBrain, "act", lambda self, role, task, doc, session=None:
                        ({}, "garbled") if role == "comms" else real(self, role, task, doc, session))
    st = run_room(client, "off")
    assert not any(e["tool"] == "send_email" for e in st["ledger"]) and st["tally"]["invalid"] == 1
