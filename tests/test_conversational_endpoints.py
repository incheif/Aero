"""
tests/test_conversational_endpoints.py

Integration tests for conversational human-in-the-loop endpoints:
  1. POST /api/vla/pause
  2. POST /api/vla/guidance
  3. POST /api/vla/chat
"""

import pytest
from fastapi.testclient import TestClient
from dashboard.server import app, session_manager


@pytest.fixture
def client():
    return TestClient(app)


def test_conversational_endpoints_flow(client):
    sid = "test_conv_session"
    sess = session_manager.get_or_create(sid)

    # 1. Send initial command via /api/vla/command
    res = client.post("/api/vla/command", json={"command": "make a letter D with blocks", "session_id": sid})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "easy_steps" in data

    # Verify dialogue supervisor state
    assert sess.supervisor.state == "EXECUTING"
    assert len(sess.supervisor.chat_history) >= 2

    # 2. Trigger pause via /api/vla/pause
    pause_res = client.post("/api/vla/pause", json={"reason": "Operator pause test", "session_id": sid})
    assert pause_res.status_code == 200
    pdata = pause_res.json()
    assert pdata["paused"] is True
    assert sess.is_paused is True
    assert sess.supervisor.state == "PAUSED_BY_USER"
    assert len(pdata["dialogue"]["options"]) > 0

    # 3. Send guidance via /api/vla/guidance: "use purple instead"
    guide_res = client.post("/api/vla/guidance", json={"guidance": "use purple instead", "session_id": sid})
    assert guide_res.status_code == 200
    gdata = guide_res.json()
    assert gdata["resumed"] is True
    assert sess.is_paused is False
    assert sess.supervisor.state == "EXECUTING"

    # Verify memory recorded this intervention
    assert len(sess.memory_agent.episodes) >= 1

    # 4. Test chat endpoint pause
    chat_pause = client.post("/api/vla/chat", json={"message": "wait stop", "session_id": sid})
    assert chat_pause.status_code == 200
    cpdata = chat_pause.json()
    assert cpdata["action"] == "PAUSED"
    assert sess.is_paused is True

    # 5. Resume via chat
    chat_resume = client.post("/api/vla/chat", json={"message": "continue", "session_id": sid})
    assert chat_resume.status_code == 200
    crdata = chat_resume.json()
    assert crdata["action"] in ("RESUME", "GUIDANCE_PROCESSED")
    assert sess.is_paused is False

