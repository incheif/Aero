"""
tests/test_server_session_limits.py

Unit and integration tests for server session management:
- Maximum of 5 concurrent sessions
- Earliest opened connection is closed when capacity is exceeded (FIFO eviction)
- WebSocket closure with code 4001 and eviction payload
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from dashboard.server import SessionManager, UserSession


@pytest.fixture
def manager():
    return SessionManager(max_sessions=5)


def test_max_sessions_default_is_five():
    sm = SessionManager()
    assert sm.MAX_SESSIONS == 5


def test_up_to_five_sessions_allowed(manager):
    for i in range(5):
        sess = manager.get_or_create(f"sess_{i}")
        assert sess is not None
        assert sess.session_id == f"sess_{i}"
    assert len(manager.sessions) == 5


def test_sixth_session_evicts_oldest_idle_session(manager):
    # Create 5 sessions
    for i in range(5):
        manager.get_or_create(f"sess_{i}")
        time.sleep(0.01)
    assert len(manager.sessions) == 5
    assert "sess_0" in manager.sessions

    # Create 6th session
    manager.get_or_create("sess_5")

    # Total sessions must still be 5
    assert len(manager.sessions) == 5
    # sess_0 was the oldest and had no connections, so it was evicted
    assert "sess_0" not in manager.sessions
    assert "sess_5" in manager.sessions


def test_sixth_session_closes_first_opened_connection(manager):
    async def _test():
        mock_sockets = {}

        # Create 5 sessions and attach mock websockets with staggered connection times
        for i in range(5):
            sess = manager.get_or_create(f"sess_{i}")
            ws = MagicMock()
            ws.send_json = AsyncMock()
            ws.close = AsyncMock()
            sess.websockets.add(ws)
            sess.first_connected_at = time.time()
            mock_sockets[f"sess_{i}"] = ws
            await asyncio.sleep(0.02)

        assert len(manager.sessions) == 5

        # sess_0 was connected first (earliest connection opened)
        oldest_ws = mock_sockets["sess_0"]
        second_ws = mock_sockets["sess_1"]

        # Now a 6th session arrives and requests capacity
        await manager.ensure_capacity("sess_5")
        sess_5 = manager.get_or_create("sess_5")
        ws_5 = MagicMock()
        ws_5.send_json = AsyncMock()
        ws_5.close = AsyncMock()
        sess_5.websockets.add(ws_5)
        sess_5.first_connected_at = time.time()

        # Total active sessions must remain capped at 5
        assert len(manager.sessions) == 5
        assert "sess_0" not in manager.sessions
        assert "sess_5" in manager.sessions

        # The oldest websocket (sess_0) must have been notified and closed
        oldest_ws.send_json.assert_awaited()
        sent_payload = oldest_ws.send_json.await_args[0][0]
        assert sent_payload["type"] == "session_evicted"
        oldest_ws.close.assert_awaited_with(code=4001, reason="Session limit exceeded")

        # The second session (sess_1) must NOT have been closed
        second_ws.close.assert_not_awaited()

    asyncio.run(_test())


def test_reconnecting_evicted_session_evicts_next_oldest(manager):
    async def _test():
        # Establish 5 connected sessions
        for i in range(5):
            sess = manager.get_or_create(f"sess_{i}")
            ws = MagicMock()
            ws.send_json = AsyncMock()
            ws.close = AsyncMock()
            sess.websockets.add(ws)
            sess.first_connected_at = time.time()
            await asyncio.sleep(0.01)

        # Evict sess_0 by adding connected sess_5
        await manager.ensure_capacity("sess_5")
        sess_5 = manager.get_or_create("sess_5")
        ws_5 = MagicMock()
        ws_5.send_json = AsyncMock()
        ws_5.close = AsyncMock()
        sess_5.websockets.add(ws_5)
        sess_5.first_connected_at = time.time()
        assert "sess_0" not in manager.sessions
        assert len(manager.sessions) == 5

        # Now sess_0 reconnects
        await manager.ensure_capacity("sess_0")
        sess_0 = manager.get_or_create("sess_0")
        ws_0 = MagicMock()
        ws_0.send_json = AsyncMock()
        ws_0.close = AsyncMock()
        sess_0.websockets.add(ws_0)
        sess_0.first_connected_at = time.time()

        # Total sessions still 5, and now sess_1 was the oldest connected, so it got evicted
        assert len(manager.sessions) == 5
        assert "sess_1" not in manager.sessions
        assert "sess_0" in manager.sessions
        assert "sess_5" in manager.sessions

    asyncio.run(_test())


def test_existing_session_lookup_does_not_trigger_eviction(manager):
    for i in range(5):
        manager.get_or_create(f"sess_{i}")
    assert len(manager.sessions) == 5

    # Looking up existing session 2 should not evict anything
    sess_2 = manager.get_or_create("sess_2")
    assert sess_2.session_id == "sess_2"
    assert len(manager.sessions) == 5
    assert "sess_0" in manager.sessions
