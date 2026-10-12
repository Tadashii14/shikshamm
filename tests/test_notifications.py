"""Regression coverage for browser-facing attendance notifications."""
import asyncio
import uuid

from app.notifications import manager
from app.routers.face import notify_user as notify_face_user


class _FakeWebSocket:
    def __init__(self):
        self.payloads = []

    async def accept(self):
        return None

    async def send_json(self, payload):
        self.payloads.append(payload)


def test_face_attendance_notification_reaches_shared_websocket_as_json():
    user_id = "test-" + uuid.uuid4().hex
    websocket = _FakeWebSocket()

    async def exercise():
        await manager.connect(user_id, websocket)
        try:
            await notify_face_user(user_id, "Attendance marked")
        finally:
            manager.disconnect(user_id, websocket)

    asyncio.run(exercise())
    assert websocket.payloads == [{"message": "Attendance marked"}]
