from __future__ import annotations

from typing import Dict, Set

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self.user_id_to_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, user_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        connections = self.user_id_to_connections.setdefault(user_id, set())
        connections.add(websocket)

    def disconnect(self, user_id: str, websocket: WebSocket) -> None:
        connections = self.user_id_to_connections.get(user_id)
        if not connections:
            return
        connections.discard(websocket)
        if not connections:
            self.user_id_to_connections.pop(user_id, None)

    async def send_text(self, user_id: str, message: str) -> None:
        connections = self.user_id_to_connections.get(user_id)
        if not connections:
            return
        to_remove: Set[WebSocket] = set()
        for ws in list(connections):
            try:
                await ws.send_text(message)
            except Exception:
                to_remove.add(ws)
        for ws in to_remove:
            self.disconnect(user_id, ws)


manager = ConnectionManager()


async def notify_user(user_id: int | str, message: str) -> None:
    await manager.send_text(str(user_id), message)



