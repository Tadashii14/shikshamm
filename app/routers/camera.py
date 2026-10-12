from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.services.local_camera import CameraUnavailableError, local_camera


router = APIRouter(prefix="/camera", tags=["camera"])


def _require_local_same_origin(request: Request) -> None:
    client_host = request.client.host if request.client else ""
    try:
        if not ipaddress.ip_address(client_host).is_loopback:
            raise HTTPException(status_code=403, detail="Local camera is available only on this computer.")
    except ValueError:
        raise HTTPException(status_code=403, detail="Local camera is available only on this computer.") from None

    fetch_site = request.headers.get("sec-fetch-site", "").lower()
    if fetch_site in {"cross-site", "same-site"}:
        raise HTTPException(status_code=403, detail="Local camera requests must come from this app page.")

    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
        origin_host = urlsplit(origin).netloc
        app_host = request.headers.get("host", "")
        if origin_host.casefold() != app_host.casefold():
            raise HTTPException(status_code=403, detail="Local camera requests must come from this app page.")


@router.get("/status")
def camera_status(request: Request):
    _require_local_same_origin(request)
    try:
        frame = local_camera.latest_frame()
        return {
            "available": True,
            "source": "local OpenCV camera",
            "frame_bytes": len(frame),
            "detail": "Live frames are held in memory and are not saved to disk.",
        }
    except CameraUnavailableError as exc:
        return {
            "available": False,
            "source": "local OpenCV camera",
            "detail": str(exc),
            "setup": "Install the optional local camera dependencies from requirements-local.txt.",
        }


@router.get("/frame")
def camera_frame(request: Request):
    _require_local_same_origin(request)
    try:
        frame = local_camera.latest_frame()
    except CameraUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )
