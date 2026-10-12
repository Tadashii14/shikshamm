import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.routers.camera import _require_local_same_origin


def make_request(client_host="127.0.0.1", headers=()):
    encoded_headers = [(key.lower().encode(), value.encode()) for key, value in headers]
    return Request({
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/camera/frame",
        "raw_path": b"/camera/frame",
        "query_string": b"",
        "headers": encoded_headers,
        "server": ("127.0.0.1", 8765),
        "client": (client_host, 45678),
        "root_path": "",
    })


def test_local_camera_allows_same_origin_loopback_request():
    request = make_request(headers=[
        ("host", "127.0.0.1:8765"),
        ("origin", "http://127.0.0.1:8765"),
        ("sec-fetch-site", "same-origin"),
    ])
    _require_local_same_origin(request)


def test_local_camera_rejects_network_clients():
    with pytest.raises(HTTPException) as exc:
        _require_local_same_origin(make_request(client_host="192.168.1.20"))
    assert exc.value.status_code == 403


def test_local_camera_rejects_cross_origin_browser_requests():
    request = make_request(headers=[
        ("host", "127.0.0.1:8765"),
        ("origin", "http://attacker.example"),
        ("sec-fetch-site", "cross-site"),
    ])
    with pytest.raises(HTTPException) as exc:
        _require_local_same_origin(request)
    assert exc.value.status_code == 403
