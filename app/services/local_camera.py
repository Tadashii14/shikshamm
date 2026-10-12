"""Small, localhost-only webcam source for browsers without camera pass-through.

Frames stay in memory and are discarded as soon as they are replaced. The
camera is released after the last preview closes, so it is not held open while
the app is idle.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Iterator


class CameraUnavailableError(RuntimeError):
    pass


class LocalCamera:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._frame: bytes | None = None
        self._sequence = 0
        self._error: str | None = None
        self._clients = 0
        self._idle_since: float | None = None
        self._stop = threading.Event()

    def _start(self) -> None:
        with self._condition:
            if self._thread and self._thread.is_alive():
                return
            self._frame = None
            self._error = None
            self._stop.clear()
            self._idle_since = time.monotonic()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="attendai-local-camera",
                daemon=True,
            )
            self._thread.start()

    @staticmethod
    def _open_camera(cv2):
        indices = (0, 1, 2, 3)
        backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if os.name == "nt" else (cv2.CAP_ANY,)
        for index in indices:
            for backend in backends:
                capture = cv2.VideoCapture(index, backend)
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    capture.set(cv2.CAP_PROP_FPS, 15)
                    try:
                        capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    except Exception:
                        pass
                    try:
                        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    return capture
                capture.release()
        return None

    def _capture_loop(self) -> None:
        capture = None
        try:
            import cv2

            capture = self._open_camera(cv2)
            if capture is None:
                raise CameraUnavailableError("No usable local camera was found by OpenCV.")

            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise CameraUnavailableError("The local camera opened but stopped returning frames.")
                height, width = frame.shape[:2]
                if width > 640 or height > 480:
                    scale = min(640 / width, 480 / height)
                    frame = cv2.resize(
                        frame,
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                encoded_ok, encoded = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
                )
                if not encoded_ok:
                    continue
                with self._condition:
                    self._frame = encoded.tobytes()
                    self._sequence += 1
                    self._condition.notify_all()
                    if self._clients:
                        self._idle_since = None
                    else:
                        if self._idle_since is None:
                            self._idle_since = time.monotonic()
                        elif time.monotonic() - self._idle_since >= 15:
                            break
                self._stop.wait(0.015)
        except Exception as exc:
            with self._condition:
                self._error = (
                    str(exc)
                    if isinstance(exc, CameraUnavailableError)
                    else f"Camera startup failed ({exc.__class__.__name__})."
                )
                self._condition.notify_all()
        finally:
            if capture is not None:
                capture.release()
            with self._condition:
                self._frame = None
                self._thread = None
                self._condition.notify_all()

    def latest_frame(self, timeout: float = 8.0) -> bytes:
        self._start()
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._frame is None:
                if self._error:
                    raise CameraUnavailableError(self._error)
                if not self._thread or not self._thread.is_alive():
                    raise CameraUnavailableError("The local camera did not start.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraUnavailableError("Timed out waiting for a local camera frame.")
                self._condition.wait(min(remaining, 0.5))
            if not self._clients:
                self._idle_since = time.monotonic()
            return self._frame

    def frames(self) -> Iterator[bytes]:
        self._start()
        with self._condition:
            self._clients += 1
            self._idle_since = None
            sequence = self._sequence - 1
        try:
            while True:
                with self._condition:
                    while self._sequence <= sequence and not self._error:
                        if not self._thread or not self._thread.is_alive():
                            return
                        self._condition.wait(timeout=1.0)
                    if self._error:
                        return
                    frame = self._frame
                    sequence = self._sequence
                if frame is not None:
                    yield frame
        finally:
            with self._condition:
                self._clients = max(0, self._clients - 1)
                if self._clients == 0:
                    self._idle_since = time.monotonic()


local_camera = LocalCamera()
