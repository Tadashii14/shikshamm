import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js is needed to exercise the browser camera helper")
@pytest.mark.parametrize("template", ["admin.html", "student.html"])
def test_empty_camera_inventory_still_requests_browser_and_skips_remote_local_camera(template):
    html = (ROOT / "app" / "templates" / template).read_text(encoding="utf-8")
    start = html.index("      const CAMERA_REQUEST_TIMEOUT_MS = 15000;")
    end = html.index("      function stopCameraSource(source, videoElement, imageElement) {", start)
    camera_functions = html[start:end]

    harness = r"""
const vm = require('node:vm');
const cameraFunctions = JSON.parse(process.argv[1]);
const calls = { capture: 0, localFallback: 0 };
const requests = [];
let failBrowserCamera = false;
const context = {
  navigator: {
    mediaDevices: {
      enumerateDevices: async () => [],
      getUserMedia: async constraints => {
        calls.capture += 1;
        requests.push(constraints);
        if (failBrowserCamera) {
          const error = new Error('Requested device not found');
          error.name = 'NotFoundError';
          throw error;
        }
        return { getTracks: () => [] };
      }
    }
  },
  window: { location: { hostname: 'attendai-api.onrender.com' } },
  fetch: async () => {
    calls.localFallback += 1;
    return { ok: false, json: async () => ({ detail: 'Local camera is available only on this computer.' }) };
  },
  setTimeout,
  clearTimeout
};
vm.runInNewContext(cameraFunctions, context);
(async () => {
  let stream = null;
  let inventoryErrorName = '';
  try {
    stream = await context.getUserMediaSafe();
  } catch (error) {
    inventoryErrorName = error.name;
  }
  failBrowserCamera = true;
  let remoteErrorName = '';
  try {
    await context.startCameraSource({ srcObject: null, hidden: true }, { hidden: true });
  } catch (error) {
    remoteErrorName = error.name;
  }
  console.log(JSON.stringify({
    streamReturned: Boolean(stream),
    inventoryErrorName,
    captureCalls: calls.capture,
    requestedWidth: requests[0].video.width.ideal,
    requestedHeight: requests[0].video.height.ideal,
    localFallbackCalls: calls.localFallback,
    remoteErrorName
  }));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [NODE, "-e", harness, json.dumps(camera_functions)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome == {
        "streamReturned": True,
        "inventoryErrorName": "",
        "captureCalls": 3,
        "requestedWidth": 640,
        "requestedHeight": 480,
        "localFallbackCalls": 0,
        "remoteErrorName": "NotFoundError",
    }
