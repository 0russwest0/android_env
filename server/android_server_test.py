"""Basic tests for android_env FastAPI server using the Fake simulator.

These tests do not require a real Android emulator. They load the server,
initialize a Fake simulator-based environment with the bundled dummy task,
and exercise core endpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from server.android_server import app


def _dummy_task_path() -> str:
  # .../android_env/server -> parents[1] is /android_env
  return str((Path(__file__).resolve().parents[1] / "tasks" / "dummy.textproto"))


def test_load_fake_and_core_endpoints():
  # Ensure autoload is disabled for this test
  os.environ.pop("ANDROID_ENV_AUTOLOAD", None)

  with TestClient(app) as client:
    # Load Fake environment
    req = {
        "task_path": _dummy_task_path(),
        "screen_height": 960,
        "screen_width": 540,
        "verbose_logs": False,
        "interaction_rate_sec": 0.0,
    }
    r = client.post("/load/fake", data=json.dumps(req))
    assert r.status_code == 200, r.text
    assert r.json()["status"] in {"success", "skipped"}

    # Health should be OK
    r = client.get("/health")
    assert r.status_code == 200, r.text
    assert r.json()["initialized"] is True

    # Specs should be present
    r = client.get("/specs")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "action_spec" in body and "observation_spec" in body
    assert "pixels" in body["observation_spec"]

    # Reset should return a timestep-like response
    r = client.post("/reset", params={"include_pixels": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(["step_type", "reward", "discount", "observation"]) <= set(body.keys())

    # Observation endpoint
    r = client.get("/observation", params={"include_pixels": False})
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), dict)

    # Screenshot endpoint (PNG)
    r = client.get("/screenshot")
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("image/png")

    # Close should succeed
    r = client.post("/close")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "success"



def test_send_action_and_fetch_screenshot():
  # Ensure autoload is disabled for this test
  os.environ.pop("ANDROID_ENV_AUTOLOAD", None)

  with TestClient(app) as client:
    # Load Fake environment
    req = {
        "task_path": _dummy_task_path(),
        "screen_height": 960,
        "screen_width": 540,
        "verbose_logs": False,
        "interaction_rate_sec": 0.0,
    }
    r = client.post("/load/fake", data=json.dumps(req))
    assert r.status_code == 200, r.text

    # Reset once
    r = client.post("/reset", params={"include_pixels": False})
    assert r.status_code == 200, r.text

    # Send a click action (center)
    action = {"action": "click", "coordinate": [0.5, 0.5]}
    r = client.post("/step", params={"include_pixels": False}, json={"action": action})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(["step_type", "reward", "discount", "observation"]) <= set(body.keys())

    # Fetch screenshot (PNG)
    r = client.get("/screenshot")
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("image/png")

    # Close
    r = client.post("/close")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "success"

