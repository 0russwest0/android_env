"""FastAPI server for managing and interacting with AndroidEnv.

This server mirrors the android_world server style but targets the
android_env library directly, exposing endpoints to load an environment
(emulator or fake), reset/step, inspect specs/observations, manage state,
and perform health checks. It is intended for future Dockerization.
"""

from __future__ import annotations

import contextlib
import os
import typing
from typing import Any

import fastapi
import numpy as np
import pydantic
import uvicorn

from android_env import env_interface as ae_interface
from android_env import loader as ae_loader
from android_env.components import config_classes as ae_config
from android_env.proto import state_pb2


class SpecField(pydantic.BaseModel):
  name: str | None = None
  shape: list[int] | None = None
  dtype: str | None = None
  minimum: float | int | None = None
  maximum: float | int | None = None


class StepResponse(pydantic.BaseModel):
  step_type: str
  reward: float | None
  discount: float | None
  observation: dict[str, typing.Any]


class LoadEmulatorRequest(pydantic.BaseModel):
  task_path: str
  emulator_path: str = "/opt/android/emulator/emulator"
  android_sdk_root: str = "/opt/android"
  avd_name: str = ""
  android_avd_home: str = "~/.android/avd"
  snapshot_name: str = ""
  run_headless: bool = True
  gpu_mode: str = ae_config.GPUMode.SWIFTSHADER_INDIRECT.value
  adb_path: str = "/opt/android/platform-tools/adb"
  adb_server_port: int = 5037
  device_name: str = ""
  verbose_logs: bool = False
  interaction_rate_sec: float = 0.0
  # If all three are provided (non-zero), connects to an existing emulator
  adb_port: int = 0
  emulator_console_port: int = 0
  grpc_port: int = 0


class LoadFakeRequest(pydantic.BaseModel):
  task_path: str
  screen_height: int = 960
  screen_width: int = 540
  verbose_logs: bool = False
  interaction_rate_sec: float = 0.0


class SaveOrLoadStateRequest(pydantic.BaseModel):
  args: dict[str, str] = {}


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
  app.state.android_env = None
  # Optional auto-load via environment variables
  # ANDROID_ENV_AUTOLOAD=true|1 enables autoload
  try:
    autoload = os.environ.get("ANDROID_ENV_AUTOLOAD", "").lower() in {"1", "true", "yes"}
    if autoload:
      mode = os.environ.get("ANDROID_ENV_MODE", "emulator").lower()
      task_path = os.environ.get("ANDROID_ENV_TASK_PATH", "")
      if mode == "emulator" and task_path:
        req = LoadEmulatorRequest(
            task_path=task_path,
            adb_path=os.environ.get("ADB_PATH", "/opt/android/platform-tools/adb"),
            adb_server_port=int(os.environ.get("ADB_SERVER_PORT", "5037")),
            verbose_logs=os.environ.get("VERBOSE_LOGS", "false").lower() in {"1", "true", "yes"},
            interaction_rate_sec=float(os.environ.get("INTERACTION_RATE_SEC", "0.0")),
            adb_port=int(os.environ.get("ADB_PORT", "5555")),
            emulator_console_port=int(os.environ.get("EMULATOR_CONSOLE_PORT", "5554")),
            grpc_port=int(os.environ.get("GRPC_PORT", "8554")),
        )
        # Only set the three ports to connect an existing emulator; other
        # emulator_launcher fields are ignored in this mode.
        cfg = ae_config.AndroidEnvConfig(
            task=ae_config.FilesystemTaskConfig(path=req.task_path),
            simulator=ae_config.EmulatorConfig(
                verbose_logs=req.verbose_logs,
                interaction_rate_sec=req.interaction_rate_sec,
                emulator_launcher=ae_config.EmulatorLauncherConfig(
                    adb_port=req.adb_port,
                    emulator_console_port=req.emulator_console_port,
                    grpc_port=req.grpc_port,
                ),
                adb_controller=ae_config.AdbControllerConfig(
                    adb_path=req.adb_path,
                    adb_server_port=req.adb_server_port,
                ),
            ),
        )
        try:
          app.state.android_env = ae_loader.load(cfg)
        except Exception as exc:
          # Don't block server startup; just warn and continue.
          import logging as _logging  # local import to avoid global if removed
          _logging.warning(
              "Autoload failed (emulator connect on ports %s/%s/%s): %s",
              req.adb_port,
              req.emulator_console_port,
              req.grpc_port,
              exc,
          )
          app.state.android_env = None
      elif mode == "fake" and task_path:
        screen_h = int(os.environ.get("SCREEN_HEIGHT", "960"))
        screen_w = int(os.environ.get("SCREEN_WIDTH", "540"))
        cfg = ae_config.AndroidEnvConfig(
            task=ae_config.FilesystemTaskConfig(path=task_path),
            simulator=ae_config.FakeSimulatorConfig(
                verbose_logs=os.environ.get("VERBOSE_LOGS", "false").lower() in {"1", "true", "yes"},
                interaction_rate_sec=float(os.environ.get("INTERACTION_RATE_SEC", "0.0")),
                screen_dimensions=(screen_h, screen_w),
            ),
        )
        try:
          app.state.android_env = ae_loader.load(cfg)
        except Exception as exc:
          import logging as _logging  # local import
          _logging.warning("Autoload failed (fake): %s", exc)
          app.state.android_env = None
  except Exception:
    # Autoload is best-effort; proceed even if it fails
    app.state.android_env = None
  yield
  # Shutdown
  if app.state.android_env is not None:
    app.state.android_env.close()
    app.state.android_env = None


app = fastapi.FastAPI(lifespan=lifespan)


def _ensure_env(request: fastapi.Request) -> ae_interface.AndroidEnvInterface:
  env = request.app.state.android_env
  if not isinstance(env, ae_interface.AndroidEnvInterface):
    raise fastapi.HTTPException(status_code=500, detail="Environment not loaded")
  return env


AndroidEnvDep = typing.Annotated[
    ae_interface.AndroidEnvInterface, fastapi.Depends(_ensure_env)
]


def _spec_to_model(spec_obj: Any, name: str | None = None) -> SpecField:
  # dm_env specs expose dtype, shape, minimum, maximum sometimes
  shape = list(spec_obj.shape) if getattr(spec_obj, "shape", None) is not None else None
  dtype = str(getattr(spec_obj, "dtype", None)) if getattr(spec_obj, "dtype", None) is not None else None
  minimum = getattr(spec_obj, "minimum", None)
  maximum = getattr(spec_obj, "maximum", None)
  return SpecField(name=name, shape=shape, dtype=dtype, minimum=minimum, maximum=maximum)


def _timestep_to_response(ts: Any, include_pixels: bool) -> StepResponse:
  # ts is dm_env.TimeStep
  obs_out: dict[str, Any] = {}
  if ts.observation is not None:
    for k, v in ts.observation.items():
      if k == "pixels":
        if include_pixels:
          obs_out[k] = np.asarray(v).tolist()
      else:
        # Ensure JSON-serializable types
        if hasattr(v, "tolist"):
          obs_out[k] = v.tolist()
        else:
          obs_out[k] = typing.cast(Any, v)
  return StepResponse(
      step_type=str(getattr(ts.step_type, "name", ts.step_type)),
      reward=None if ts.reward is None else float(ts.reward),
      discount=None if ts.discount is None else float(ts.discount),
      observation=obs_out,
  )


@app.post("/load/emulator")
async def load_emulator(req: LoadEmulatorRequest, request: fastapi.Request, replace: bool = True):
  cfg = ae_config.AndroidEnvConfig(
      task=ae_config.FilesystemTaskConfig(path=req.task_path),
      simulator=ae_config.EmulatorConfig(
          verbose_logs=req.verbose_logs,
          interaction_rate_sec=req.interaction_rate_sec,
          emulator_launcher=ae_config.EmulatorLauncherConfig(
              emulator_path=req.emulator_path,
              android_sdk_root=req.android_sdk_root,
              avd_name=req.avd_name,
              android_avd_home=req.android_avd_home,
              snapshot_name=req.snapshot_name,
              run_headless=req.run_headless,
              gpu_mode=req.gpu_mode,
              adb_port=req.adb_port,
              emulator_console_port=req.emulator_console_port,
              grpc_port=req.grpc_port,
          ),
          adb_controller=ae_config.AdbControllerConfig(
              adb_path=req.adb_path,
              adb_server_port=req.adb_server_port,
              device_name=req.device_name,
          ),
      ),
  )

  try:
    env = ae_loader.load(cfg)
  except Exception as exc:  # pylint: disable=broad-except
    raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc

  # Replace any existing env
  if request.app.state.android_env is not None and replace:
    try:
      request.app.state.android_env.close()
    except Exception:  # pylint: disable=broad-except
      pass
  if request.app.state.android_env is None or replace:
    request.app.state.android_env = env
  else:
    # Do not replace
    return {"status": "skipped", "message": "Environment already loaded and replace=false"}
  return {"status": "success"}


@app.post("/load/fake")
async def load_fake(req: LoadFakeRequest, request: fastapi.Request, replace: bool = True):
  cfg = ae_config.AndroidEnvConfig(
      task=ae_config.FilesystemTaskConfig(path=req.task_path),
      simulator=ae_config.FakeSimulatorConfig(
          verbose_logs=req.verbose_logs,
          interaction_rate_sec=req.interaction_rate_sec,
          screen_dimensions=(req.screen_height, req.screen_width),
      ),
  )

  try:
    env = ae_loader.load(cfg)
  except Exception as exc:  # pylint: disable=broad-except
    raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc

  if request.app.state.android_env is not None and replace:
    try:
      request.app.state.android_env.close()
    except Exception:  # pylint: disable=broad-except
      pass
  if request.app.state.android_env is None or replace:
    request.app.state.android_env = env
  else:
    return {"status": "skipped", "message": "Environment already loaded and replace=false"}
  return {"status": "success"}


@app.get("/specs")
async def get_specs(app_env: AndroidEnvDep):
  action_spec = {k: _spec_to_model(v, name=k) for k, v in app_env.action_spec().items()}
  obs_spec = {k: _spec_to_model(v, name=k) for k, v in app_env.observation_spec().items()}
  return {
      "action_spec": {k: v.model_dump() for k, v in action_spec.items()},
      "observation_spec": {k: v.model_dump() for k, v in obs_spec.items()},
  }


@app.post("/reset")
async def reset(app_env: AndroidEnvDep, include_pixels: bool = False):
  ts = app_env.reset()
  return _timestep_to_response(ts, include_pixels=include_pixels).model_dump()


@app.post("/step")
async def step(app_env: AndroidEnvDep, action: dict[str, typing.Any], include_pixels: bool = False):
  # Convert inputs to numpy arrays where applicable
  np_action: dict[str, np.ndarray] = {}
  for k, v in action.items():
    if isinstance(v, list):
      np_action[k] = np.asarray(v)
    else:
      np_action[k] = np.asarray(v)
  ts = app_env.step(np_action)
  return _timestep_to_response(ts, include_pixels=include_pixels).model_dump()


@app.get("/observation")
async def get_observation(app_env: AndroidEnvDep, include_pixels: bool = False):
  obs = app_env.raw_observation
  out: dict[str, Any] = {}
  for k, v in obs.items():
    if k == "pixels":
      if include_pixels:
        out[k] = np.asarray(v).tolist()
    else:
      out[k] = v.tolist() if hasattr(v, "tolist") else v
  return out


@app.get("/screenshot")
async def get_screenshot(app_env: AndroidEnvDep, flatten: bool = True):
  obs = app_env.raw_observation
  pixels = np.asarray(obs.get("pixels")) if obs else None
  if pixels is None:
    # Force a reset to populate observation
    ts = app_env.reset()
    pixels = np.asarray(ts.observation.get("pixels")) if ts.observation is not None else None
  if pixels is None:
    return {"pixels": []}
  if flatten:
    return {"pixels": pixels.flatten().tolist()}
  return {"pixels": pixels.tolist()}


@app.get("/stats")
async def get_stats(app_env: AndroidEnvDep):
  return app_env.stats()


@app.post("/save_state")
async def save_state(req: SaveOrLoadStateRequest, app_env: AndroidEnvDep):
  response = app_env.save_state(state_pb2.SaveStateRequest(args=dict(req.args)))
  return {
      "status": int(response.status),
      "error_message": response.error_message,
      "additional_info": dict(response.additional_info),
  }


@app.post("/load_state")
async def load_state(req: SaveOrLoadStateRequest, app_env: AndroidEnvDep):
  response = app_env.load_state(state_pb2.LoadStateRequest(args=dict(req.args)))
  return {
      "status": int(response.status),
      "error_message": response.error_message,
      "additional_info": dict(response.additional_info),
  }


@app.post("/close")
async def close(request: fastapi.Request):
  if request.app.state.android_env is not None:
    request.app.state.android_env.close()
    request.app.state.android_env = None
  return {"status": "success"}


@app.get("/health")
async def health(request: fastapi.Request):
  env = request.app.state.android_env
  return {"initialized": isinstance(env, ae_interface.AndroidEnvInterface)}


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=5000)


