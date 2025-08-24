# coding=utf-8
# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper adapting the Mobile Agent action space (MobileAgentAction) via ADB.

MobileAgentAction here explicitly refers to the action schema used by
"Mobile Agent". This wrapper accepts those high-level actions (click, swipe,
type, key, system_button, open, wait, answer, terminate), translates them to
AndroidEnv ADB requests, and then issues a no-op REPEAT to fetch the freshest
observation from the underlying environment.
"""

from __future__ import annotations

import json
import time
import unicodedata
from typing import Any, Dict, Iterable

from absl import logging
from android_env.components import action_type as android_action_type_lib
from android_env.wrappers import base_wrapper
from android_env.proto import adb_pb2
import dm_env
from dm_env import specs
import numpy as np


def _adb_text_format(text: str) -> str:
  """Prepares text for use with adb input text."""
  to_escape = [
      '\\',
      ';',
      '|',
      '`',
      '\r',
      ' ',
      "'",
      '"',
      '&',
      '<',
      '>',
      '(',
      ')',
      '#',
      '$',
  ]
  for char in to_escape:
    text = text.replace(char, '\\' + char)
  normalized_text = unicodedata.normalize('NFKD', text)
  return normalized_text.encode('ascii', 'ignore').decode('ascii')


def _split_words_and_newlines(text: str) -> Iterable[str]:
  """Split lines of text into individual words and newline chars."""
  lines = text.split('\n')
  for i, line in enumerate(lines):
    words = line.split(' ')
    for j, word in enumerate(words):
      if word:
        yield word
      if j < len(words) - 1:
        yield '%s'
    if i < len(lines) - 1:
      yield '\n'


class MobileAgentActionWrapper(base_wrapper.BaseWrapper):
  """AndroidEnv wrapper adapting the Mobile Agent action space via ADB."""

  def __init__(self, env, *, post_action_sleep_sec: float = 0.0) -> None:
    super().__init__(env)
    self._display_width_px, self._display_height_px = self._get_screen_dimensions()
    self._post_action_sleep_sec = float(post_action_sleep_sec)
    self._init_state()

    # System button mapping for ADB
    self._system_button_mapping = {
        'back': adb_pb2.AdbRequest.PressButton.Button.BACK,
        'home': adb_pb2.AdbRequest.PressButton.Button.HOME,
        'enter': adb_pb2.AdbRequest.PressButton.Button.ENTER,
    }

  def _init_state(self) -> None:
    self._last_answer = ''
    self._termination_status = ''
    self._last_action_success = True

  def _get_screen_dimensions(self) -> tuple[int, int]:
    """Gets screen dimensions from observation_spec."""
    try:
      obs_spec = self._env.observation_spec()
      pixels_spec = obs_spec['pixels']
      height, width = pixels_spec.shape[0], pixels_spec.shape[1]
      logging.info('Screen dimensions: %dx%d', width, height)
      return width, height
    except Exception as e:  # pylint: disable=broad-except
      logging.error('Failed to get screen dimensions: %s, defaulting to 1080x2340', e)
      return 1080, 2340

  def action_spec(self) -> Dict[str, specs.Array]:
    """Returns the spec for the Mobile Agent action space (MobileAgentAction)."""
    # Use fixed-length Unicode dtype for string specs.
    U = np.dtype('<U64')
    return {
        'action': specs.Array(shape=(), dtype=U, name='action'),
        'coordinate': specs.BoundedArray(
            shape=(2,),
            dtype=np.int32,
            minimum=[0, 0],
            maximum=[self._display_width_px - 1, self._display_height_px - 1],
            name='coordinate',
        ),
        'coordinate2': specs.BoundedArray(
            shape=(2,),
            dtype=np.int32,
            minimum=[0, 0],
            maximum=[self._display_width_px - 1, self._display_height_px - 1],
            name='coordinate2',
        ),
        'text': specs.Array(shape=(), dtype=U, name='text'),
        'time': specs.BoundedArray(
            shape=(),
            dtype=np.float32,
            minimum=0.0,
            maximum=10.0,
            name='time',
        ),
        'button': specs.Array(shape=(), dtype=U, name='button'),
        'status': specs.Array(shape=(), dtype=U, name='status'),
    }

  # ---------------------------------------------------------------------------
  # Action processing helpers
  # ---------------------------------------------------------------------------

  def _to_px(self, x: float | int, y: float | int) -> tuple[int, int]:
    """Converts possibly-normalized coordinates to pixel coordinates."""
    try:
      xf = float(x)
      yf = float(y)
    except Exception:  # Fallback if they are not numbers
      return 0, 0

    if 0.0 <= xf <= 1.0 and 0.0 <= yf <= 1.0:
      x_px = int(xf * (self._display_width_px - 1))
      y_px = int(yf * (self._display_height_px - 1))
    else:
      x_px = int(round(xf))
      y_px = int(round(yf))

    x_px = max(0, min(self._display_width_px - 1, x_px))
    y_px = max(0, min(self._display_height_px - 1, y_px))
    return x_px, y_px

  def _process_action(self, action: Dict[str, Any] | str) -> bool:
    """Processes an action from the Mobile Agent action space via ADB."""
    if isinstance(action, str):
      try:
        action = json.loads(action)
      except json.JSONDecodeError as e:
        logging.error('Failed to parse action JSON: %s', e)
        return False

    action_type_str = str(action.get('action', '')).strip().lower()

    try:
      if action_type_str == 'click':
        return self._handle_click(action)
      elif action_type_str == 'long_press':
        return self._handle_long_press(action)
      elif action_type_str == 'swipe':
        return self._handle_swipe(action)
      elif action_type_str == 'type':
        return self._handle_type(action)
      elif action_type_str == 'key':
        return self._handle_key(action)
      elif action_type_str == 'system_button':
        return self._handle_system_button(action)
      elif action_type_str == 'open':
        return self._handle_open_app(action)
      elif action_type_str == 'wait':
        return self._handle_wait(action)
      elif action_type_str == 'answer':
        return self._handle_answer(action)
      elif action_type_str == 'terminate':
        return self._handle_terminate(action)
      else:
        logging.warning('Unknown action type: %s', action_type_str)
        return False
    except Exception as e:  # pylint: disable=broad-except
      logging.error('Error processing action %r: %s', action, e)
      return False

  # ---------------------------------------------------------------------------
  # Specific action handlers
  # ---------------------------------------------------------------------------

  def _handle_click(self, action: Dict[str, Any]) -> bool:
    coordinate = action.get('coordinate', [0, 0])
    x, y = self._to_px(coordinate[0], coordinate[1])

    adb_request = adb_pb2.AdbRequest()
    adb_request.tap.x = x
    adb_request.tap.y = y
    adb_request.timeout_sec = 5.0

    try:
      response = self.execute_adb_call(adb_request)
      if response.status != adb_pb2.AdbResponse.Status.OK:
        logging.error('Click failed: %s', response.error_message)
        return False
      return True
    except Exception as e:  # pylint: disable=broad-except
      logging.error('ADB click error: %s', e)
      return False

  def _handle_long_press(self, action: Dict[str, Any]) -> bool:
    coordinate = action.get('coordinate', [0, 0])
    duration_ms = int(float(action.get('time', 1.0)) * 1000)
    x, y = self._to_px(coordinate[0], coordinate[1])

    adb_request = adb_pb2.AdbRequest()
    adb_request.generic.args.extend([
        'shell', 'input', 'swipe', str(x), str(y), str(x), str(y), str(duration_ms)
    ])
    adb_request.timeout_sec = 5.0

    try:
      response = self.execute_adb_call(adb_request)
      if response.status != adb_pb2.AdbResponse.Status.OK:
        logging.error('Long press failed: %s', response.error_message)
        return False
      return True
    except Exception as e:  # pylint: disable=broad-except
      logging.error('ADB long press error: %s', e)
      return False

  def _handle_swipe(self, action: Dict[str, Any]) -> bool:
    start_coord = action.get('coordinate', [0, 0])
    end_coord = action.get('coordinate2', [0, 0])
    duration_ms = int(float(action.get('time', 0.0)) * 1000)

    x1, y1 = self._to_px(start_coord[0], start_coord[1])
    x2, y2 = self._to_px(end_coord[0], end_coord[1])

    adb_request = adb_pb2.AdbRequest()
    args = ['shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2)]
    if duration_ms > 0:
      args.append(str(duration_ms))
    adb_request.generic.args.extend(args)
    adb_request.timeout_sec = 5.0

    try:
      response = self.execute_adb_call(adb_request)
      if response.status != adb_pb2.AdbResponse.Status.OK:
        logging.error('Swipe failed: %s', response.error_message)
        return False
      return True
    except Exception as e:  # pylint: disable=broad-except
      logging.error('ADB swipe error: %s', e)
      return False

  def _handle_type(self, action: Dict[str, Any]) -> bool:
    text = str(action.get('text', ''))
    if not text:
      logging.warning('Empty text for type action')
      return False

    for word in _split_words_and_newlines(text):
      if word == '\n':
        logging.info('Found \\n, pressing enter button.')
        self._handle_system_button({'button': 'enter'})
        continue
      formatted = _adb_text_format(word)
      logging.info('Typing word: %r', formatted)
      adb_request = adb_pb2.AdbRequest()
      adb_request.input_text.text = formatted
      adb_request.timeout_sec = 5.0

      try:
        response = self.execute_adb_call(adb_request)
        if response.status != adb_pb2.AdbResponse.Status.OK:
          logging.error('Text input failed: %s', response.error_message)
          return False
      except Exception as e:  # pylint: disable=broad-except
        logging.error('ADB text input error: %s', e)
        return False
    return True

  def _handle_key(self, action: Dict[str, Any]) -> bool:
    key_name = str(action.get('text', '')).strip().lower()
    key_mapping = {
        'volume_up': 'KEYCODE_VOLUME_UP',
        'volume_down': 'KEYCODE_VOLUME_DOWN',
        'power': 'KEYCODE_POWER',
        'camera': 'KEYCODE_CAMERA',
        'clear': 'KEYCODE_CLEAR',
        'back': 'KEYCODE_BACK',
        'esc': 'KEYCODE_BACK',
        'home': 'KEYCODE_HOME',
        'menu': 'KEYCODE_MENU',
        'enter': 'KEYCODE_ENTER',
        'return': 'KEYCODE_ENTER',
    }

    if key_name not in key_mapping:
      logging.warning('Unknown key: %s', key_name)
      return False

    adb_request = adb_pb2.AdbRequest()
    adb_request.generic.args.extend(['shell', 'input', 'keyevent', key_mapping[key_name]])
    adb_request.timeout_sec = 5.0

    try:
      response = self.execute_adb_call(adb_request)
      if response.status != adb_pb2.AdbResponse.Status.OK:
        logging.error('Key event failed: %s', response.error_message)
        return False
      return True
    except Exception as e:  # pylint: disable=broad-except
      logging.error('ADB key event error: %s', e)
      return False

  def _handle_system_button(self, action: Dict[str, Any]) -> bool:
    button_name = str(action.get('button', '')).strip().lower()

    if button_name == 'menu':
      return self._handle_key({'text': 'menu'})

    if button_name not in self._system_button_mapping:
      logging.error('Unknown system button: %s', button_name)
      return False

    adb_request = adb_pb2.AdbRequest()
    adb_request.press_button.button = self._system_button_mapping[button_name]
    adb_request.timeout_sec = 5.0

    try:
      response = self.execute_adb_call(adb_request)
      if response.status != adb_pb2.AdbResponse.Status.OK:
        logging.error('System button press failed: %s', response.error_message)
        return False
      return True
    except Exception as e:  # pylint: disable=broad-except
      logging.error('ADB system button error: %s', e)
      return False

  def _handle_open_app(self, action: Dict[str, Any]) -> bool:
    app_name = str(action.get('text', '')).strip()
    if not app_name:
      logging.error('No app name provided for open action')
      return False

    adb_request = adb_pb2.AdbRequest()
    adb_request.start_activity.full_activity = app_name
    adb_request.start_activity.force_stop = True
    adb_request.timeout_sec = 10.0

    try:
      response = self.execute_adb_call(adb_request)
      if response.status != adb_pb2.AdbResponse.Status.OK:
        logging.error('App opening failed: %s', response.error_message)
        return False
      return True
    except Exception as e:  # pylint: disable=broad-except
      logging.error('ADB app opening error: %s', e)
      return False

  def _handle_wait(self, action: Dict[str, Any]) -> bool:
    wait_time = float(action.get('time', 1.0))
    try:
      time.sleep(wait_time)
      logging.info('Waited for %.3f seconds', wait_time)
      return True
    except Exception as e:  # pylint: disable=broad-except
      logging.error('Wait error: %s', e)
      return False

  def _handle_answer(self, action: Dict[str, Any]) -> bool:
    answer = str(action.get('text', ''))
    logging.info('Agent answer: %s', answer)
    self._last_answer = answer
    return True

  def _handle_terminate(self, action: Dict[str, Any]) -> bool:
    status = str(action.get('status', 'success'))
    logging.info('Task terminated with status: %s', status)
    self._termination_status = status
    return True

  # ---------------------------------------------------------------------------
  # BaseWrapper overrides
  # ---------------------------------------------------------------------------

  def step(self, action: Any) -> dm_env.TimeStep:
    """Executes the ADB action, then fetches the latest state with REPEAT."""
    is_terminate_action = False
    if isinstance(action, dict) and str(action.get('action', '')).lower() == 'terminate':
      is_terminate_action = True
    elif isinstance(action, str):
      try:
        parsed_action = json.loads(action)
        if str(parsed_action.get('action', '')).lower() == 'terminate':
          is_terminate_action = True
      except json.JSONDecodeError:
        pass

    action_success = self._process_action(action)
    self._last_action_success = bool(action_success)

    if self._post_action_sleep_sec > 0:
      time.sleep(self._post_action_sleep_sec)

    # Issue a minimal REPEAT to obtain the freshest observation.
    parent_spec = self._env.action_spec()
    no_op_action = {
        'action_type': np.array(
            android_action_type_lib.ActionType.REPEAT,
            dtype=parent_spec['action_type'].dtype,
        ),
        'touch_position': np.array(
            [0.5, 0.5], dtype=parent_spec['touch_position'].dtype
        ),
    }
    timestep = self._env.step(no_op_action)

    if is_terminate_action:
      # Convert to a LAST timestep, preserving the latest observation/reward.
      return dm_env.TimeStep(
          step_type=dm_env.StepType.LAST,
          reward=timestep.reward,
          discount=0.0,
          observation=timestep.observation,
      )
    return timestep

  def _reset_state(self) -> None:
    self._init_state()

  def _wrapper_stats(self) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    stats['last_answer'] = self._last_answer
    stats['termination_status'] = self._termination_status
    stats['last_action_success'] = self._last_action_success
    return stats


