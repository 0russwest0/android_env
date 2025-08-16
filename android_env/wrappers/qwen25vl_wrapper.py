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

"""Wrapper for adapting Qwen25VL action space to AndroidEnv using direct ADB commands."""

import time
from typing import Any, Dict, Union
import json
import unicodedata
from typing import Iterable

from absl import logging
from android_env.wrappers import base_wrapper
from android_env.proto import adb_pb2
import dm_env
from dm_env import specs
import numpy as np

def _adb_text_format(text: str) -> str:
  """Prepares text for use with adb."""
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


class Qwen25VLWrapper(base_wrapper.BaseWrapper):
  """AndroidEnv wrapper that adapts Qwen25VL action space using direct ADB commands."""

  def __init__(self, env):
    super().__init__(env)
    self._display_width_px, self._display_height_px = self._get_screen_dimensions()
    
    # System button mapping for ADB
    self._system_button_mapping = {
        'Back': adb_pb2.AdbRequest.PressButton.Button.BACK,
        'Home': adb_pb2.AdbRequest.PressButton.Button.HOME, 
        'Enter': adb_pb2.AdbRequest.PressButton.Button.ENTER
    }

  def _get_screen_dimensions(self):
    """get screen dimensions from observation_spec"""
    try:
      obs_spec = self._env.observation_spec()
      pixels_spec = obs_spec['pixels']
      height, width = pixels_spec.shape[0], pixels_spec.shape[1]
      logging.info(f"get screen dimensions: {width}x{height}")
      return width, height
    except Exception as e:
      logging.error(f"get screen dimensions error: {e}，using default value 1080x2340")
      return 1080, 2340

  def action_spec(self) -> Dict[str, specs.Array]:
    """Returns the action specification for Qwen25VL actions."""
    return {
        'action': specs.StringArray(shape=(), name='action'),
        'coordinate': specs.BoundedArray(
            shape=(2,),
            dtype=np.int32,
            minimum=[0, 0],
            maximum=[self._display_width_px, self._display_height_px],
            name='coordinate'
        ),
        'coordinate2': specs.BoundedArray(
            shape=(2,),
            dtype=np.int32,
            minimum=[0, 0],
            maximum=[self._display_width_px, self._display_height_px],
            name='coordinate2'
        ),
        'text': specs.StringArray(shape=(), name='text'),
        'time': specs.BoundedArray(
            shape=(),
            dtype=np.float32,
            minimum=0.0,
            maximum=10.0,
            name='time'
        ),
        'button': specs.StringArray(shape=(), name='button'),
        'status': specs.StringArray(shape=(), name='status')
    }

  def _process_action(self, action: Dict[str, Any]) -> bool:
    """Processes a Qwen25VL action using direct ADB commands."""
    
    if isinstance(action, str):
      # Handle JSON string input
      try:
        action = json.loads(action)
      except json.JSONDecodeError as e:
        logging.error(f"Failed to parse action JSON: {e}")
        return False
    
    action_type_str = action.get('action', '').lower()
    
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
        logging.warning(f"Unknown action type: {action_type_str}")
        return False
        
    except Exception as e:
      logging.error(f"Error processing action {action}: {e}")
      return False

  def _handle_click(self, action: Dict[str, Any]) -> bool:
    """Handles click action using ADB tap command."""
    coordinate = action.get('coordinate', [0, 0])
    x, y = coordinate[0], coordinate[1]
    
    # Clamp coordinates to screen bounds
    x = max(0, min(self._display_width_px, x))
    y = max(0, min(self._display_height_px, y))
    
    adb_request = adb_pb2.AdbRequest()
    adb_request.tap.x = x
    adb_request.tap.y = y
    
    try:
      response = self.execute_adb_call(adb_request)
      if response.status != adb_pb2.AdbResponse.Status.OK:
        logging.error(f"Click failed: {response.error_message}")
        return False
      return True
    except Exception as e:
      logging.error(f"ADB click error: {e}")
      return False

  def _handle_long_press(self, action: Dict[str, Any]) -> bool:
    """Handles long press action using ADB swipe command with same start/end coordinates."""
    coordinate = action.get('coordinate', [0, 0])
    duration_ms = int(action.get('time', 1.0) * 1000)  # Convert to milliseconds
    x, y = coordinate[0], coordinate[1]
    
    # Clamp coordinates
    x = max(0, min(self._display_width_px, x))
    y = max(0, min(self._display_height_px, y))
    
    # Use generic ADB command for long press (swipe with same start/end coordinates)
    adb_request = adb_pb2.AdbRequest()
    adb_request.generic.args.extend([
        'shell', 'input', 'swipe', str(x), str(y), str(x), str(y), str(duration_ms)
    ])
    
    try:
      response = self.execute_adb_call(adb_request)
      if response.status != adb_pb2.AdbResponse.Status.OK:
        logging.error(f"Long press failed: {response.error_message}")
        return False
      return True
    except Exception as e:
      logging.error(f"ADB long press error: {e}")
      return False

  def _handle_swipe(self, action: Dict[str, Any]) -> bool:
    """Handles swipe action using ADB swipe command."""
    start_coord = action.get('coordinate', [0, 0])
    end_coord = action.get('coordinate2', [0, 0])
    
    x1, y1 = start_coord[0], start_coord[1]
    x2, y2 = end_coord[0], end_coord[1]
    
    # Clamp coordinates
    x1 = max(0, min(self._display_width_px, x1))
    y1 = max(0, min(self._display_height_px, y1))
    x2 = max(0, min(self._display_width_px, x2))
    y2 = max(0, min(self._display_height_px, y2))
    
    # Use generic ADB command for swipe
    adb_request = adb_pb2.AdbRequest()
    adb_request.generic.args.extend([
        'shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2)
    ])
    
    try:
      response = self.execute_adb_call(adb_request)
      if response.status != adb_pb2.AdbResponse.Status.OK:
        logging.error(f"Swipe failed: {response.error_message}")
        return False
      return True
    except Exception as e:
      logging.error(f"ADB swipe error: {e}")
      return False

  def _handle_type(self, action: Dict[str, Any]) -> bool:
    """Handles text input action using ADB input text command."""
    text = action.get('text', '')
    if not text:
      logging.warning("Empty text for type action")
      return False
    
    words = _split_words_and_newlines(text)
    for word in words:
      if word == '\n':
        logging.info('Found \\n, pressing enter button.')
        self._handle_system_button({'button': 'Enter'})
        continue
      formatted = _adb_text_format(word)
      logging.info('Attempting to type word: %r', formatted)
      adb_request = adb_pb2.AdbRequest()
      adb_request.input_text.text = formatted

      try:
        response = self.execute_adb_call(adb_request)
        if response.status != adb_pb2.AdbResponse.Status.OK:
          logging.error(f"Text input failed: {response.error_message}")
          return False
      except Exception as e:
        logging.error(f"ADB text input error: {e}")
        return False
    
    return True

  def _handle_key(self, action: Dict[str, Any]) -> bool:
    """Handles key event action using ADB keyevent command."""
    key_name = action.get('text', '').lower()
    
    # Map key names to Android key event codes
    key_mapping = {
        'volume_up': 'KEYCODE_VOLUME_UP',
        'volume_down': 'KEYCODE_VOLUME_DOWN', 
        'power': 'KEYCODE_POWER',
        'camera': 'KEYCODE_CAMERA',
        'clear': 'KEYCODE_CLEAR',
        'back': 'KEYCODE_BACK',
        'home': 'KEYCODE_HOME',
        'menu': 'KEYCODE_MENU',
        'enter': 'KEYCODE_ENTER'
    }
    
    if key_name not in key_mapping:
      logging.warning(f"Unknown key: {key_name}")
      return False
    
    keycode = key_mapping[key_name]
    
    # Use generic ADB command for key event
    adb_request = adb_pb2.AdbRequest()
    adb_request.generic.args.extend(['shell', 'input', 'keyevent', keycode])
    
    try:
      response = self.execute_adb_call(adb_request)
      if response.status != adb_pb2.AdbResponse.Status.OK:
        logging.error(f"Key event failed: {response.error_message}")
        return False
      return True
    except Exception as e:
      logging.error(f"ADB key event error: {e}")
      return False

  def _handle_system_button(self, action: Dict[str, Any]) -> bool:
    """Handles system button press using ADB press_button command."""
    button_name = action.get('button', '')

    if button_name == 'Menu':
      return self._handle_key({'text': 'menu'})
    
    if button_name not in self._system_button_mapping:
      logging.error(f"Unknown system button: {button_name}")
      return False
    
    adb_request = adb_pb2.AdbRequest()
    adb_request.press_button.button = self._system_button_mapping[button_name]
    
    try:
      response = self.execute_adb_call(adb_request)
      if response.status != adb_pb2.AdbResponse.Status.OK:
        logging.error(f"System button press failed: {response.error_message}")
        return False
      return True
    except Exception as e:
      logging.error(f"ADB system button error: {e}")
      return False

  def _handle_open_app(self, action: Dict[str, Any]) -> bool:
    """Handles opening an app using ADB start_activity command."""
    app_name = action.get('text', '')
    if not app_name:
      logging.error("No app name provided for open action")
      return False
    
    adb_request = adb_pb2.AdbRequest()
    adb_request.start_activity.full_activity = app_name
    adb_request.start_activity.force_stop = True
    
    try:
      response = self.execute_adb_call(adb_request)
      if response.status != adb_pb2.AdbResponse.Status.OK:
        logging.error(f"App opening failed: {response.error_message}")
        return False
      return True
    except Exception as e:
      logging.error(f"ADB app opening error: {e}")
      return False

  def _handle_wait(self, action: Dict[str, Any]) -> bool:
    """Handles wait action by sleeping."""
    wait_time = action.get('time', 1.0)
    
    try:
      time.sleep(wait_time)
      logging.info(f"Waited for {wait_time} seconds")
      return True
    except Exception as e:
      logging.error(f"Wait error: {e}")
      return False

  def _handle_answer(self, action: Dict[str, Any]) -> bool:
    """Handles answer action by storing the answer."""
    answer = action.get('text', '')
    logging.info(f"Agent answer: {answer}")
    
    # Store the answer in wrapper stats
    self._last_answer = answer
    return True

  def _handle_terminate(self, action: Dict[str, Any]) -> bool:
    """Handles terminate action by storing the status."""
    status = action.get('status', 'success')
    logging.info(f"Task terminated with status: {status}")
    
    # Store termination status
    self._termination_status = status
    return True

  def step(self, action: Any) -> dm_env.TimeStep:
    """Takes a step by executing the ADB action and then getting environment state."""
    
    # Check if this is a terminate action
    is_terminate_action = False
    if isinstance(action, dict) and action.get('action', '').lower() == 'terminate':
      is_terminate_action = True
    elif isinstance(action, str):
      try:
        parsed_action = json.loads(action)
        if parsed_action.get('action', '').lower() == 'terminate':
          is_terminate_action = True
      except json.JSONDecodeError:
        pass
    
    # Execute the Qwen25VL action via ADB
    action_success = self._process_action(action)

    # Wait for 1 second to ensure the action is executed
    time.sleep(1.0)
    
    # Take a minimal step in the environment to get the updated state
    # We use a no-op action (REPEAT) to just get the current state
    no_op_action = {
        'action_type': np.array(2, dtype=self._env.action_spec()['action_type'].dtype),  # REPEAT
        'touch_position': np.array([0.5, 0.5], dtype=self._env.action_spec()['touch_position'].dtype)
    }
    
    timestep = self._env.step(no_op_action)
    
    # Add action success information to the observation if needed
    if hasattr(self, '_last_action_success'):
      self._last_action_success = action_success
    
    # If this was a terminate action, return a LAST timestep
    if is_terminate_action:
      timestep = timestep._replace(step_type=dm_env.StepType.LAST)
    
    return timestep

  def _reset_state(self):
    """Resets wrapper state."""
    self._last_answer = ""
    self._termination_status = ""
    self._last_action_success = True

  def _wrapper_stats(self) -> Dict[str, Any]:
    """Returns wrapper-specific statistics."""
    stats = {}
    if hasattr(self, '_last_answer'):
      stats['last_answer'] = self._last_answer
    if hasattr(self, '_termination_status'):
      stats['termination_status'] = self._termination_status
    if hasattr(self, '_last_action_success'):
      stats['last_action_success'] = self._last_action_success
    return stats 