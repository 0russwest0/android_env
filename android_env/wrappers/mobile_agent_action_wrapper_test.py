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

"""Tests for android_env.wrappers.mobile_agent_action_wrapper."""

from unittest import mock

from absl.testing import absltest
from android_env import env_interface
from android_env.components import action_type as action_type_lib
from android_env.proto import adb_pb2
from android_env.wrappers import mobile_agent_action_wrapper
import dm_env
from dm_env import specs
import numpy as np


class _DummySpec:
  def __init__(self, shape):
    self.shape = shape


def _make_array_spec(shape, dtype, name):
  return specs.BoundedArray(
      name=name,
      shape=shape,
      dtype=dtype,
      minimum=np.zeros(shape),
      maximum=np.ones(shape),
  )


class MobileAgentActionWrapperTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._base_action_spec = {
        'action_type': specs.DiscreteArray(num_values=3, name='action_type'),
        'touch_position': _make_array_spec(
            shape=(2,), dtype=np.float32, name='touch_position'),
    }
    self.base_env = mock.create_autospec(env_interface.AndroidEnvInterface)
    self.base_env.action_spec.return_value = self._base_action_spec
    # Provide pixels shape (H, W, C) = (2340, 1080, 3)
    self.base_env.observation_spec.return_value = {
        'pixels': _DummySpec((2340, 1080, 3))
    }
    # Ensure stats() returns a real dict like a real env would, not a MagicMock.
    self.base_env.stats.return_value = {}

  def test_action_spec_fields(self):
    wrapped_env = mobile_agent_action_wrapper.MobileAgentActionWrapper(
        self.base_env)
    spec = wrapped_env.action_spec()
    self.assertCountEqual(
        spec.keys(),
        ['action', 'coordinate', 'coordinate2', 'text', 'time', 'button', 'status'],
    )
    # Coordinate bounds should reflect screen size (W-1, H-1) = (1079, 2339)
    self.assertTrue(np.all(spec['coordinate'].maximum == np.array([1079, 2339])))
    self.assertTrue(np.all(spec['coordinate'].minimum == np.array([0, 0])))

  def test_click_executes_adb_and_repeat_step(self):
    wrapped_env = mobile_agent_action_wrapper.MobileAgentActionWrapper(
        self.base_env)

    fake_timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=0.5,
        discount=1.0,
        observation={'pixels': np.zeros((2340, 1080, 3), dtype=np.uint8)},
    )
    self.base_env.step.return_value = fake_timestep

    ok_response = adb_pb2.AdbResponse(
        status=adb_pb2.AdbResponse.Status.OK,
        error_message='',
    )
    self.base_env.execute_adb_call.return_value = ok_response

    ts = wrapped_env.step({'action': 'click', 'coordinate': [100, 200]})

    # ADB was invoked
    self.assertTrue(self.base_env.execute_adb_call.called)
    # Then a REPEAT no-op was sent to the parent env
    self.base_env.step.assert_called_once()
    sent = self.base_env.step.call_args[0][0]
    self.assertIn('action_type', sent)
    self.assertEqual(
        sent['action_type'],
        np.array(action_type_lib.ActionType.REPEAT, dtype=self._base_action_spec['action_type'].dtype),
    )
    self.assertEqual(fake_timestep, ts)

  def test_terminate_returns_last(self):
    wrapped_env = mobile_agent_action_wrapper.MobileAgentActionWrapper(
        self.base_env)
    fake_timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=1.0,
        discount=1.0,
        observation={'pixels': np.zeros((2340, 1080, 3), dtype=np.uint8)},
    )
    self.base_env.step.return_value = fake_timestep

    ts = wrapped_env.step({'action': 'terminate', 'status': 'success'})
    # Underlying env was still stepped to refresh observation
    self.base_env.step.assert_called_once()
    self.assertEqual(dm_env.StepType.LAST, ts.step_type)
    self.assertEqual(fake_timestep.reward, ts.reward)
    self.assertEqual(fake_timestep.observation, ts.observation)

  def test_stats_after_answer_and_terminate(self):
    wrapped_env = mobile_agent_action_wrapper.MobileAgentActionWrapper(
        self.base_env)
    # Step once so that base_env.step is invoked (returns anything)
    self.base_env.step.return_value = dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=0.0,
        discount=1.0,
        observation={'pixels': np.zeros((2340, 1080, 3), dtype=np.uint8)},
    )

    _ = wrapped_env.step({'action': 'answer', 'text': 'hello'})
    _ = wrapped_env.step({'action': 'terminate', 'status': 'done'})
    stats = wrapped_env.stats()
    self.assertIn('last_answer', stats)
    self.assertEqual('hello', stats['last_answer'])
    self.assertIn('termination_status', stats)
    self.assertEqual('done', stats['termination_status'])
    self.assertIn('last_action_success', stats)
    self.assertTrue(stats['last_action_success'])


if __name__ == '__main__':
  absltest.main()


