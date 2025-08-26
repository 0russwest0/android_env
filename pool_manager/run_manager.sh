#!/bin/bash
set -euo pipefail

# Usage: ./pool_manager/run_manager.sh [--python /path/to/python]

PY=${1:-python3}

export ANDROID_ENV_IMAGE=${ANDROID_ENV_IMAGE:-android-env}
export POOL_TARGET=${POOL_TARGET:-40}
export POOL_MIN_IDLE=${POOL_MIN_IDLE:-8}
export POOL_MAX=${POOL_MAX:-64}
export CONTAINER_CPUS=${CONTAINER_CPUS:-2}
export CONTAINER_MEM=${CONTAINER_MEM:4g}
export ENABLE_KVM=${ENABLE_KVM:-true}
export USE_PRIVILEGED=${USE_PRIVILEGED:-false}
export DEFAULT_TASK_PATH=${DEFAULT_TASK_PATH:-/tasks/dummy.textproto}
export AUTOLOAD_MODE=${AUTOLOAD_MODE:-emulator}
export HOST_INTERFACE=${HOST_INTERFACE:-127.0.0.1}
export DOCKER_NETWORK=${DOCKER_NETWORK:-android-env-net}
export SDK_VOLUME=${SDK_VOLUME:-android-sdk}
export MANAGER_HOST=${MANAGER_HOST:-0.0.0.0}
export MANAGER_PORT=${MANAGER_PORT:-8080}
export PORT_RANGE_START=${PORT_RANGE_START:-5000}
export PORT_RANGE_END=${PORT_RANGE_END:-5100}

# Install manager deps into current environment
if ! ${PY} -c "import fastapi, uvicorn, httpx, docker" >/dev/null 2>&1; then
  echo "Installing pool manager dependencies..."
  ${PY} -m pip install -r pool_manager/requirements.txt
fi

exec ${PY} -m pool_manager.manager


