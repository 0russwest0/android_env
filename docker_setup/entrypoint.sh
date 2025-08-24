#!/bin/bash
set -euo pipefail

# Ensure PATH contains potential cmdline-tools locations
export PATH="$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/cmdline-tools/tools/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH"

ANDROID_CMD_DEFAULT="commandlinetools-linux-11076708_latest.zip"
SDK_PKGS_DEFAULT="${EMULATOR_PACKAGE:-} platform-tools emulator"

ensure_cmdline_tools() {
  if command -v sdkmanager >/dev/null 2>&1; then
    return
  fi
  echo "sdkmanager not found. Installing cmdline-tools to mounted SDK..."
  mkdir -p "$ANDROID_SDK_ROOT/cmdline-tools"
  pushd "$ANDROID_SDK_ROOT/cmdline-tools" >/dev/null
  curl -sSL -o /tmp/cmdtools.zip "https://dl.google.com/android/repository/${ANDROID_CMD:-$ANDROID_CMD_DEFAULT}"
  unzip -q /tmp/cmdtools.zip -d tmp
  # Normalize into 'latest'
  rm -rf latest
  mkdir -p latest
  mv tmp/*/* latest/ || mv tmp/* latest/ || true
  rm -rf tmp /tmp/cmdtools.zip
  popd >/dev/null
}

ensure_sdk_packages() {
  local pkgs
  pkgs="${ANDROID_SDK_PACKAGES:-$SDK_PKGS_DEFAULT}"
  echo "Ensuring SDK packages: $pkgs"
  yes Y | sdkmanager --licenses >/dev/null
  yes Y | sdkmanager --verbose --no_https $pkgs || true
}

ensure_avd() {
  local avd_dir
  avd_dir="$HOME/.android/avd/${EMULATOR_NAME}.avd"
  if [ -d "$avd_dir" ]; then
    return
  fi
  echo "Creating AVD ${EMULATOR_NAME}..."
  echo "no" | avdmanager --verbose create avd --force --name "${EMULATOR_NAME}" --device "${DEVICE_NAME}" --package "${EMULATOR_PACKAGE}"
}

# If host SDK is mounted, install missing pieces into it at runtime.
ensure_cmdline_tools
ensure_sdk_packages
ensure_avd

# Start Emulator headless then start server
./docker_setup/start_emu_headless.sh
adb root || true
python3 -m android_env.server.android_server | cat


