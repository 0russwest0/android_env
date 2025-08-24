docker run --rm \
  -p 5000:5000 -p 8554:8554 \
  --device /dev/kvm \
  -e ANDROID_ENV_AUTOLOAD=true \
  -e ANDROID_ENV_MODE=emulator \
  -e ANDROID_ENV_TASK_PATH=/tasks/dummy.textproto \
  -e EMULATOR_PATH=/opt/android/emulator/emulator \
  -e ANDROID_SDK_ROOT=/opt/android \
  -e ADB_PATH=/opt/android/platform-tools/adb \
  android-env:latest