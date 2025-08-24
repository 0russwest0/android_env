docker run --rm \
    --name android-env \
    -p 5000:5000 \
    -p 8554:8554 \
    -v /root/Android/Sdk:/opt/android \
    --device /dev/kvm \
    -e ANDROID_ENV_AUTOLOAD=true \
    -e ANDROID_ENV_MODE=emulator \
    -e ANDROID_ENV_TASK_PATH=/tasks/dummy.textproto \
    android-env:test