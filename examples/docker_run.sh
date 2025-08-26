docker run --rm \
    --device /dev/kvm \
    --name android-env \
    -p 5000:5000 \
    -v android-sdk:/opt/android \
    -e ANDROID_ENV_AUTOLOAD=true \
    -e ANDROID_ENV_MODE=emulator \
    -e ANDROID_ENV_TASK_PATH=/tasks/dummy.textproto \
    --cpus=2 \
    --memory=4g \
    android-env