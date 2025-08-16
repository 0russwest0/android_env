# Logs
LOG_DIR=/root/android_env/server/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
PORTSERVER_LOG="$LOG_DIR/portserver_$TS.log"
SERVER_LOG="$LOG_DIR/server_$TS.log"
ln -sf "$PORTSERVER_LOG" "$LOG_DIR/portserver_latest.log"
ln -sf "$SERVER_LOG" "$LOG_DIR/server_latest.log"

# Start portserver only if not already running
if ! pgrep -f "/root/miniconda3/envs/android_env/bin/portserver.py.*--portserver_address=@android-env-portserver" >/dev/null 2>&1; then
  /root/miniconda3/envs/android_env/bin/portserver.py --portserver_address=@android-env-portserver --portserver_static_pool 15000-24999 2>&1 | tee -a "$PORTSERVER_LOG" &
  sleep 0.2
fi
export PORTSERVER_ADDRESS=@android-env-portserver

python server_v1.py 2>&1 | tee -a "$SERVER_LOG"