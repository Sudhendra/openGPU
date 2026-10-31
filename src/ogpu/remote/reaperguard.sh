#!/bin/bash
# Keeps the idle reaper alive: if anything kills it, it is restarted within a minute (a dead reaper = a pod that
# bills until someone notices). Single instance per container via a lock in /tmp, which a restart clears.
OG=/workspace/.ogpu
exec 9>/tmp/ogpu-reaperguard.lock
flock -n 9 || exit 0
while true; do
  pid=$(cut -d' ' -f1 "$OG/reaper.pid" 2>/dev/null)
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "$(date) guard: reaper not running; starting it" >> "$OG/reaper.log"
    (set -a; [ -f "$OG/env.sh" ] && source "$OG/env.sh"; set +a
     setsid nohup python3 "$OG/bin/reaper.py" >> "$OG/reaper.log" 2>&1 < /dev/null 9>&- &)
    sleep 5 9>&-
  fi
  sleep 60 9>&-  # children must not inherit the lock, or a killed guard's sleep blocks its successor
done
