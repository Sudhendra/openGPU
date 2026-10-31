#!/bin/bash
# Pod start command (injected by `ogpu up` via OGPU_BOOTSTRAP). Gives the judge image a real sshd on the public
# TCP port, snapshots the container env for SSH sessions, and restarts the idle reaper on every boot/resume.
set -u
OG=/workspace/.ogpu
mkdir -p "$OG/jobs" "$OG/bin" /run/sshd /root/.ssh
chmod 700 /root/.ssh

if ! command -v sshd >/dev/null || ! command -v rsync >/dev/null || ! command -v flock >/dev/null; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server rsync util-linux >/dev/null
fi
for k in "${OGPU_PUBKEY:-}" "${PUBLIC_KEY:-}"; do
  [ -n "$k" ] && ! grep -qF "$k" /root/.ssh/authorized_keys 2>/dev/null && echo "$k" >> /root/.ssh/authorized_keys
done
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A >/dev/null 2>&1
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
/usr/sbin/sshd

# SSH sessions do not inherit the container env (CUDA paths, RUNPOD_POD_ID, RUNPOD_API_KEY, ...).
export -p | grep -vE ' (OGPU_BOOTSTRAP|HOSTNAME|PWD|OLDPWD|SHLVL|_)=' > "$OG/env.sh"
chmod 600 "$OG/env.sh"
date +%s > "$OG/last_activity"

if [ -f "$OG/bin/reaperguard.sh" ]; then  # the guard starts the reaper and restarts it if it ever dies
  rm -f "$OG/reaper.pid"
  setsid nohup bash "$OG/bin/reaperguard.sh" >/dev/null 2>&1 < /dev/null &
elif [ -f "$OG/bin/reaper.py" ]; then
  nohup python3 "$OG/bin/reaper.py" >> "$OG/reaper.log" 2>&1 &
fi
exec sleep infinity
