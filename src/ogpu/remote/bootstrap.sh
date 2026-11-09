#!/bin/bash
# Machine start script (injected by `ogpu up` via OGPU_BOOTSTRAP). Gives the image a real sshd on the public TCP port
# (key auth only, host keys kept on the persistent volume so they survive stop/resume), snapshots the container env
# for SSH sessions, and (re)starts the idle reaper on every boot/resume.
#   OGPU_SSHD=0       the provider already runs sshd (Vast): only add our key
#   OGPU_KEEPALIVE=0  run as a hook (Vast onstart), not as the container's main process
set -u
OG=/workspace/.ogpu
umask 077
mkdir -p "$OG/jobs" "$OG/bin" /root/.ssh
chmod 700 "$OG" /root/.ssh

if [ "${OGPU_SSHD:-1}" = 1 ] && { ! command -v sshd >/dev/null || ! command -v rsync >/dev/null || ! command -v flock >/dev/null; }; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server rsync util-linux >/dev/null
elif ! command -v rsync >/dev/null || ! command -v flock >/dev/null; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync util-linux >/dev/null
fi
for k in "${OGPU_PUBKEY:-}" "${PUBLIC_KEY:-}"; do
  [ -n "$k" ] && ! grep -qF "$k" /root/.ssh/authorized_keys 2>/dev/null && echo "$k" >> /root/.ssh/authorized_keys
done
chmod 600 /root/.ssh/authorized_keys

if [ "${OGPU_SSHD:-1}" = 1 ]; then
  mkdir -p /run/sshd "$OG/ssh_host"
  if ls "$OG"/ssh_host/ssh_host_*_key >/dev/null 2>&1; then  # same host key after a resume: ogpu pins it
    cp "$OG"/ssh_host/ssh_host_* /etc/ssh/
  else
    ssh-keygen -A >/dev/null 2>&1
    cp /etc/ssh/ssh_host_* "$OG/ssh_host/"
  fi
  chmod 600 /etc/ssh/ssh_host_*_key
  sed -i -e 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' \
         -e 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' \
         -e 's/^#\?KbdInteractiveAuthentication.*/KbdInteractiveAuthentication no/' /etc/ssh/sshd_config
  /usr/sbin/sshd
fi

# SSH sessions do not inherit the container env (CUDA paths, the provider's machine id and scoped key, ...).
export -p | grep -vE ' (OGPU_BOOTSTRAP|HOSTNAME|PWD|OLDPWD|SHLVL|_)=' > "$OG/env.sh"
date +%s > "$OG/last_activity"

if [ -f "$OG/bin/reaperguard.sh" ]; then  # the guard starts the reaper and restarts it if it ever dies
  rm -f "$OG/reaper.pid"
  setsid nohup bash "$OG/bin/reaperguard.sh" >/dev/null 2>&1 < /dev/null &
fi
[ "${OGPU_KEEPALIVE:-1}" = 1 ] && exec sleep infinity
exit 0
