#!/bin/sh
# Install the hourly posting timer for the bot in this checkout, run as you.
#   deploy/install.sh
set -e
dir=$(cd "$(dirname "$0")/.." && pwd)
user=$(id -un)

if [ ! -x "$dir/.venv/bin/python" ]; then
    echo "No virtualenv at $dir/.venv; run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi
[ -f "$dir/whobot.env" ] || echo "Warning: no $dir/whobot.env yet; posting will fail until it exists." >&2

sed -e "s|@DIR@|$dir|g" -e "s|@USER@|$user|g" "$dir/deploy/whobot.service" \
    | sudo tee /etc/systemd/system/whobot.service >/dev/null
sudo cp "$dir/deploy/whobot.timer" /etc/systemd/system/whobot.timer
sudo systemctl daemon-reload
sudo systemctl enable --now whobot.timer
echo "Installed for $user in $dir"
systemctl list-timers whobot.timer --no-pager
