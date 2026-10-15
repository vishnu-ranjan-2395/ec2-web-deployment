
#!/bin/bash
set -euo pipefail

for attempt in $(seq 1 20); do
    if curl --fail --silent http://127.0.0.1:8000/health \
        | grep -q '"model_loaded":true'; then
        echo "ML application is healthy."
        exit 0
    fi
    sleep 3
done

echo "ML application failed health validation."
systemctl status ml-web-app --no-pager || true
journalctl -u ml-web-app -n 60 --no-pager || true
exit 1
