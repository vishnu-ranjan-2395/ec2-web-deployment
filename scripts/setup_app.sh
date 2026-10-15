
#!/bin/bash
set -euo pipefail

APP_DIR="/opt/ml-web-app"
APP_USER="ec2-user"

cd "$APP_DIR"

# Give the application user ownership of the app files.
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# Create an isolated Python environment.
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/venv"

# Install the Python packages.
sudo -u "$APP_USER" "$APP_DIR/venv/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"

# Train and save the model on the EC2 instance.
sudo -u "$APP_USER" "$APP_DIR/venv/bin/python" "$APP_DIR/train_model.py"

# Install the systemd service.
cat > /etc/systemd/system/ml-web-app.service <<'SERVICE'
[Unit]
Description=FastAPI Machine Learning Web App
After=network.target

[Service]
User=ec2-user
Group=ec2-user
WorkingDirectory=/opt/ml-web-app
Environment="PYTHONUNBUFFERED=1"
ExecStart=/opt/ml-web-app/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable ml-web-app
