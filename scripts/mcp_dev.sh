#!/bin/bash
set -e

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Install Node.js dependencies
echo "registry=https://registry.npmjs.org/" > .npmrc
npm install
npm ci --omit=dev

# Install Python dependencies
pip install -r requirements.txt

# Setup MongoDB and Redis
docker-compose up -d mongodb redis

# Run development server
uvicorn server.api.main:app --host 0.0.0.0 --port 8000 &

# Placeholder: Setup Tauri development environment
# cargo install tauri-cli
# tauri dev

# Placeholder: Setup OBS WebSocket for SVG video
# npm install obs-websocket-js
# node scripts/setup_obs.js

echo "Development environment setup complete"
