#!/bin/bash
set -e

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Generate package-lock.json if missing
if [ ! -f package-lock.json ]; then
  npm install
fi

# Build and deploy
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml up -d

# Setup SSL with certbot
certbot --nginx -d vial.github.io --non-interactive --agree-tos -m admin@vial.github.io

# Configure backups
echo "0 0 * * * docker exec mongo mongodump --archive=/backup/vial_mcp_$(date +%F).archive" | crontab -

# Health check
curl -f http://localhost:8000/health || exit 1
