#!/bin/bash
set -e

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Ensure package-lock.json
if [ ! -f package-lock.json ]; then
  npm install
  git add package-lock.json
  git commit -m "Add package-lock.json for npm ci"
  git push
fi

# Build and push Docker image
docker build -t vial/mcp-server:latest .
docker login -u "$DOCKER_USERNAME" -p "$DOCKER_PASSWORD"
docker push vial/mcp-server:latest

# Health check
curl -f http://localhost:8000/health || exit 1

# Placeholder: OBS/SVG video health check
# curl -f http://localhost:8000/obs/health || exit 1

echo "Docker build and push successful"
