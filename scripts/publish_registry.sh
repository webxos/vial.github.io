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

# Publish to MCP registry
curl -X POST https://registry.mcp.io/servers \
  -H "Authorization: Bearer $REGISTRY_TOKEN" \
  -d '{"name": "Vial MCP", "url": "https://vial.github.io", "tools_endpoint": "https://vial.github.io/mcp/tools"}'

# Health check
curl -f http://localhost:8000/health || exit 1

echo "Published to MCP registry successfully"
