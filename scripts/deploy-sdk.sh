#!/bin/bash
set -e

echo "Starting WebXOS 2025 Vial MCP SDK deployment..."

# Build Docker images
docker build -f build/dockerfiles/mcp-complete.Dockerfile -t webxos-mcp:latest .

# Create secrets for Helm
kubectl create secret generic webxos-secrets \
  --from-literal=openai-api-key=$OPENAI_API_KEY \
  --from-literal=nasa-api-key=$NASA_API_KEY \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy with Helm
helm upgrade --install webxos ./helm/webxos -f helm/webxos/values.yaml

# Validate deployment
echo "Waiting for services to start..."
sleep 30
curl http://localhost:8000/mcp/auth/login -o /dev/null -s -w "%{http_code}\n" | grep 200 || { echo "Backend failed to start"; exit 1; }
curl http://localhost:3000/vial.html -o /dev/null -s -w "%{http_code}\n" | grep 200 || { echo "Frontend failed to start"; exit 1; }
sqlite3 webxos.db "SELECT * FROM wallets" || { echo "Database validation failed"; exit 1; }
curl http://localhost:9090/api/v1/query?query=webxos_requests_total -o /dev/null -s -w "%{http_code}\n" | grep 200 || { echo "Prometheus validation failed"; exit 1; }

echo "WebXOS 2025 Vial MCP SDK deployed successfully!"
