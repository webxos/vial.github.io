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

# Deploy to Kubernetes
helm upgrade --install vial-mcp k8s/helm-chart.yaml \
  --set image.tag=latest \
  --kubeconfig="$KUBE_CONFIG"

# Validate deployment
kubectl rollout status deployment/vial-mcp --timeout=5m

# Placeholder: OBS/SVG video health check
# curl -f http://localhost:8000/obs/health || exit 1

# Placeholder: Tauri desktop app deployment
# tauri build --target x86_64-pc-linux-gnu

echo "Deployment successful"
