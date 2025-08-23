import logging
import subprocess
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel
import os

logger = logging.getLogger(__name__)

class DeploymentConfig(BaseModel):
    docker_image: str = "vial/mcp-server:latest"
    helm_chart: str = "k8s/helm-chart.yaml"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def deploy():
    """Deploy MCP server to Kubernetes."""
    try:
        # Ensure package-lock.json
        if not os.path.exists("package-lock.json"):
            subprocess.run(["npm", "install"], check=True)
            subprocess.run(["git", "add", "package-lock.json"], check=True)
            subprocess.run(["git", "commit", "-m", "Add package-lock.json"], check=True)
            subprocess.run(["git", "push"], check=True)

        # Build and push Docker image
        subprocess.run(["docker", "build", "-t", DeploymentConfig().docker_image, "."], check=True)
        subprocess.run(["docker", "push", DeploymentConfig().docker_image], check=True)

        # Deploy Helm chart
        subprocess.run([
            "helm", "upgrade", "--install", "vial-mcp", DeploymentConfig().helm_chart,
            "--set", "image.tag=latest", "--kubeconfig", os.getenv("KUBE_CONFIG")
        ], check=True)

        # Validate deployment
        subprocess.run(["kubectl", "rollout", "status", "deployment/vial-mcp", "--timeout=5m"], check=True)

        logger.info("Deployment successful")
    except subprocess.CalledProcessError as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    deploy()
