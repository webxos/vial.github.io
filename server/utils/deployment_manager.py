from pydantic_settings import BaseSettings
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import os
from kubernetes import client, config
from server.config.settings import settings
from server.utils.scaling_config import scaling_config

logger = logging.getLogger(__name__)

class DeploymentManager:
    def __init__(self):
        self.kube_config = scaling_config.KUBE_CONFIG_PATH or os.getenv("KUBE_CONFIG_PATH")
        if self.kube_config:
            config.load_kube_config(self.kube_config)
        else:
            config.load_incluster_config()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def deploy_service(self, service_name: str, replicas: int) -> dict:
        """Deploy service to Kubernetes cluster."""
        try:
            v1 = client.AppsV1Api()
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(name=service_name),
                spec=client.V1DeploymentSpec(
                    replicas=replicas,
                    selector=client.V1LabelSelector(match_labels={"app": service_name}),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(labels={"app": service_name}),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name=service_name,
                                    image=f"vial-mcp-{service_name}:latest",
                                    env=[client.V1EnvVar(name=k, value=str(v)) for k, v in settings.dict().items()]
                                )
                            ]
                        )
                    )
                )
            )
            v1.create_namespaced_deployment(namespace="default", body=deployment)
            logger.info(f"Deployed service: {service_name} with {replicas} replicas")
            return {"status": "deployed", "service": service_name}
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            raise

deployment_manager = DeploymentManager()
