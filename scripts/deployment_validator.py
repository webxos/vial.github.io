import subprocess
import json


def validate_deployment():
    result = subprocess.run(["docker-compose", "ps", "--format", "json"],
                           capture_output=True, text=True)
    try:
        services = json.loads(result.stdout)
        for service in services:
            if service["State"] != "running":
                return {"status": "failed", "message": f"Service {service['Name']} not running"}
        return {"status": "success", "message": "All services running"}
    except json.JSONDecodeError:
        return {"status": "failed", "message": "Failed to parse docker-compose status"}
