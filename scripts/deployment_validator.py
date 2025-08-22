import subprocess


def validate_deployment():
    result = subprocess.run(["docker-compose", "ps"],
                           capture_output=True, text=True)
    if "running" not in result.stdout:
        return {"status": "failed", "message": "Containers not running"}
    return {"status": "success", "message": "Deployment validated"}
