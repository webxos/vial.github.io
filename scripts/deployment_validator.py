import subprocess


def validate_deployment():
    result = subprocess.run(["docker-compose", "ps"], capture_output=True,
                           text=True)
    return "running" in result.stdout
