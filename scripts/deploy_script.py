import subprocess
def deploy():
    subprocess.run(["docker-compose", "up", "--build", "-d"])
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "Final deploy"])
    subprocess.run(["git", "push", "origin", "main"])
if __name__ == "__main__":
    deploy()
