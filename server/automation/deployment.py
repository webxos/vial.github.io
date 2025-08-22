from fastapi import FastAPI

def setup_deployment(app: FastAPI):
    print("Deploying application")


async def deploy_vps(app: FastAPI):
    print("Deploying to VPS with configuration")
