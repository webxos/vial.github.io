from fastapi import FastAPI

def setup_sdk(app: FastAPI):
    print("SDK setup")


def run_sdk(app: FastAPI):
    print("SDK running")
