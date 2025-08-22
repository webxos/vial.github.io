# setup.py
from setuptools import setup, find_packages

setup(
    name="vial-mcp-controller",
    version="1.0.0",
    description="Vial MCP Controller with WebXOS wallet and reputation system",
    author="xAI",
    author_email="support@x.ai",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "fastapi==0.103.2",
        "uvicorn==0.23.2",
        "sqlalchemy==2.0.21",
        "pydantic-settings==2.0.3",
        "torch==2.0.1",
        "redis==4.6.0",
        "pytest==7.4.2",
        "pytest-asyncio==0.21.1",
        "alembic==1.12.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
