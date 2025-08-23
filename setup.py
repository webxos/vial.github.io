from setuptools import setup, find_packages

setup(
    name="vial-mcp",
    version="1.0.0",
    description="Vial MCP Server for quantum, LLM, and industrial applications",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Vial MCP Contributors",
    url="https://vial.github.io",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.115.0",
        "uvicorn==0.30.6",
        "pydantic==2.9.2",
        "sqlalchemy==2.0.35",
        "torch==2.4.1",
        "qiskit==1.2.4",
        "httpx==0.27.2",
        "tenacity==9.0.0",
        "pymongo==4.8.0",
        "redis==5.0.8",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
