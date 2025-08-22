from setuptools import setup, find_packages

setup(
    name="vial_mcp",
    version="2.9.3",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.110.0",
        "uvicorn==0.29.0",
        "sqlalchemy==2.0.30",
        "torch==2.4.0",
        "qiskit==1.1.0",
        "pymongo==4.6.3",
        "requests-oauthlib==2.0.0",
        "gitpython==3.1.43",
    ],
    author="xAI",
    author_email="support@x.ai",
    description="Modular Control Plane for AI-driven task management",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vial/vial.github.io",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
