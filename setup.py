from setuptools import setup, find_packages

setup(
    name="vial-mcp-alchemist",
    version="0.1.0",
    author="WebXOS",
    author_email="support@webxos.com",
    description="MCP Alchemist for Vial MCP Controller",
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vial/vial.github.io",
    packages=["server.models"],
    package_dir={"": "server"},
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "pymongo>=4.0.0",
        "python-dotenv>=0.21.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
