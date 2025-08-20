#!/bin/bash

# Initialize the Vial MCP project
echo "Initializing Vial MCP Template..."

# Create data directory if it doesn't exist
mkdir -p server/data

# Install Python dependencies
pip install -r server/requirements.txt

# Install Node.js dependencies
cd web && npm install && cd ..

# Build Docker images
docker-compose build

echo "Initialization complete. Run 'docker-compose up' to start the project."
