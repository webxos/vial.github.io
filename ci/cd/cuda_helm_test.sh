#!/bin/bash
set -e

# Environment setup
export MCP_TEST_DIR="./build"
export MCP_OUTPUT_DIR="./test_results"
export MCP_TEST_CONFIG="test_args.json"
export MCP_PARALLEL="4"
export HELM_CHART_DIR="./deploy/helm/mcp-stack"

# Build CUDA samples
mkdir -p $MCP_TEST_DIR
cd $MCP_TEST_DIR
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75
make -j$MCP_PARALLEL

# Run CUDA tests
cd -
python3 run_tests.py \
    --dir $MCP_TEST_DIR \
    --output $MCP_OUTPUT_DIR \
    --config $MCP_TEST_CONFIG \
    --parallel $MCP_PARALLEL

# Package and deploy Helm chart
helm lint $HELM_CHART_DIR
helm dependency update $HELM_CHART_DIR
helm install mcp-test $HELM_CHART_DIR \
    --namespace mcp-test \
    --set mcpServer.image.repository=your-registry/mcp-server \
    --set mcpServer.image.tag=latest \
    --wait

echo "CUDA and Helm tests completed. Check $MCP_OUTPUT_DIR and Kubernetes logs."
