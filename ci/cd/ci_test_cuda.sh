#!/bin/bash

# Exit on any error
set -e

# Environment setup
export MCP_TEST_DIR="./build"
export MCP_OUTPUT_DIR="./test_results"
export MCP_TEST_CONFIG="test_args.json"
export MCP_PARALLEL="4"

# Build phase
mkdir -p $MCP_TEST_DIR
cd $MCP_TEST_DIR
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75
make -j$MCP_PARALLEL

# Test phase
cd -
python3 run_tests.py \
    --dir $MCP_TEST_DIR \
    --output $MCP_OUTPUT_DIR \
    --config $MCP_TEST_CONFIG \
    --parallel $MCP_PARALLEL

echo "CUDA tests completed. Check $MCP_OUTPUT_DIR for logs."
