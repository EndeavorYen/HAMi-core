#!/bin/bash

# --- Test Configuration ---
IMAGE_NAME="cuda_vmem:tf1.8-cu90"
GPU_DEVICE_ID=0
# Ensure the correct, newly compiled benchmark program is used
BENCHMARK_PROGRAM="/test_build/test/test_performance_benchmark"

# --- Main Program ---
echo "Starting GPU performance step test (from 100% to 10%)..."
echo "=============================================================="

# Check if the benchmark program exists
if [ ! -f "build/test/test_performance_benchmark" ]; then
    echo "Error: Benchmark program 'build/test/test_performance_benchmark' not found."
    echo "Please ensure you have created the .cu file and re-run 'make build-in-docker'."
    exit 1
fi

# Loop from 100 decreasing to 10, decrementing by 10 each time
for limit in $(seq 100 -10 10); do
    echo ""
    echo "--- Testing performance with ${limit}% compute limit ---"
    
    # Start a new, clean container to run a single benchmark
    # The container will automatically remove itself after execution (--rm)
    docker run --rm \
        --gpus device=${GPU_DEVICE_ID} \
        --mount type=tmpfs,destination=/tmp/vgpulock \
        -v "$(pwd)/build":/test_build \
        -e CUDA_DEVICE_SM_LIMIT="${limit}" \
        -e LD_PRELOAD=/test_build/libvgpu.so \
        "${IMAGE_NAME}" \
        "${BENCHMARK_PROGRAM}"

    # Check if the Docker command executed successfully
    if [ $? -ne 0 ]; then
        echo "Error: Docker execution failed at ${limit}% compute limit, aborting test."
        exit 1
    fi
    
    echo "--- ${limit}% compute limit test completed ---"
    sleep 2 # Short pause between tests
done

echo ""
echo "=============================================================="
echo "All performance tests completed."