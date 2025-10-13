#!/bin/bash

# --- Test Configuration ---
IMAGE_NAME="cuda_vmem:tf1.8-cu90"
# Set different compute power limits for three containers, totaling 90%
SM_LIMITS=(20 30 40)
GPU_DEVICE_ID=0
# Execute our newly created performance evaluation program
BENCHMARK_PROGRAM="/test_build/test/test_performance_benchmark"

# --- Global Variables ---
declare -a child_pids=()
declare -a container_names=()

# --- Cleanup Function ---
cleanup() {
    echo ""
    echo "--------------------------------------------------"
    echo "Interrupt signal caught, stopping all background processes and containers..."
    if [ ${#child_pids[@]} -ne 0 ]; then
        kill "${child_pids[@]}" 2>/dev/null
    fi
    if [ ${#container_names[@]} -ne 0 ]; then
        docker stop "${container_names[@]}" > /dev/null
        echo "All containers have been stopped."
    fi
    echo "Cleanup complete."
    exit 0
}

# --- Main Program Start ---
echo "Starting multi-container parallel performance evaluation..."
echo "Launching 3 containers with compute power limits: ${SM_LIMITS[*]}%"
echo "Press Ctrl+C to abort the test at any time."
echo "--------------------------------------------------"

# Set trap
trap cleanup SIGINT SIGTERM

# Check if the benchmark program exists
if [ ! -f "build/test/test_performance_benchmark" ]; then
    echo "Error: 'build/test/test_performance_benchmark' not found."
    echo "Please run 'make build-in-docker' first."
    exit 1
fi

# --- Launch Containers ---
for limit in "${SM_LIMITS[@]}"; do
    name="perf-test-${limit}p"
    container_names+=("${name}")
    
    echo "Launching container in the background: ${name} (SM Limit: ${limit}%)..."
    
    docker run -d --rm \
        --name "${name}" \
        --gpus device=${GPU_DEVICE_ID} \
        --mount type=tmpfs,destination=/tmp/vgpulock \
        -v "$(pwd)/build":/test_build \
        -e CUDA_DEVICE_SM_LIMIT="${limit}" \
        -e LD_PRELOAD=/test_build/libvgpu.so \
        "${IMAGE_NAME}" \
        "${BENCHMARK_PROGRAM}" > /dev/null
done

echo "--------------------------------------------------"
echo "All containers have been launched, real-time logs are being tracked..."
echo "Meanwhile, please run 'watch -n 1 nvidia-smi' in another terminal window to monitor overall GPU utilization."
echo ""

# --- Display Logs in Parallel ---
for name in "${container_names[@]}"; do
    { docker logs -f "${name}"; } | sed "s/^/[${name}] /" &
    child_pids+=($!)
done

# Wait for all background 'docker logs' processes to finish
wait
# Once all log processes finish (indicating all containers have completed), perform cleanup
cleanup