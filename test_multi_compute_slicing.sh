#!/bin/bash

# --- Test Configuration ---
IMAGE_NAME="cuda_vmem:tf1.8-cu90"
SM_LIMITS=(20 30 40)
GPU_DEVICE_ID=0
TEST_PROGRAM_PATH="/test_build/test/test_performance_benchmark" 

# --- Global variables to store background PIDs ---
declare -a child_pids=()
declare -a container_names=()

# --- Cleanup function (revised version) ---
cleanup() {
    echo ""
    echo "--------------------------------------------------"
    echo "Interrupt signal caught, stopping all background processes and containers..."

    # 1. Terminate all background 'docker logs' processes first
    # This is key to releasing the 'wait' in the main script
    if [ ${#child_pids[@]} -ne 0 ]; then
        echo "Terminating log tracking processes..."
        kill "${child_pids[@]}" 2>/dev/null
    fi

    # 2. Stop all test containers
    if [ ${#container_names[@]} -ne 0 ]; then
        echo "Stopping Docker containers..."
        docker stop "${container_names[@]}" > /dev/null
        echo "All containers stopped."
    fi
    
    echo "Cleanup complete."
    # Ensure the script exits cleanly after the trap is executed
    exit 0
}

# --- Main Program Start ---
echo "Starting multi-container Compute Slicing concurrent stress test..."
echo "Press Ctrl+C to stop and clean up all containers at any time."
echo "--------------------------------------------------"

# Set up trap to catch Ctrl+C (SIGINT) and termination signals (SIGTERM)
trap cleanup SIGINT SIGTERM

# Check if the test program exists
if [ ! -f "build/test/test_runtime_launch" ]; then
    echo "Error: 'build/test/test_runtime_launch' executable not found."
    echo "Please run 'make build-in-docker' to compile it first."
    exit 1
fi

# --- Start Containers ---
for limit in "${SM_LIMITS[@]}"; do
    name="compute-test-${limit}p"
    container_names+=("${name}")
    
    echo "Starting container in the background: ${name} (SM Limit: ${limit}%)..."
    
    docker run -d --rm \
        --name "${name}" \
        --gpus device=${GPU_DEVICE_ID} \
        --mount type=tmpfs,destination=/tmp/vgpulock \
        -v "$(pwd)/build":/test_build \
        -e CUDA_DEVICE_SM_LIMIT="${limit}" \
        -e LD_PRELOAD=/test_build/libvgpu.so \
        "${IMAGE_NAME}" \
        "${TEST_PROGRAM_PATH}" > /dev/null
done

echo "--------------------------------------------------"
echo "All containers started, now tracking logs in real-time..."
echo "Meanwhile, run 'watch -n 1 nvidia-smi' in another terminal window to monitor overall GPU utilization."
echo ""

# --- Parallel Log Display (revised version) ---
for name in "${container_names[@]}"; do
    { docker logs -f "${name}"; echo "[${name}] Process Exited."; } | sed "s/^/[${name}] /" &
    # Store the PID of the just-started background process into the array
    child_pids+=($!)
done

# 'wait' will pause the script here until the trap is triggered and exit is executed
wait