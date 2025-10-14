#!/bin/bash

# --- Test Configuration ---
IMAGE_NAME="cuda_vmem:tf1.8-cu90"
VRAM_LIMITS=("1g" "4g" "6g")
GPU_DEVICE_ID=0

# --- Cleanup Function ---
# This function is called when the script is interrupted (Ctrl+C) or exits normally
cleanup() {
    echo ""
    echo "--------------------------------------------------"
    echo "Stopping and cleaning up all test containers..."
    # If the container_names array is not empty
    if [ ${#container_names[@]} -ne 0 ]; then
        docker stop "${container_names[@]}" > /dev/null
        echo "All containers have been stopped."
    fi
    # Kill all background docker logs processes
    kill 0
}

# --- Main Program Start ---
echo "Starting multi-container VRAM limit parallel stress test..."
echo "Press Ctrl+C to stop and clean up all containers at any time."
echo "--------------------------------------------------"

# Set trap to intercept Ctrl+C (SIGINT) and termination signals (SIGTERM)
trap cleanup SIGINT SIGTERM

# Check if the required build directory exists
if [ ! -d "build" ] || [ ! -f "build/test/test_vram_limit" ]; then
    echo "Error: Cannot find build/test/test_vram_limit executable."
    echo "Please run 'make build-in-docker' to compile it first."
    exit 1
fi

# Array to store container names
container_names=()

# --- Start Containers ---
for limit in "${VRAM_LIMITS[@]}"; do
    name="vram-test-${limit}"
    container_names+=("${name}")
    
    echo "Starting container in the background: ${name} (VRAM limit: ${limit})..."
    
    # Start the container in the background (-d)
    docker run -d --rm \
        --name "${name}" \
        --gpus device=${GPU_DEVICE_ID} \
        --mount type=tmpfs,destination=/tmp/vgpulock \
        -v "$(pwd)/build":/test_build \
        -e CUDA_DEVICE_MEMORY_LIMIT="${limit}" \
        -e LD_PRELOAD=/test_build/libvgpu.so \
        "${IMAGE_NAME}" \
        bash -c "sleep 2 && /test_build/test/test_vram_limit" > /dev/null
done

echo "--------------------------------------------------"
echo "All containers have been started, now tracking logs in real-time..."
echo ""

# --- Display Logs in Parallel ---
# Start a 'docker logs' process in the background for each container
# Prefix each process's output with the container name using sed
for name in "${container_names[@]}"; do
    docker logs -f "${name}" | sed "s/^/[${name}] /" &
done

# --- Wait and Cleanup ---
# 'wait' pauses the script here until all background processes (docker logs) finish
# When you press Ctrl+C, the trap triggers cleanup, kills all background processes, and wait ends
wait