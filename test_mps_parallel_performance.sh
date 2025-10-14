#!/bin/bash
# ==============================================================================
# Multi-Container Parallel Performance Evaluation with NVIDIA MPS (v1.1)
#
# Tech Lead: GPU Slicing 專案總監
# Engineer: You
#
# v1.1 Changelog:
# - Fixed: Container name conflict when SM_LIMITS contains duplicate values.
#   The container name now includes a unique index.
# ==============================================================================

set -euo pipefail

# --- Test Configuration ---
IMAGE_NAME="cuda_vmem:tf1.8-cu90"
GPU_DEVICE_ID=0
BENCHMARK_PROGRAM="/test_build/test/test_performance_benchmark"

# Define the compute power limits for each parallel container.
# This version now correctly handles duplicate values.
# SM_LIMITS=(100 100 100)
SM_LIMITS=(33 33 33)

# --- MPS Specific Paths ---
MPS_PIPE_DIR="/tmp/nvidia-mps-pipe-parallel-${GPU_DEVICE_ID}"
MPS_LOG_DIR="/tmp/nvidia-mps-log-parallel-${GPU_DEVICE_ID}"

# --- Global Variables ---
declare -a child_pids=()
declare -a container_names=()

# --- MPS Management Functions (Unchanged) ---
ensure_mps_is_dead() {
    echo " -> Executing thorough cleanup of the MPS environment..."
    local pids_ctl pids_srv
    pids_ctl=$(pgrep -f "nvidia-cuda-mps-control" || true)
    if [ -n "$pids_ctl" ]; then
        echo " -> Control daemon detected (PIDs: $pids_ctl), attempting to send 'quit'..."
        sudo -E env CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID} \
                   CUDA_MPS_PIPE_DIRECTORY=${MPS_PIPE_DIR} \
                   bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
    fi
    local timeout=5; echo -n " -> Waiting for MPS processes to terminate (max ${timeout}s)..."
    while [ "$timeout" -gt 0 ]; do
        pids_ctl=$(pgrep -f "nvidia-cuda-mps-control" || true); pids_srv=$(pgrep -f "nvidia-cuda-mps-server" || true)
        if [ -z "$pids_ctl" ] && [ -z "$pids_srv" ]; then break; fi
        sleep 1; echo -n "."; timeout=$((timeout - 1))
    done; echo ""
    pids_ctl=$(pgrep -f "nvidia-cuda-mps-control" || true); pids_srv=$(pgrep -f "nvidia-cuda-mps-server" || true)
    if [ -n "$pids_ctl" ] || [ -n "$pids_srv" ]; then
        echo " -> WARNING: Graceful shutdown failed. Forcibly killing remaining processes."
        [ -n "$pids_ctl" ] && sudo kill -9 $pids_ctl || true; [ -n "$pids_srv" ] && sudo kill -9 $pids_srv || true
        echo " -> Resetting GPU ${GPU_DEVICE_ID} compute mode to Default..."; sudo nvidia-smi -i ${GPU_DEVICE_ID} -c DEFAULT || echo " -> WARNING: Failed to reset compute mode."
    else
        echo " -> CONFIRMED: All MPS processes have terminated successfully."
    fi
    sudo rm -rf "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"; echo " -> MPS environment cleaned up."
}

start_mps_daemon() {
    ensure_mps_is_dead
    echo " -> Starting MPS control daemon..."
    sudo mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"; sudo chmod 1777 "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
    if ! sudo -E env CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID} \
                     CUDA_MPS_PIPE_DIRECTORY=${MPS_PIPE_DIR} \
                     CUDA_MPS_LOG_DIRECTORY=${MPS_LOG_DIR} \
                     nvidia-cuda-mps-control -d; then
        echo " -> [ERROR] Failed to execute the MPS control daemon start command." >&2; exit 1
    fi
    local control_pipe="${MPS_PIPE_DIR}/control"; local wait_count=0
    echo -n " -> Waiting for MPS control pipe to be ready..."; while [ ! -e "${control_pipe}" ] && [ "${wait_count}" -lt 10 ]; do sleep 0.5; echo -n "."; ((wait_count++)); done; echo ""
    if [ ! -e "${control_pipe}" ]; then echo " -> [ERROR] Timeout waiting for MPS control pipe: ${control_pipe}." >&2; exit 1; fi
    sudo nvidia-smi -i ${GPU_DEVICE_ID} -c EXCLUSIVE_PROCESS
    echo " -> MPS daemon is fully ready. GPU compute mode set to EXCLUSIVE_PROCESS."
}

# --- Main Cleanup Trap (Unchanged) ---
cleanup() {
    echo ""
    echo "--------------------------------------------------"
    echo "Interrupt signal caught, stopping containers and shutting down MPS..."
    if [ ${#container_names[@]} -ne 0 ]; then
        docker stop "${container_names[@]}" > /dev/null
        echo "All containers have been stopped."
    fi
    ensure_mps_is_dead
    echo "Cleanup complete."
    exit 0
}
trap cleanup EXIT SIGINT SIGTERM

# --- Prerequisite Check (Unchanged) ---
if [ ! -f "build/test/test_performance_benchmark" ]; then
    echo "Error: 'build/test/test_performance_benchmark' not found." >&2
    exit 1
fi

# --- Main Program ---
echo "Starting multi-container parallel performance evaluation using NVIDIA MPS..."
echo "=============================================================="

start_mps_daemon

echo "Launching ${#SM_LIMITS[@]} containers with compute power limits: ${SM_LIMITS[*]}%"
echo "Press Ctrl+C to abort the test at any time."
echo "--------------------------------------------------"

# --- ✨✨✨【核心修正】Launch Containers in Parallel ✨✨✨---
# Use a C-style loop to get an index for unique naming.
for i in "${!SM_LIMITS[@]}"; do
    limit="${SM_LIMITS[$i]}"
    # Append the index 'i' to the name to ensure uniqueness.
    name="mps-perf-test-${limit}p-${i}"
    container_names+=("${name}")

    echo "Launching container in the background: ${name} (Compute Limit: ${limit}%)..."

    docker run -d --rm \
        --name "${name}" \
        --gpus device=${GPU_DEVICE_ID} \
        --ipc=host \
        --shm-size=1g \
        -v "${MPS_PIPE_DIR}:${MPS_PIPE_DIR}" \
        -v "${MPS_LOG_DIR}:${MPS_LOG_DIR}" \
        -v "$(pwd)/build":/test_build \
        -e CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID} \
        -e CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
        -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${limit}" \
        "${IMAGE_NAME}" \
        "${BENCHMARK_PROGRAM}" > /dev/null
done

echo "--------------------------------------------------"
echo "All containers launched. Tailing logs..."
echo "Run 'watch -n 1 nvidia-smi' in another terminal to monitor GPU."
echo ""

# --- Display Logs in Parallel (Unchanged) ---
for name in "${container_names[@]}"; do
    { docker logs -f "${name}"; } | sed "s/^/[${name}] /" &
    child_pids+=($!)
done

wait "${child_pids[@]}"

echo "=============================================================="
echo "All benchmark containers have finished."