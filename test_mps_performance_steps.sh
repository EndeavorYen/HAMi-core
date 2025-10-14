#!/bin/bash
# ==============================================================================
# MPS Compute Slicing Performance Step Test - v2 (專家修正版)
#
# Tech Lead: GPU Slicing 專案總監
# Engineer: You
#
# v2
# - 【重大修正】採納 MPS 專家建議，採用「關閉並重建」策略。
#   在每次迴圈結束後，手動終止當前的 MPS server，以強制 control daemon
#   在下次迴圈時根據新設定啟動一個全新的 server。
# - 【修正】修正 `set_default_active_thread_percentage` 指令，根據文件
#   移除不必要的 GPU ID 參數。
# ==============================================================================

set -euo pipefail

# --- Test Configuration ---
IMAGE_NAME="cuda_vmem:tf1.8-cu90"
GPU_DEVICE_ID=0
BENCHMARK_PROGRAM="/test_build/test/test_performance_benchmark"

# --- MPS Specific Paths ---
MPS_PIPE_DIR="/tmp/nvidia-mps-pipe-${GPU_DEVICE_ID}"
MPS_LOG_DIR="/tmp/nvidia-mps-log-${GPU_DEVICE_ID}"


# --- Function Definitions ---
# (ensure_mps_is_dead, start_mps_daemon 函式保持不變，它們已經很穩健)
ensure_mps_is_dead() {
    echo "  -> 執行 MPS 環境的徹底清理程序..."
    local pids_ctl pids_srv
    pids_ctl=$(pgrep -f "nvidia-cuda-mps-control" || true)
    if [ -n "$pids_ctl" ]; then
        echo "  -> 偵測到 control daemon (PIDs: $pids_ctl)，嘗試發送 'quit'..."
        sudo -E bash -c "export CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID}; export CUDA_MPS_PIPE_DIRECTORY='${MPS_PIPE_DIR}'; echo quit | nvidia-cuda-mps-control" >/dev/null 2>&1 || true
    fi
    local timeout=5; echo -n "  -> 等待 MPS 程序終止 (最多 ${timeout} 秒)..."
    while [ "$timeout" -gt 0 ]; do
        pids_ctl=$(pgrep -f "nvidia-cuda-mps-control" || true); pids_srv=$(pgrep -f "nvidia-cuda-mps-server" || true)
        if [ -z "$pids_ctl" ] && [ -z "$pids_srv" ]; then break; fi
        sleep 1; echo -n "."; timeout=$((timeout - 1))
    done; echo ""
    pids_ctl=$(pgrep -f "nvidia-cuda-mps-control" || true); pids_srv=$(pgrep -f "nvidia-cuda-mps-server" || true)
    if [ -n "$pids_ctl" ] || [ -n "$pids_srv" ]; then
        echo "  -> 警告：溫和關閉失敗。將強制終止殘留程序。"
        [ -n "$pids_ctl" ] && sudo kill -9 $pids_ctl || true; [ -n "$pids_srv" ] && sudo kill -9 $pids_srv || true
        echo "  -> 重設 GPU ${GPU_DEVICE_ID} 的計算模式為 Default..."; sudo nvidia-smi -i ${GPU_DEVICE_ID} -c DEFAULT || echo "  -> 警告：重設計算模式失敗。"
    else
        echo "  -> 確認：所有 MPS 程序已成功終止。"
    fi
    sudo rm -rf "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"; echo "  -> MPS 環境清理完畢。"
}

start_mps_daemon() {
    ensure_mps_is_dead
    echo "  -> 正在啟動 MPS daemon..."
    sudo mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"; sudo chmod 1777 "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
    if ! sudo -E bash -c "export CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID}; export CUDA_MPS_PIPE_DIRECTORY='${MPS_PIPE_DIR}'; export CUDA_MPS_LOG_DIRECTORY='${MPS_LOG_DIR}'; nvidia-cuda-mps-control -d"; then
        echo "  -> [錯誤] 啟動 MPS control daemon 命令執行失敗。"; exit 1
    fi
    local control_pipe="${MPS_PIPE_DIR}/control"; local wait_count=0
    echo -n "  -> 等待 MPS control pipe 就緒..."; while [ ! -e "${control_pipe}" ] && [ "${wait_count}" -lt 10 ]; do sleep 0.5; echo -n "."; ((wait_count++)); done; echo ""
    if [ ! -e "${control_pipe}" ]; then echo "  -> [錯誤] 等待逾時！找不到 MPS control pipe: ${control_pipe}。"; exit 1; fi
    echo "  -> MPS daemon 已完全就緒。"
}

# ✨✨✨【核心修正 1】✨✨✨
# 新增的輔助函式，只關閉 server，不影響 control daemon
kill_active_mps_server() {
    echo "  -> 正在關閉當前的 MPS server..."
    local server_pid
    server_pid=$(pgrep -f "nvidia-cuda-mps-server" || true)

    if [ -n "$server_pid" ]; then
        sudo kill "$server_pid"
        local count=0
        while pgrep -f "nvidia-cuda-mps-server" >/dev/null && [ "$count" -lt 5 ]; do
            sleep 0.5; ((count++))
        done
        # 最終確認，如果還在就強制 kill
        if pgrep -f "nvidia-cuda-mps-server" >/dev/null; then
            sudo kill -9 $(pgrep -f "nvidia-cuda-mps-server")
        fi
        echo "  -> MPS server 已關閉。"
    else
        echo "  -> 沒有偵測到正在運行的 MPS server。"
    fi
}

# --- Main Cleanup Trap ---
cleanup() {
    echo ""; echo "=============================================================="
    echo "捕獲到中斷信號，正在執行最終清理..."; ensure_mps_is_dead; echo "最終清理完畢。"
}
trap cleanup EXIT SIGINT SIGTERM

# --- Prerequisite Check ---
if [ ! -f "build/test/test_performance_benchmark" ]; then
    echo "錯誤：找不到 'build/test/test_performance_benchmark' 執行檔。"; exit 1
fi

# --- Main Program ---
echo "Starting MPS GPU performance step test (from 100% to 10%)..."; echo "=============================================================="

# 在整個測試迴圈開始前，只啟動一次 control daemon
start_mps_daemon

# Loop from 100 decreasing to 10, decrementing by 10 each time
for limit in $(seq 100 -10 10); do
    echo ""
    echo "--- Testing performance with ${limit}% compute limit ---"

    # ✨✨✨【核心修正 2】✨✨✨
    # 修正指令，不再需要 GPU ID
    echo "  -> 正在為 *下一個* server 設定算力上限: ${limit}%..."
    if ! sudo -E bash -c "export CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID}; export CUDA_MPS_PIPE_DIRECTORY='${MPS_PIPE_DIR}'; echo set_default_active_thread_percentage ${limit} | nvidia-cuda-mps-control"; then
        echo "  -> [錯誤] 設定 MPS 算力上限失敗。"; exit 1
    fi
    echo "  -> 算力上限設定成功。"

    # 執行 MPS client 容器，這次執行會觸發一個帶有新限制的 server 啟動
    docker run --rm \
        --gpus device=${GPU_DEVICE_ID} \
        --ipc=host \
        --shm-size=1g \
        -v "${MPS_PIPE_DIR}:${MPS_PIPE_DIR}" \
        -v "${MPS_LOG_DIR}:${MPS_LOG_DIR}" \
        -v "$(pwd)/build":/test_build \
        -e CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID} \
        -e CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
        "${IMAGE_NAME}" \
        "${BENCHMARK_PROGRAM}"

    if [ $? -ne 0 ]; then
        echo "錯誤：Docker 執行失敗於 ${limit}% 算力限制，測試中止。"; exit 1
    fi

    # ✨✨✨【核心修正 3】✨✨✨
    # 測試完成後，手動關閉剛剛啟動的 MPS server
    # 確保下一次迴圈能根據新設定生成一個全新的 server
    kill_active_mps_server

    echo "--- ${limit}% 算力限制測試完成 ---"
    sleep 2 # 在兩次測試之間稍作停頓
done

echo ""
echo "=============================================================="
echo "所有 MPS 效能階梯測試已完成。"