#!/bin/bash
# ==============================================================================
# API 呼叫開銷量化測試腳本 v2 (排除順序效應)
#
# Tech Lead: GPU Slicing 專案總監
# Engineer: You
#
# v2 更新:
# - 增加獨立的系統預熱階段，以穩定 CPU 時脈與系統快取。
# - 將三種測試情境函式化，並以隨機順序執行，徹底排除順序效應。
# ==============================================================================

set -euo pipefail

# --- 測試設定 ---
readonly IMAGE_NAME="cuda_vmem:tf1.8-cu90"
readonly GPU_DEVICE_ID=0
readonly TEST_PROGRAM_NAME="test_api_latency"
readonly PROJECT_ROOT="/libvgpu"
readonly TEST_PROGRAM_CONTAINER_PATH="${PROJECT_ROOT}/build/test/${TEST_PROGRAM_NAME}"
readonly VGPU_LIBRARY_PATH="${PROJECT_ROOT}/build/libvgpu.so"

# --- MPS 管理設定 ---
readonly MPS_PIPE_DIR="/tmp/nvidia-mps-pipe-${GPU_DEVICE_ID}"
readonly MPS_LOG_DIR="/tmp/nvidia-mps-log-${GPU_DEVICE_ID}"


# --- MPS 管理函式 (維持已驗證版本) ---
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

# --- ✨✨✨【核心修正 1: 測試函式化】✨✨✨ ---
run_baseline_test() {
    echo "🚀 測試 Baseline (原生) 延遲..."
    docker run --rm \
        --gpus "device=${GPU_DEVICE_ID}" \
        -v "$(pwd)":"${PROJECT_ROOT}" -w "${PROJECT_ROOT}" \
        "${IMAGE_NAME}" \
        "${TEST_PROGRAM_CONTAINER_PATH}"
    echo "✅ Baseline 測試完成。"
}

run_hamicore_test() {
    echo "🚀 測試 HAMi-core (In-Process Hook) 延遲..."
    docker run --rm \
        --gpus "device=${GPU_DEVICE_ID}" \
        --mount type=tmpfs,destination=/tmp/vgpulock \
        -v "$(pwd)":"${PROJECT_ROOT}" -w "${PROJECT_ROOT}" \
        -e LD_PRELOAD="${VGPU_LIBRARY_PATH}" \
        "${IMAGE_NAME}" \
        "${TEST_PROGRAM_CONTAINER_PATH}"
    echo "✅ HAMi-core 測試完成。"
}

run_mps_test() {
    echo "🚀 測試 MPS (Client-Server) 延遲..."
    start_mps_daemon

    docker run --rm \
        --gpus "device=${GPU_DEVICE_ID}" \
        --ipc=host \
        -v "${MPS_PIPE_DIR}:${MPS_PIPE_DIR}" \
        -v "${MPS_LOG_DIR}:${MPS_LOG_DIR}" \
        -v "$(pwd)":"${PROJECT_ROOT}" -w "${PROJECT_ROOT}" \
        -e CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID} \
        -e CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
        "${IMAGE_NAME}" \
        "${TEST_PROGRAM_CONTAINER_PATH}"
    echo "✅ MPS 測試完成。"
}

# --- 全局清理陷阱 ---
cleanup() {
    echo ""; echo "=============================================================="
    echo "執行最終清理程序..."; ensure_mps_is_dead; echo "最終清理完畢。"
}
trap cleanup EXIT SIGINT SIGTERM

# --- 主測試程式 ---
if [ ! -f "./build/test/${TEST_PROGRAM_NAME}" ]; then
    echo "[錯誤] 找不到測試程式: ./build/test/${TEST_PROGRAM_NAME}" >&2; exit 1
fi

echo "開始 API 呼叫開銷量化測試 (V2: 隨機順序)..."
echo "=============================================================="
ensure_mps_is_dead # 確保初始環境乾淨

# --- ✨✨✨【核心修正 2: 系統預熱】✨✨✨ ---
echo "🔥 執行系統預熱 (System Warm-up)..."
docker run --rm --gpus "device=${GPU_DEVICE_ID}" "${IMAGE_NAME}" nvidia-smi > /dev/null
echo "✅ 系統預熱完成。"
echo "--------------------------------------------------------------"
sleep 2

# --- ✨✨✨【核心修正 3: 隨機化執行順序】✨✨✨ ---
# 定義要執行的測試函式陣列
declare -a test_functions=("run_baseline_test" "run_hamicore_test" "run_mps_test")

# 使用 shuf (shuffle) 工具打亂陣列順序
# mapfile 或 read -a 是將 shuf 的輸出讀回陣列的標準方法
mapfile -t shuffled_functions < <(printf "%s\n" "${test_functions[@]}" | shuf)

echo "📊 將以下列隨機順序執行測試: ${shuffled_functions[*]}"
echo "--------------------------------------------------------------"

# 依隨機順序執行測試
for func in "${shuffled_functions[@]}"; do
    # 執行函式
    "$func"
    echo "--------------------------------------------------------------"
    sleep 2 # 每次測試之間休息
done


echo "=============================================================="
echo "所有開銷測試已完成。"