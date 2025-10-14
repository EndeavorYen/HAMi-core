#!/bin/bash
# ==============================================================================
# VRAM 限制功能比較測試 (HAMi-core vs. MPS) - v9 (IPC Namespace 修正版)
#
# Tech Lead: GPU Slicing 專案總監
# Engineer: You
#
# v9
# - 【核心修正】為 MPS 測試容器增加 `--ipc=host` 選項。這讓容器共享主機的
#   IPC Namespace，是解決 MPS client-server 通信失敗 (cuInit: 205)
#   的關鍵步驟。
# - 【診斷強化】在啟動容器前，增加對主機 MPS 目錄的權限檢查。
# - 【診斷強化】在容器失敗後，自動印出 MPS server 的日誌 (`server.log`)，
#   提供最直接的錯誤線索。
# ==============================================================================

set -euo pipefail

# --- 測試設定 ---
IMAGE_NAME="cuda_vmem:tf1.8-cu90"
GPU_DEVICE_ID=0
TEST_PROGRAM_PATH="/test_build/test/test_vram_limit"
VRAM_LIMIT="512m"

# --- MPS 特定路徑 ---
MPS_PIPE_DIR="/tmp/nvidia-mps-pipe-${GPU_DEVICE_ID}"
MPS_LOG_DIR="/tmp/nvidia-mps-log-${GPU_DEVICE_ID}"


# --- 函式定義 ---
# (ensure_mps_is_dead, start_mps_daemon, cleanup 函式保持不變)
ensure_mps_is_dead() {
    echo "  -> 執行 MPS 環境的徹底清理程序..."
    local pids_ctl pids_srv

    pids_ctl=$(pgrep -f "nvidia-cuda-mps-control" || true)
    if [ -n "$pids_ctl" ]; then
        echo "  -> 偵測到 control daemon (PIDs: $pids_ctl)，嘗試發送 'quit'..."
        sudo -E bash -c "export CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID}; export CUDA_MPS_PIPE_DIRECTORY='${MPS_PIPE_DIR}'; echo quit | nvidia-cuda-mps-control" >/dev/null 2>&1 || true
    fi

    local timeout=5
    echo -n "  -> 等待 MPS 程序終止 (最多 ${timeout} 秒)..."
    while [ "$timeout" -gt 0 ]; do
        pids_ctl=$(pgrep -f "nvidia-cuda-mps-control" || true)
        pids_srv=$(pgrep -f "nvidia-cuda-mps-server" || true)
        if [ -z "$pids_ctl" ] && [ -z "$pids_srv" ]; then break; fi
        sleep 1; echo -n "."; timeout=$((timeout - 1))
    done; echo ""

    pids_ctl=$(pgrep -f "nvidia-cuda-mps-control" || true)
    pids_srv=$(pgrep -f "nvidia-cuda-mps-server" || true)
    if [ -n "$pids_ctl" ] || [ -n "$pids_srv" ]; then
        echo "  -> 警告：溫和關閉失敗。將強制終止殘留程序。"
        [ -n "$pids_ctl" ] && sudo kill -9 $pids_ctl || true
        [ -n "$pids_srv" ] && sudo kill -9 $pids_srv || true
        echo "  -> 重設 GPU ${GPU_DEVICE_ID} 的計算模式為 Default..."
        sudo nvidia-smi -i ${GPU_DEVICE_ID} -c DEFAULT || echo "  -> 警告：重設計算模式失敗，但繼續執行。"
    else
        echo "  -> 確認：所有 MPS 程序已成功終止。"
    fi

    sudo rm -rf "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
    echo "  -> MPS 環境清理完畢。"
}

start_mps_daemon() {
    ensure_mps_is_dead
    echo "  -> 正在啟動 MPS daemon..."
    sudo mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
    sudo chmod 1777 "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"

    if ! sudo -E bash -c "export CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID}; export CUDA_MPS_PIPE_DIRECTORY='${MPS_PIPE_DIR}'; export CUDA_MPS_LOG_DIRECTORY='${MPS_LOG_DIR}'; nvidia-cuda-mps-control -d"; then
        echo "  -> [錯誤] 啟動 MPS control daemon 命令執行失敗。"; exit 1
    fi

    local control_pipe="${MPS_PIPE_DIR}/control"; local wait_count=0
    echo -n "  -> 等待 MPS control pipe 就緒..."
    while [ ! -e "${control_pipe}" ] && [ "${wait_count}" -lt 10 ]; do
        sleep 0.5; echo -n "."; ((wait_count++))
    done; echo ""

    if [ ! -e "${control_pipe}" ]; then
        echo "  -> [錯誤] 等待逾時！找不到 MPS control pipe: ${control_pipe}。"
        if [ -f "${MPS_LOG_DIR}/control.log" ]; then
            echo "  -> 偵測到 daemon 日誌，內容如下:"; echo "==================== MPS control.log ===================="
            sudo cat "${MPS_LOG_DIR}/control.log"; echo "========================================================="
        else
            echo "  -> 未找到 daemon 日誌檔案: ${MPS_LOG_DIR}/control.log"
        fi; exit 1
    fi
    echo "  -> MPS daemon 已完全就緒。"
}

cleanup() {
    echo ""; echo "--------------------------------------------------"; echo "正在執行最終清理..."
    docker ps -a --filter "name=hami-vram-test" --filter "name=mps-vram-test" -q | xargs -r docker stop >/dev/null 2>&1
    ensure_mps_is_dead
    echo "最終清理完畢。"
}
trap cleanup EXIT SIGINT SIGTERM

# --- 主流程 ---
echo "正在執行初始環境清理..."; ensure_mps_is_dead; echo "--------------------------------------------------"

echo "開始 VRAM 限制功能比較測試..."; echo "預設 VRAM 上限: ${VRAM_LIMIT}"; echo "--------------------------------------------------"

echo "🚀 (1/2) 執行 HAMi VRAM 限制測試..."
docker run --rm --name "hami-vram-test" --gpus device=${GPU_DEVICE_ID} --mount type=tmpfs,destination=/tmp/vgpulock -v "$(pwd)/build":/test_build -e CUDA_DEVICE_MEMORY_LIMIT="${VRAM_LIMIT}" -e LD_PRELOAD=/test_build/libvgpu.so "${IMAGE_NAME}" "${TEST_PROGRAM_PATH}"
echo "✅ HAMi 測試完成."; echo "--------------------------------------------------"; sleep 2

echo "🚀 (2/2) 執行 MPS VRAM 限制測試..."
start_mps_daemon

echo "  -> 正在為 MPS 伺服器設定 VRAM 上限: ${VRAM_LIMIT}..."
if ! sudo -E bash -c "export CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID}; export CUDA_MPS_PIPE_DIRECTORY='${MPS_PIPE_DIR}'; echo set_default_device_pinned_mem_limit ${GPU_DEVICE_ID} ${VRAM_LIMIT} | nvidia-cuda-mps-control"; then
    echo "  -> [錯誤] 設定 MPS VRAM 上限失敗。"; exit 1
fi
echo "  -> VRAM 上限設定成功。"

# ✨✨✨【診斷強化 1】✨✨✨
echo "  -> 檢查主機上的 MPS pipe 目錄權限:"
ls -ld "${MPS_PIPE_DIR}"

echo "  -> 正在啟動 MPS 測試容器..."
# ✨✨✨【核心修正】✨✨✨
# 增加 --ipc=host 來共享主機的 IPC namespace
docker run --rm \
    --name "mps-vram-test" \
    --gpus device=${GPU_DEVICE_ID} \
    --ipc=host \
    --shm-size=1g \
    -v "${MPS_PIPE_DIR}:${MPS_PIPE_DIR}" \
    -v "${MPS_LOG_DIR}:${MPS_LOG_DIR}" \
    -e CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID} \
    -e CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    -v "$(pwd)/build":/test_build \
    "${IMAGE_NAME}" \
    bash -c "echo '--- 容器內部權限檢查 ---'; ls -la ${MPS_PIPE_DIR}; echo '--- 開始執行測試程式 ---'; ${TEST_PROGRAM_PATH}" || true
# 使用 `|| true` 確保即使容器內程式失敗，腳本也會繼續執行下面的診斷

# ✨✨✨【診斷強化 2】✨✨✨
echo "  -> 容器執行完畢。檢查是否有 MPS server 日誌..."
# MPS server 的日誌會以 server-<PID>.log 的形式存在
server_log=$(sudo ls -t ${MPS_LOG_DIR}/server-*.log 2>/dev/null | head -n 1)
if [ -n "$server_log" ] && [ -f "$server_log" ]; then
    echo "  -> 發現最新的 server 日誌: $server_log"
    echo "==================== MPS server.log ===================="
    sudo cat "$server_log"
    echo "========================================================="
else
    echo "  -> 未找到 MPS server 日誌。"
fi

echo "✅ MPS 測試完成."; echo "--------------------------------------------------"; echo "所有測試已執行完畢。"