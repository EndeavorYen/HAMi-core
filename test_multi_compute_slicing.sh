#!/bin/bash

# --- 測試設定 ---
IMAGE_NAME="cuda_vmem:tf1.8-cu90"
SM_LIMITS=(20 30 40)
GPU_DEVICE_ID=0
TEST_PROGRAM_PATH="/test_build/test/test_runtime_launch" 

# --- 全域變數，用於存放背景 PID ---
declare -a child_pids=()
declare -a container_names=()

# --- 清理函式 (修正版) ---
cleanup() {
    echo ""
    echo "--------------------------------------------------"
    echo "捕獲到中斷信號，正在停止所有背景程序與容器..."

    # 1. 優先終止所有背景的 'docker logs' 程序
    # 這是讓主腳本的 'wait' 得以解除的關鍵
    if [ ${#child_pids[@]} -ne 0 ]; then
        echo "正在終止日誌追蹤程序..."
        kill "${child_pids[@]}" 2>/dev/null
    fi

    # 2. 停止所有測試容器
    if [ ${#container_names[@]} -ne 0 ]; then
        echo "正在停止 Docker 容器..."
        docker stop "${container_names[@]}" > /dev/null
        echo "所有容器已停止。"
    fi
    
    echo "清理完畢。"
    # 確保腳本在 trap 執行完畢後乾淨地退出
    exit 0
}

# --- 主程式開始 ---
echo "開始多容器 Compute Slicing 並行壓力測試..."
echo "按下 Ctrl+C 可隨時停止並清理所有容器。"
echo "--------------------------------------------------"

# 設定 trap，攔截 Ctrl+C (SIGINT) 和終止信號 (SIGTERM)
trap cleanup SIGINT SIGTERM

# 檢查測試程式是否存在
if [ ! -f "build/test/test_runtime_launch" ]; then
    echo "錯誤：找不到 'build/test/test_runtime_launch' 執行檔。"
    echo "請先執行 'make build-in-docker' 進行編譯。"
    exit 1
fi

# --- 啟動容器 ---
for limit in "${SM_LIMITS[@]}"; do
    name="compute-test-${limit}p"
    container_names+=("${name}")
    
    echo "正在背景啟動容器: ${name} (SM 限制: ${limit}%)..."
    
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
echo "所有容器已啟動，正在即時追蹤日誌..."
echo "同時，請在另一個終端機視窗執行 'watch -n 1 nvidia-smi' 來監控 GPU 總利用率。"
echo ""

# --- 平行顯示日誌 (修正版) ---
for name in "${container_names[@]}"; do
    { docker logs -f "${name}"; echo "[${name}] Process Exited."; } | sed "s/^/[${name}] /" &
    # 將剛剛啟動的背景程序的 PID 存入陣列
    child_pids+=($!)
done

# 'wait' 會讓腳本在此暫停，直到 trap 被觸發並執行 exit
wait