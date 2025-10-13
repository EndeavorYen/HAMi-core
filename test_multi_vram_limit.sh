#!/bin/bash

# --- 測試設定 ---
IMAGE_NAME="cuda_vmem:tf1.8-cu90"
VRAM_LIMITS=("1g" "4g" "6g")
GPU_DEVICE_ID=0

# --- 清理函式 ---
# 當腳本被中斷 (Ctrl+C) 或正常結束時，此函式會被呼叫
cleanup() {
    echo ""
    echo "--------------------------------------------------"
    echo "正在停止並清理所有測試容器..."
    # 如果 container_names 陣列不為空
    if [ ${#container_names[@]} -ne 0 ]; then
        docker stop "${container_names[@]}" > /dev/null
        echo "所有容器已停止。"
    fi
    # 殺掉所有在背景執行的 docker logs 程序
    kill 0
}

# --- 主程式開始 ---
echo "開始多容器 VRAM 限制並行壓力測試..."
echo "按下 Ctrl+C 可隨時停止並清理所有容器。"
echo "--------------------------------------------------"

# 設定 trap，攔截 Ctrl+C (SIGINT) 和終止信號 (SIGTERM)
trap cleanup SIGINT SIGTERM

# 檢查必要的 build 目錄是否存在
if [ ! -d "build" ] || [ ! -f "build/test/test_vram_limit" ]; then
    echo "錯誤：找不到 build/test/test_vram_limit 執行檔。"
    echo "請先執行 'make build-in-docker' 進行編譯。"
    exit 1
fi

# 用於存放 container 名稱的陣列
container_names=()

# --- 啟動容器 ---
for limit in "${VRAM_LIMITS[@]}"; do
    name="vram-test-${limit}"
    container_names+=("${name}")
    
    echo "正在背景啟動容器: ${name} (VRAM 限制: ${limit})..."
    
    # 在背景 (-d) 啟動容器
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
echo "所有容器已啟動，正在即時追蹤日誌..."
echo ""

# --- 平行顯示日誌 ---
# 為每個容器在背景啟動一個 'docker logs' 程序
# 每個程序的輸出都透過 sed 加上容器名稱前綴
for name in "${container_names[@]}"; do
    docker logs -f "${name}" | sed "s/^/[${name}] /" &
done

# --- 等待與收尾 ---
# 'wait' 會讓腳本在此暫停，直到所有背景程序 (docker logs) 結束
# 當你按下 Ctrl+C 時，trap 會觸發 cleanup，殺掉所有背景程序，wait 也隨之結束
wait