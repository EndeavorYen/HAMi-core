#!/bin/bash

# --- 測試設定 ---
# 基礎 Docker Image 名稱
IMAGE_NAME="cuda_vmem:tf1.8-cu90"
# VRAM 限制陣列，你可以依需求修改
# VRAM_LIMITS=("1g" "4g" "7g")
VRAM_LIMITS=("1g")
# 要使用的 GPU 設備 ID
GPU_DEVICE_ID=0

# --- 腳本開始 ---
echo "開始多容器 VRAM 限制並行壓力測試..."
echo "--------------------------------------------------"

# 檢查必要的 build 目錄是否存在
if [ ! -d "build" ] || [ ! -f "build/test/test_vram_limit" ]; then
    echo "錯誤：找不到 build/test_vram_limit 執行檔。"
    echo "請先執行 'make build-in-docker' 進行編譯。"
    exit 1
fi

# 用於存放 container ID 的陣列
container_ids=()

# 啟動所有測試容器
for limit in "${VRAM_LIMITS[@]}"; do
    container_name="vram-test-${limit}"
    echo "正在啟動容器: ${container_name} (VRAM 限制: ${limit})..."
    
    # 在背景 (-d) 啟動容器，並直接執行測試程式
    cid=$(docker run -d --rm \
        --name "${container_name}" \
        --gpus device=${GPU_DEVICE_ID} \
        --mount type=tmpfs,destination=/tmp/vgpulock \
        -v "$(pwd)/build":/test_build \
        -e CUDA_DEVICE_MEMORY_LIMIT="${limit}" \
        -e LD_PRELOAD=/test_build/libvgpu.so \
        "${IMAGE_NAME}" \
        /test_build/test/test_vram_limit)

    
    # 檢查容器是否成功啟動
    if [ $? -ne 0 ]; then
        echo "錯誤：啟動容器 ${container_name} 失敗！"
    else
        echo "容器 ${container_name} 已啟動，ID: ${cid:0:12}"
        container_ids+=("${container_name}")
    fi
done

echo "--------------------------------------------------"
echo "所有測試容器皆已在背景啟動。"
echo "您可以使用以下指令分別查看各個容器的即時日誌："
echo ""
for name in "${container_ids[@]}"; do
    echo "  docker logs -f ${name}"
done
echo ""
echo "預期結果：每個容器的輸出應顯示其在各自的 VRAM 限制下觸發 OOM。"
echo "--------------------------------------------------"
echo "測試執行中... 當您想結束測試時，請執行以下指令來停止所有容器："
echo ""
echo "  docker stop ${container_ids[@]}"
echo ""