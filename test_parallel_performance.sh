#!/bin/bash

# --- 測試設定 ---
IMAGE_NAME="cuda_vmem:tf1.8-cu90"
# 為三個容器設定不同的算力限制，總和為 90%
SM_LIMITS=(10 11 12)
GPU_DEVICE_ID=0
# 執行我們新建的效能評測程式
BENCHMARK_PROGRAM="/test_build/test/test_performance_benchmark"

# --- 全域變數 ---
declare -a child_pids=()
declare -a container_names=()

# --- 清理函式 ---
cleanup() {
    echo ""
    echo "--------------------------------------------------"
    echo "捕獲到中斷信號，正在停止所有背景程序與容器..."
    if [ ${#child_pids[@]} -ne 0 ]; then
        kill "${child_pids[@]}" 2>/dev/null
    fi
    if [ ${#container_names[@]} -ne 0 ]; then
        docker stop "${container_names[@]}" > /dev/null
        echo "所有容器已停止。"
    fi
    echo "清理完畢。"
    exit 0
}

# --- 主程式開始 ---
echo "開始多容器並行效能評測..."
echo "將啟動 3 個容器，算力限制分別為: ${SM_LIMITS[*]}%"
echo "按下 Ctrl+C 可隨時中止測試。"
echo "--------------------------------------------------"

# 設定 trap
trap cleanup SIGINT SIGTERM

# 檢查評測程式是否存在
if [ ! -f "build/test/test_performance_benchmark" ]; then
    echo "錯誤: 找不到 'build/test/test_performance_benchmark'。"
    echo "請先執行 'make build-in-docker'。"
    exit 1
fi

# --- 啟動容器 ---
for limit in "${SM_LIMITS[@]}"; do
    name="perf-test-${limit}p"
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
        "${BENCHMARK_PROGRAM}" > /dev/null
done

echo "--------------------------------------------------"
echo "所有容器已啟動，正在即時追蹤日誌..."
echo "同時，請在另一個終端機視窗執行 'watch -n 1 nvidia-smi' 來監控 GPU 總利用率。"
echo ""

# --- 平行顯示日誌 ---
for name in "${container_names[@]}"; do
    { docker logs -f "${name}"; } | sed "s/^/[${name}] /" &
    child_pids+=($!)
done

# 等待所有背景 'docker logs' 程序結束
wait
# 所有日誌程序都結束後（代表所有容器都執行完畢），再執行一次清理
cleanup