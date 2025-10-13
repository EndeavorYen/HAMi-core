#!/bin/bash

# --- 測試設定 ---
IMAGE_NAME="cuda_vmem:tf1.8-cu90"
GPU_DEVICE_ID=0
# 確保指向正確的、新編譯的評測程式
BENCHMARK_PROGRAM="/test_build/test/test_performance_benchmark"

# --- 主程式 ---
echo "開始 GPU 效能階梯測試 (從 100% 到 10%)..."
echo "=============================================================="

# 檢查評測程式是否存在
if [ ! -f "build/test/test_performance_benchmark" ]; then
    echo "錯誤: 找不到評測程式 'build/test/test_performance_benchmark'。"
    echo "請確認您已建立 .cu 檔案並重新執行 'make build-in-docker'。"
    exit 1
fi

# 迴圈從 100 遞減到 10，每次遞減 10
for limit in $(seq 100 -10 10); do
    echo ""
    echo "--- 正在測試 ${limit}% 算力限制下的效能 ---"
    
    # 啟動一個新的、乾淨的容器來執行單次評測
    # 容器執行完畢後會自動移除 (--rm)
    docker run --rm \
        --gpus device=${GPU_DEVICE_ID} \
        --mount type=tmpfs,destination=/tmp/vgpulock \
        -v "$(pwd)/build":/test_build \
        -e CUDA_DEVICE_SM_LIMIT="${limit}" \
        -e LD_PRELOAD=/test_build/libvgpu.so \
        "${IMAGE_NAME}" \
        "${BENCHMARK_PROGRAM}"

    # 檢查 Docker 指令是否成功執行
    if [ $? -ne 0 ]; then
        echo "錯誤: 於 ${limit}% 算力限制下執行 Docker 失敗，測試中止。"
        exit 1
    fi
    
    echo "--- ${limit}% 算力測試完成 ---"
    sleep 2 # 每次測試之間短暫休息
done

echo ""
echo "=============================================================="
echo "所有效能評測已完成。"