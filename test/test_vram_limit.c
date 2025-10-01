#include <stdio.h>
#include <unistd.h>
#include <cuda.h>
#include "test_utils.h"

// 每次分配的記憶體增量 (100 MB)
#define ALLOC_INCREMENT (100 * 1024 * 1024)

int main() {
    CUresult res;
    CUdevice device;
    CUcontext ctx;
    CUdeviceptr dptr;
    size_t total_allocated = 0;
    int allocation_count = 0;

    // --- 初始化 CUDA ---
    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", res);
        return -1;
    }

    res = cuDeviceGet(&device, TEST_DEVICE_ID);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGet failed: %d\n", res);
        return -1;
    }

    res = cuCtxCreate(&ctx, 0, device);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxCreate failed: %d\n", res);
        return -1;
    }

    printf("開始 VRAM 分配測試，每次增加 100MB...\n");
    printf("預計在達到設定的 VRAM 上限時會出現 Out of Memory 錯誤。\n\n");

    // --- 進入無限迴圈，持續分配記憶體 ---
    while (1) {
        // 嘗試分配 100MB
        res = cuMemAlloc(&dptr, ALLOC_INCREMENT);

        // --- 檢查分配結果 ---
        if (res != CUDA_SUCCESS) {
            // 如果錯誤是 Out of Memory，代表我們的限制成功生效
            if (res == CUDA_ERROR_OUT_OF_MEMORY) {
                printf("\n=========================================================\n");
                printf("成功捕獲預期的 CUDA_ERROR_OUT_OF_MEMORY 錯誤！\n");
                printf("VRAM 限制功能正常運作。\n");
                printf("在 OOM 之前，總共成功分配了: %.2f MB\n", (double)total_allocated / (1024 * 1024));
                printf("=========================================================\n");
                break; // 測試成功，跳出迴圈
            } else {
                // 如果是其他非預期的錯誤，則印出錯誤訊息
                fprintf(stderr, "\ncuMemAlloc 發生非預期的錯誤: %d\n", res);
                cuCtxDestroy(ctx);
                return -1;
            }
        }

        // 分配成功，更新計數並印出目前狀態
        total_allocated += ALLOC_INCREMENT;
        allocation_count++;
        printf("第 %d 次分配: 成功分配 100MB. 目前總分配量: %.2f MB\n",
               allocation_count, (double)total_allocated / (1024 * 1024));

        // 你可以取消下面這行的註解，讓每次分配之間暫停一下，方便觀察
        // sleep(1);
    }

    // --- 清理資源 ---
    cuCtxDestroy(ctx);
    printf("\nVRAM 限制測試完成。\n");

    return 0;
}