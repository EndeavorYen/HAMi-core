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

    // --- Initialize CUDA ---
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

    printf("Starting VRAM allocation test, increasing by 100MB each time...\n");
    printf("An Out of Memory error is expected when the VRAM limit is reached.\n\n");

    // --- Enter an infinite loop to continuously allocate memory ---
    while (1) {
        // Attempt to allocate 100MB
        res = cuMemAlloc(&dptr, ALLOC_INCREMENT);

        // --- Check allocation result ---
        if (res != CUDA_SUCCESS) {
            // If the error is Out of Memory, it means the limit was successfully enforced
            if (res == CUDA_ERROR_OUT_OF_MEMORY) {
                printf("\n=========================================================\n");
                printf("Successfully caught the expected CUDA_ERROR_OUT_OF_MEMORY error!\n");
                printf("VRAM limit functionality is working correctly.\n");
                printf("Total successfully allocated before OOM: %.2f MB\n", (double)total_allocated / (1024 * 1024));
                printf("=========================================================\n");
                break; // Test successful, exit the loop
            } else {
                // If it's an unexpected error, print the error message
                fprintf(stderr, "\ncuMemAlloc encountered an unexpected error: %d\n", res);
                cuCtxDestroy(ctx);
                return -1;
            }
        }

        // Allocation successful, update counters and print current status
        total_allocated += ALLOC_INCREMENT;
        allocation_count++;
        printf("Allocation #%d: Successfully allocated 100MB. Total allocated: %.2f MB\n",
               allocation_count, (double)total_allocated / (1024 * 1024));

        // You can uncomment the following line to pause between allocations for observation
        // sleep(1);
    }

    // --- Clean up resources ---
    cuCtxDestroy(ctx);
    printf("\nVRAM limit test completed.\n");

    return 0;
}