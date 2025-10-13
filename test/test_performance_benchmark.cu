#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "test_utils.h"

// Kernel 內部計算迴圈次數 (從 50 降低到 5)
#define INNER_LOOPS 5
// Kernel 總啟動次數 (從 500 降低到 200)
#define NUM_LAUNCHES 200

__global__ void computeKernel(double* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        double val = data[tid];
        for(int i = 0; i < INNER_LOOPS; ++i) { 
            val = sin(val) * cos(val);
        }
        data[tid] = val;
    }
}

int main() {
    int N = 1 << 26;
    double* d_data;

    cudaError_t err = cudaMalloc(&d_data, N * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        return -1;
    }

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("開始快速效能評測，將執行 %d 次 Kernel 啟動...\n", NUM_LAUNCHES);

    cudaEventRecord(start);

    for (int i = 0; i < NUM_LAUNCHES; ++i) {
        computeKernel<<<blocks, threadsPerBlock>>>(d_data, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    float iterations_per_second = NUM_LAUNCHES / seconds;

    const char* sm_limit_env = getenv("CUDA_DEVICE_SM_LIMIT");
    printf("\n--- 評測結果 ---\n");
    printf("算力限制設定:   %s%%\n", sm_limit_env ? sm_limit_env : "100 (Default)");
    printf("總共執行次數:   %d\n", NUM_LAUNCHES);
    printf("總耗時 (秒):    %.4f\n", seconds);
    printf("效能:           %.2f launches/sec\n", iterations_per_second);
    printf("--------------------\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);

    return 0;
}