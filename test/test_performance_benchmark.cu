#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "test_utils.h"

// Number of inner loop iterations in the kernel (reduced from 50 to 5)
// #define INNER_LOOPS 5
#define INNER_LOOPS 50
// Total number of kernel launches (reduced from 500 to 100)
#define NUM_LAUNCHES 100

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

    printf("Starting performance benchmark, executing %d kernel launches...\n", NUM_LAUNCHES);

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
    printf("\n--- Benchmark Results ---\n");
    printf("SM Limit Setting:   %s%%\n", sm_limit_env ? sm_limit_env : "100 (Default)");
    printf("Total Launches:     %d\n", NUM_LAUNCHES);
    printf("Total Time (sec):   %.4f\n", seconds);
    printf("Performance:        %.2f launches/sec\n", iterations_per_second);
    printf("-------------------------\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);

    return 0;
}