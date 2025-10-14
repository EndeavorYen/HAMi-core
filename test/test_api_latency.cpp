#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

#define CHECK_CUDA_RUNTIME_RESULT(res, msg) \
    if (res != cudaSuccess) { \
        fprintf(stderr, "CUDA Runtime error: %s failed with %s\n", msg, cudaGetErrorString(res)); \
        return -1; \
    }

int main() {
    cudaError_t res;

    // 1. 初始化 Runtime API
    res = cudaFree(0);
    CHECK_CUDA_RUNTIME_RESULT(res, "cudaFree (for initialization)");

    int device_count = 0;

    const int warm_up_iterations = 1000;
    // ✨✨✨【穩定性強化】✨✨✨ 增加 10 倍的測量迭代次數以降低雜訊
    const int test_iterations = 1000000;
    const int measurement_rounds = 10;

    // 2. 預熱
    for (int i = 0; i < warm_up_iterations; ++i) {
        res = cudaGetDeviceCount(&device_count);
        if (res != cudaSuccess) {
             CHECK_CUDA_RUNTIME_RESULT(res, "cudaGetDeviceCount (warm-up)");
             return 1;
        }
    }

    std::vector<double> round_latencies;

    printf("Starting performance measurement...\n");
    printf("Rounds: %d, Iterations per round: %d\n", measurement_rounds, test_iterations);
    printf("====================================================\n");

    for (int j = 0; j < measurement_rounds; ++j) {
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < test_iterations; ++i) {
            cudaGetDeviceCount(&device_count);
        }

        auto end_time = std::chrono::high_resolution_clock::now();

        res = cudaGetDeviceCount(&device_count);
        CHECK_CUDA_RUNTIME_RESULT(res, "cudaGetDeviceCount (measurement)");

        std::chrono::duration<double, std::micro> elapsed_microseconds = end_time - start_time;
        double average_latency_us = elapsed_microseconds.count() / test_iterations;
        round_latencies.push_back(average_latency_us);
    }
    printf("====================================================\n");

    double sum = std::accumulate(round_latencies.begin(), round_latencies.end(), 0.0);
    double mean = sum / round_latencies.size();

    double sq_sum = std::inner_product(round_latencies.begin(), round_latencies.end(), round_latencies.begin(), 0.0);
    double stddev = std::sqrt(sq_sum / round_latencies.size() - mean * mean);

    std::sort(round_latencies.begin(), round_latencies.end());
    double median = (round_latencies.size() % 2 == 1)
                    ? round_latencies[round_latencies.size() / 2]
                    : (round_latencies[round_latencies.size() / 2 - 1] + round_latencies[round_latencies.size() / 2]) / 2.0;

    printf("Final Statistics for cudaGetDeviceCount() Latency:\n");
    printf("  - Average: %.4f us\n", mean);
    printf("  - Median:  %.4f us\n", median);
    printf("  - StdDev:  %.4f us\n", stddev);

    return 0;
}