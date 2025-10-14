#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "test_utils.h"
#include <unistd.h>
#include <chrono>
#include <signal.h>
#include <cmath>

__global__ void add(float* a, float* b, float* c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

__global__ void computeKernel(double* data, int N, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        double temp = 0.0;
        temp += sin(data[tid]) * cos(data[tid]);
        data[tid] = temp;
    }
}

static volatile sig_atomic_t keep_running = 1;
static void handle_sigint(int) { keep_running = 0; }

int main(int argc, char** argv) {
    // default parameters (can be overridden via command-line)
    long long N = 1LL << 27; // number of elements
    int threadsPerBlock = 256;
    // int batch_size = 100; // number of launches per measurement batch
    int batch_size = 10; // number of launches per measurement batch

    if (argc > 1) N = atoll(argv[1]);
    if (argc > 2) batch_size = atoi(argv[2]);
    if (argc > 3) threadsPerBlock = atoi(argv[3]);

    signal(SIGINT, handle_sigint);

    float *a, *b, *c;
    CHECK_RUNTIME_API(cudaMalloc(&a, 1024 * sizeof(float)));
    CHECK_RUNTIME_API(cudaMalloc(&b, 1024 * sizeof(float)));
    CHECK_RUNTIME_API(cudaMalloc(&c, 1024 * sizeof(float)));

    add<<<1, 1024>>>(a, b, c);

    double* d_data = nullptr;
    CHECK_RUNTIME_API(cudaMalloc(&d_data, N * sizeof(double)));

    int blocks = (int)((N + threadsPerBlock - 1) / threadsPerBlock);
    int iterations = 1000000; // kept for compatibility with kernel signature

    printf("Starting kernel launches: N=%lld, threadsPerBlock=%d, blocks=%d, batch_size=%d\n",
           (long long)N, threadsPerBlock, blocks, batch_size);

    // Create CUDA events for GPU timing
    cudaEvent_t ev_start, ev_stop;
    CHECK_RUNTIME_API(cudaEventCreate(&ev_start));
    CHECK_RUNTIME_API(cudaEventCreate(&ev_stop));

    long long batch_count = 0;
    while (keep_running) {
        // Host wall-clock start
        auto wall_start = std::chrono::high_resolution_clock::now();

        // GPU start event
        CHECK_RUNTIME_API(cudaEventRecord(ev_start, 0));
        int launches_done = 0;
        for (int i = 0; i < batch_size && keep_running; ++i) {
            computeKernel<<<blocks, threadsPerBlock>>>(d_data, (int)N, iterations);
            ++launches_done;
        }
        // If no launches were done because user already requested stop, break
        if (launches_done == 0) break;

        // GPU stop event (mark end of launched work)
        CHECK_RUNTIME_API(cudaEventRecord(ev_stop, 0));

        // Poll for completion so we can respond quickly to SIGINT
        bool completed = false;
        while (keep_running) {
            cudaError_t qerr = cudaEventQuery(ev_stop);
            if (qerr == cudaSuccess) { completed = true; break; }
            if (qerr != cudaErrorNotReady) {
                // some error occurred
                fprintf(stderr, "cudaEventQuery error: %s\n", cudaGetErrorString(qerr));
                break;
            }
            // short sleep to avoid busy-waiting
            usleep(1000); // 1ms
        }
        if (!keep_running && !completed) {
            // User requested interrupt while GPU still running.
            // Reset device to abort ongoing kernels and make exit responsive.
            fprintf(stderr, "SIGINT received: resetting CUDA device to abort running kernels...\n");
            cudaDeviceReset();
            break;
        }

        // Host wall-clock end
        auto wall_end = std::chrono::high_resolution_clock::now();

        float gpu_ms = 0.0f;
        // Only query elapsed time if event completed
        if (completed) {
            CHECK_RUNTIME_API(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop)); // ms
        } else {
            gpu_ms = -1.0f;
        }

        std::chrono::duration<double> wall_dur = wall_end - wall_start;
        double wall_ms = wall_dur.count() * 1000.0;

        double avg_ms_per_launch_gpu = (gpu_ms > 0.0) ? (gpu_ms / launches_done) : -1.0;
        double avg_ms_per_launch_wall = wall_ms / launches_done;
        double launches_per_sec = 1000.0 / avg_ms_per_launch_wall;
        double elements_per_sec = (double)N * launches_done / (wall_ms / 1000.0);

        batch_count++;
        printf("[batch %lld] launches: %d, GPU time: %.3f ms, Host time: %.3f ms, avg GPU ms/launch: %.3f, avg host ms/launch: %.3f, launches/s: %.2f, elements/s: %.2f\n",
               batch_count, launches_done, gpu_ms, wall_ms, avg_ms_per_launch_gpu, avg_ms_per_launch_wall, launches_per_sec, elements_per_sec);

        // small sleep to avoid flooding output (optional)
        usleep(200000); // 200ms
    }

    // cleanup
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_data);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    printf("terminated by user, cleaned up\n");
    return 0;
}
