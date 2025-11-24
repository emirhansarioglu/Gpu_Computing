#include <chrono>
#include <curand.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>

#include "helper.cu"

// Kernel for Kogge-Stone parallel scan with complex multiplication
__global__ void kogge_stone_scan(float *in_d, float *out_d, size_t n_complex, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_complex && idx >= stride) {
        float real_cur = in_d[idx * 2];
        float im_cur = in_d[idx * 2 + 1];
        
        // Previous complex number at position (idx - stride)
        float real_prev = in_d[(idx - stride) * 2];
        float im_prev = in_d[(idx - stride) * 2 + 1];
        
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        out_d[idx * 2] = real_prev * real_cur - im_prev * im_cur;
        out_d[idx * 2 + 1] = real_prev * im_cur + real_cur * im_prev;
    } else if (idx < n_complex) {
        // Elements before stride remain unchanged
        out_d[idx * 2] = in_d[idx * 2];
        out_d[idx * 2 + 1] = in_d[idx * 2 + 1];
    }
}

void parallel_scan(size_t size, float *in_d, float *out_d) {
    size_t n_complex = size / 2; 
    
    int threads_per_block = 256;
    int num_blocks = (n_complex + threads_per_block - 1) / threads_per_block;
    
    // Copy input to output 
    cudaMemcpy(out_d, in_d, size * sizeof(float), cudaMemcpyDeviceToDevice);
    
    float *temp_d;
    cudaMalloc((void **)&temp_d, size * sizeof(float));
    
    int num_iterations = (int)ceil(log2((double)n_complex));
    
    // Ping-pong between output and temp
    for (int iter = 0; iter < num_iterations; iter++) {
        int stride = 1 << iter; // stride = 2^iter
        
        if (iter % 2 == 0) {
            kogge_stone_scan<<<num_blocks, threads_per_block>>>(out_d, temp_d, n_complex, stride);
        } else {
            kogge_stone_scan<<<num_blocks, threads_per_block>>>(temp_d, out_d, n_complex, stride);
        }
        
        cudaDeviceSynchronize();
    }
    
    // Make sure result is in out_d
    if (num_iterations % 2 == 1) {
        cudaMemcpy(out_d, temp_d, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(temp_d);
}

int main() {
    size_t size = 33554432 * 2;
    float *in_d, *in_h, *out_d, *out_h;

    // Allocate on host
    in_h = (float *)calloc(size, sizeof(float));
    CHECK_ALLOC(in_h);
    out_h = (float *)calloc(size, sizeof(float));
    CHECK_ALLOC(out_h);
    
    // Allocate on device
    CUDA_CALL(cudaMalloc((void **)&in_d, size * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&out_d, size * sizeof(float)));

    // Initialize
    int e = random_init(size, in_d, in_h);
    if (e == EXIT_FAILURE)
        return EXIT_FAILURE;

    auto start = std::chrono::system_clock::now();
    parallel_scan(size, in_d, out_d);
    auto end = std::chrono::system_clock::now();

    // Copy result to host
    CUDA_CALL(cudaMemcpy(out_h, out_d, size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "First 5 entries of In Vec:" << std::endl;
    for (int32_t i = 0; i < 5 * 2; i += 2)
        std::cout << in_h[i] << " + " << in_h[i + 1] << "i" << std::endl;
    
    std::cout << "First 5 entries of Out Vec:" << std::endl;
    for (int32_t i = 0; i < 5 * 2; i += 2)
        std::cout << out_h[i] << " + " << out_h[i + 1] << "i" << std::endl;

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    free(in_h);
    free(out_h);
    
    return EXIT_SUCCESS;
}