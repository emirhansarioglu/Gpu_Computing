#include <chrono>
#include <curand.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>

#include "helper.cu"

__global__ void kogge_stone_shared(float *in_d, float *out_d, size_t n_complex, int stride) {
    extern __shared__ float shared_mem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load data into shared memory 
    // Memory coalescing: consecutive threads access consecutive memory
    if (idx < n_complex) {
        shared_mem[tid * 2] = in_d[idx * 2];
        shared_mem[tid * 2 + 1] = in_d[idx * 2 + 1];
    }
    __syncthreads();
    
    int local_prev = tid - stride;
    bool should_compute = (idx >= stride) && (idx < n_complex);
    bool use_shared = (local_prev >= 0) && should_compute;
    
    float real_cur, im_cur, real_prev, im_prev;
    
    if (should_compute) {
        real_cur = shared_mem[tid * 2];
        im_cur = shared_mem[tid * 2 + 1];
        
        if (use_shared) {
            // Previous values from shared memory 
            real_prev = shared_mem[local_prev * 2];
            im_prev = shared_mem[local_prev * 2 + 1];
        } else {
            // Previous values from global memory (on an another block)
            real_prev = in_d[(idx - stride) * 2];
            im_prev = in_d[(idx - stride) * 2 + 1];
        }
        
        out_d[idx * 2] = real_prev * real_cur - im_prev * im_cur;
        out_d[idx * 2 + 1] = real_prev * im_cur + real_cur * im_prev;

    } else if (idx < n_complex) {
        out_d[idx * 2] = shared_mem[tid * 2];
        out_d[idx * 2 + 1] = shared_mem[tid * 2 + 1];
    }
}

// Thread coarsening kernel: each thread processes multiple elements
__global__ void kogge_stone_coarsened(float *in_d, float *out_d, size_t n_complex, int stride, int elements_per_thread) {
    extern __shared__ float shared_mem[];
    
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
    int tid = threadIdx.x;
    
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = base_idx + i;
        int shared_idx = tid * elements_per_thread + i;
        
        if (idx < n_complex && shared_idx < blockDim.x * elements_per_thread) {
            shared_mem[shared_idx * 2] = in_d[idx * 2];
            shared_mem[shared_idx * 2 + 1] = in_d[idx * 2 + 1];
        }
    }
    __syncthreads();
    
    // Process multiple elements per thread
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = base_idx + i;
        int shared_idx = tid * elements_per_thread + i;
        
        if (idx >= n_complex) break;
        
        int local_prev = shared_idx - stride;
        bool should_compute = (idx >= stride);
        bool use_shared = (local_prev >= 0) && should_compute;
        
        float real_cur, im_cur, real_prev, im_prev;
        
        if (should_compute) {
            real_cur = shared_mem[shared_idx * 2];
            im_cur = shared_mem[shared_idx * 2 + 1];
            
            if (use_shared) {
                real_prev = shared_mem[local_prev * 2];
                im_prev = shared_mem[local_prev * 2 + 1];
            } else {
                real_prev = in_d[(idx - stride) * 2];
                im_prev = in_d[(idx - stride) * 2 + 1];
            }
            
            out_d[idx * 2] = real_prev * real_cur - im_prev * im_cur;
            out_d[idx * 2 + 1] = real_prev * im_cur + real_cur * im_prev;
        } else {
            out_d[idx * 2] = shared_mem[shared_idx * 2];
            out_d[idx * 2 + 1] = shared_mem[shared_idx * 2 + 1];
        }
    }
}

void parallel_scan(size_t size, float *in_d, float *out_d) {
    size_t n_complex = size / 2;
    
    int threads_per_block = 256;
    int elements_per_thread = 4; // Thread coarsening factor
    
    cudaMemcpy(out_d, in_d, size * sizeof(float), cudaMemcpyDeviceToDevice);
    
    float *temp_d;
    cudaMalloc((void **)&temp_d, size * sizeof(float));
    
    int num_iterations = (int)ceil(log2((double)n_complex));
    
    // Ping-pong between output and temp
    for (int iter = 0; iter < num_iterations; iter++) {
        int stride = 1 << iter;
        
        if (stride < threads_per_block * elements_per_thread) {
            // earlier iterations where stride is small --> thread coarsening
            int num_blocks = (n_complex + (threads_per_block * elements_per_thread) - 1) / (threads_per_block * elements_per_thread);
            size_t shared_size = threads_per_block * elements_per_thread * 2 * sizeof(float);
            
            if (iter % 2 == 0) {
                kogge_stone_coarsened<<<num_blocks, threads_per_block, shared_size>>>(
                    out_d, temp_d, n_complex, stride, elements_per_thread);
            } else {
                kogge_stone_coarsened<<<num_blocks, threads_per_block, shared_size>>>(
                    temp_d, out_d, n_complex, stride, elements_per_thread);
            }
        } else {
            // later iterations where stride is big --> shared memory
            int num_blocks = (n_complex + threads_per_block - 1) / threads_per_block;
            size_t shared_size = threads_per_block * 2 * sizeof(float);
            
            if (iter % 2 == 0) {
                kogge_stone_shared<<<num_blocks, threads_per_block, shared_size>>>(
                    out_d, temp_d, n_complex, stride);
            } else {
                kogge_stone_shared<<<num_blocks, threads_per_block, shared_size>>>(
                    temp_d, out_d, n_complex, stride);
            }
        }
        
        cudaDeviceSynchronize();
    }
    
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