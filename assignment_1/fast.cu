#include <random>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

void init(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    // std::random_device dev;
    std::mt19937 prng(2024);
    std::uniform_int_distribution<int32_t> distrib(-16, 16);

    for (auto i = 0; i < size; i++)
    {
        vec_a[i] = distrib(prng);
        vec_b[i] = distrib(prng);
    }

    for (auto i = 0; i < size * size; i++)
        mat[i] = distrib(prng);
}


void pretty_print(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    std::cout << "Vec A:" << std::endl;
    for (auto i = 0; i < size; i++)
        std::cout << vec_a[i] << std::endl;

    std::cout << "Vec B:" << std::endl;
    for (auto i = 0; i < size; i++)
        std::cout << vec_b[i] << std::endl;

    std::cout << "Matrix:" << std::endl;
    for (auto i = 0; i < size; i++)
    {
        for (auto j = 0; j < size; j++)
            std::cout << mat[i * size + j] << " ";

        std::cout << std::endl;
    }
}

void check(cudaError_t err, std::string msg)
{
   if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << msg
              << " (error code " << cudaGetErrorString(err) << ")\n";
    exit(EXIT_FAILURE);
    }
}

__global__ void compute_kernel(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat, int32_t *out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
    {
        int32_t sum = 0;
        for (int j = 0; j < size; j++)
        {
            sum += (vec_a[j] + vec_b[j]) * mat[i * size + j];
        }
        out[i] = sum;
    }
}

int main() 
{
    cudaError_t err = cudaSuccess;

    // host targeted code is same as cpu only version which was given with the task (slow.cpp) 
    int32_t size = 32768;
    auto vec_a = (int32_t *)malloc(sizeof(int32_t) * size);
    auto vec_b = (int32_t *)malloc(sizeof(int32_t) * size);
    auto mat = (int32_t *)malloc(sizeof(int32_t) * size * size);
    auto out = (int32_t *)malloc(sizeof(int32_t) * size);

    init(size, vec_a, vec_b, mat);

    // memory allocation on gpu
    int32_t *gpu_a = nullptr;
    err = cudaMalloc((void **)&gpu_a, size * sizeof(int32_t));
    check(err, "Failed allocate vector a on gpu");

    int32_t *gpu_b = nullptr;
    err = cudaMalloc((void **)&gpu_b, size * sizeof(int32_t));
    check(err, "Failed allocate vector b on gpu");

    int32_t *gpu_mat = nullptr;
    err = cudaMalloc((void **)&gpu_mat, size * size * sizeof(int32_t));
    check(err, "Failed allocate matrix on gpu");

    int32_t *gpu_out = nullptr;
    err = cudaMalloc((void **)&gpu_out, size * sizeof(int32_t));
    check(err, "Failed allocate output vector on gpu");

    // copy vectors and matrix from host to gpu
    err = cudaMemcpy(gpu_a, vec_a, size * sizeof(int32_t), cudaMemcpyHostToDevice);
    check(err, "Failed to copy vector a to gpu");

    err = cudaMemcpy(gpu_b, vec_b, size * sizeof(int32_t), cudaMemcpyHostToDevice);
    check(err, "Failed to copy vector b to gpu");

    err = cudaMemcpy(gpu_mat, mat, size * size * sizeof(int32_t), cudaMemcpyHostToDevice);
    check(err, "Failed to copy matrix to gpu");

    // launch the cuda kernel for compute
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock; // 128
    auto start = std::chrono::system_clock::now();
    compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(size, gpu_a, gpu_b, gpu_mat, gpu_out);
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    check(cudaGetLastError(), "Failed to launch the kernel");

    // copy the output to host
    err = cudaMemcpy(out, gpu_out, size * sizeof(int32_t), cudaMemcpyDeviceToHost);
    check(err, "Failed to copy the result to host");

    // deallocate memory from gpu
    err = cudaFree(gpu_a);
    check(err, "Failed to deallocate vector a from gpu");

    err = cudaFree(gpu_b);
    check(err, "Failed to deallocate vector b from gpu");
    
    err = cudaFree(gpu_out);
    check(err, "Failed to deallocate output vector from gpu");

    err = cudaFree(gpu_mat);
    check(err, "Failed to deallocate matrix from gpu");

    // rest of the host code

    std::cout << "First 3 entries of Out Vec:" << std::endl;
    for (int32_t i = 0; i < 3; i++)
        std::cout << out[i] << std::endl;

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    free(vec_a);
    free(vec_b);
    free(mat);
    free(out);

    return 0;

}