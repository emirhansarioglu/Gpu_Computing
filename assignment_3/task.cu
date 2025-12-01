#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#define CHECK_CUDA(call)                                        \
    if ((call) != cudaSuccess)                                  \
    {                                                           \
        std::cerr << "CUDA error at " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                     \
    }

const int NUM_MATRICES = 10; // Number of matrix multiplications
const int MATRIX_SIZE = 4096;
const int TILE_SIZE = 32;
const int NUM_STREAMS = 10;

// Simple kernel for matrix multiplication
__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int n)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tiled kernel for matrix multiplication
__global__ void matrixMultiplyKernelTiled(const float *A, const float *B, float *C, int n)
{
    // TODO: allocate shared memory for two tiles (one for A and one for B)
    extern __shared__ float shared_memory[];
    float *As = shared_memory;
    float *Bs = &shared_memory[TILE_SIZE * TILE_SIZE]; // B starts after A

    // Global row and column index of the C element this thread computes
    int global_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int global_col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Local row and column index within the tile (0 to TILE_SIZE - 1)
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    float C_value = 0.0f;

    // TODO: iterate over tiles
    for (int k_tile = 0; k_tile < n / TILE_SIZE; ++k_tile)
    {
        // Calculate the starting index in global memory for the current tiles of A and B
        int A_global_start_col = k_tile * TILE_SIZE;
        int B_global_start_row = k_tile * TILE_SIZE;

        // TODO: copy tiles from global memory into shared memory

        // Each thread loads one element from Global Memory into Shared Memory
        // As (TILE_SIZE x TILE_SIZE) block from A (row: global_row, col: k_tile * TILE_SIZE + local_col)
        // Bs (TILE_SIZE x TILE_SIZE) block from B (row: k_tile * TILE_SIZE + local_row, col: global_col)

        // matrix size is NOT a multiple of TILE_SIZE --> no edge case
        As[local_row * TILE_SIZE + local_col] = A[global_row * n + (A_global_start_col + local_col)];   
        Bs[local_row * TILE_SIZE + local_col] = B[(B_global_start_row + local_row) * n + global_col];
        
        // Wait for all threads in the block to finish loading the tiles
        __syncthreads();

        // TODO: compute the matrix multiplication of the two tiles
        // Threads compute the dot product of a row of As and a column of Bs
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            // The thread computes the partial C_value by multiplying the element
            // from its assigned row of As and its assigned column of Bs
            C_value += As[local_row * TILE_SIZE + k] * Bs[k * TILE_SIZE + local_col];
        }

        // Wait for all threads to finish computing the partial results before the next tile load
        __syncthreads();
    } 

    // TODO: write back the results into the matrix C
    if (global_row < n && global_col < n)
    {
        C[global_row * n + global_col] = C_value;
    }
}

// Divided this given function into more for loop to  fairly benchmark against matrixMultiplyWithStream
void matrixMultiplyNoStreams()
{
    // Host and device pointers
    float *h_A[NUM_MATRICES], *h_B[NUM_MATRICES], *h_C[NUM_MATRICES];
    float *d_A[NUM_MATRICES], *d_B[NUM_MATRICES], *d_C[NUM_MATRICES];

    for (int i = 0; i < NUM_MATRICES; i++)
    {
        
        h_A[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        h_B[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        h_C[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

        // Initialize example matrices with random numbers
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++)
        {
            // pick testing values, that allow us to compute the expected result on the CPU cheaply
            h_A[i][j] = 1.0f;
            h_B[i][j] = 0.01f;
            h_C[i][j] = 0.0f;
        }

        CHECK_CUDA(cudaMalloc(&d_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
    }
    // Launch matrix multiplication kernel
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(MATRIX_SIZE/TILE_SIZE, MATRIX_SIZE/TILE_SIZE);
    size_t shmem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    std::cout << "Launching " << NUM_MATRICES << " kernels with " << blocksPerGrid.x * blocksPerGrid.y << " blocks each with " << threadsPerBlock.x * threadsPerBlock.y << " threads\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_MATRICES; i++)
    {    
        // Copy matrices A and B to the device
        CHECK_CUDA(cudaMemcpy(d_A[i], h_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B[i], h_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        //matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock, shmem_size>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        matrixMultiplyKernelTiled<<<blocksPerGrid, threadsPerBlock, shmem_size>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        CHECK_CUDA(cudaGetLastError());

        // Copy results back to the host
        CHECK_CUDA(cudaMemcpy(h_C[i], d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time No Streams: " << diff.count() << " s\n";
    // Two other loops for epsilon test and cleanup to measure time comparison with streams fairly 
    // epsilon test
    double eps = 1.e-6; 
    for (int i = 0; i < NUM_MATRICES; i++) {
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++) {
            double abs_err = fabs(h_C[i][j] - (MATRIX_SIZE * 0.01f));
            double dot_length = MATRIX_SIZE;
            double abs_val = fabs(h_C[i][j]);
            double rel_err = abs_err / abs_val / dot_length;

            if (rel_err > eps) {
                printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    j, h_C[i][j], MATRIX_SIZE * 0.01f, eps);
            }
        }
    }
    // TODO: Cleanup
    for (int i = 0; i < NUM_MATRICES; i++)
    {
        free(h_A[i]);
        free(h_B[i]);
        free(h_C[i]);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
}

void matrixMultiplyWithStreams()
{
    // Host and device pointers
    float *h_A[NUM_MATRICES], *h_B[NUM_MATRICES], *h_C[NUM_MATRICES];
    float *d_A[NUM_MATRICES], *d_B[NUM_MATRICES], *d_C[NUM_MATRICES];

    cudaStream_t streams[NUM_STREAMS];
    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    // TODO: Allocate memory, initialize data, create streams and copy data asynchronously
    for (int i = 0; i < NUM_MATRICES; i++)
    {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        CHECK_CUDA(cudaMallocHost(&h_A[i], size));
        CHECK_CUDA(cudaMallocHost(&h_B[i], size));
        CHECK_CUDA(cudaMallocHost(&h_C[i], size));

        CHECK_CUDA(cudaMalloc(&d_A[i], size));
        CHECK_CUDA(cudaMalloc(&d_B[i], size));
        CHECK_CUDA(cudaMalloc(&d_C[i], size));

        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++)
        {
            h_A[i][j] = 1.0f;
            h_B[i][j] = 0.01f;
            h_C[i][j] = 0.0f;
        }
    }
    // TODO: Launch matrix multiplication kernel for each stream
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(MATRIX_SIZE / TILE_SIZE, MATRIX_SIZE / TILE_SIZE);
    size_t shmem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    std::cout << "Launching " << NUM_MATRICES << " kernels asynchronously..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_MATRICES; i++)
    {
        CHECK_CUDA(cudaMemcpyAsync(d_A[i], h_A[i], size, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_B[i], h_B[i], size, cudaMemcpyHostToDevice, streams[i]));

        // shared memory size?
        //matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock, shmem_size, streams[i]>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        matrixMultiplyKernelTiled<<<blocksPerGrid, threadsPerBlock, shmem_size, streams[i]>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        CHECK_CUDA(cudaGetLastError());
        // TODO: Copy results back to the host asynchronously
        CHECK_CUDA(cudaMemcpyAsync(h_C[i], d_C[i], size, cudaMemcpyDeviceToHost, streams[i]));
    }
    // TODO: Synchronize all streams
    // CPU waits until all streams have finished their work
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time With Streams: " << diff.count() << " s\n";
    // epsilon test
    double eps = 1.e-6; 
    for (int i = 0; i < NUM_MATRICES; i++) {
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++) {
            double abs_err = fabs(h_C[i][j] - (MATRIX_SIZE * 0.01f));
            double dot_length = MATRIX_SIZE;
            double abs_val = fabs(h_C[i][j]);
            double rel_err = abs_err / abs_val / dot_length;

            if (rel_err > eps) {
                printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    j, h_C[i][j], MATRIX_SIZE * 0.01f, eps);
            }
        }
    }
    // TODO: Cleanup
    for (int i = 0; i < NUM_MATRICES; i++)
    {
        // Destroy stream
        CHECK_CUDA(cudaStreamDestroy(streams[i]));

        // Free H
        CHECK_CUDA(cudaFreeHost(h_A[i]));
        CHECK_CUDA(cudaFreeHost(h_B[i]));
        CHECK_CUDA(cudaFreeHost(h_C[i]));

        // Free D
        CHECK_CUDA(cudaFree(d_A[i]));
        CHECK_CUDA(cudaFree(d_B[i]));
        CHECK_CUDA(cudaFree(d_C[i]));
    }
}

int main()
{   
    // Both tiling and using streams make the multiplication faster
    // The main currently tests the tiled matrix multiplation kernel, comparing stream and no stream. 
    // Changed the matrixMultiplyNoStreams function for fair time benchmark
    cudaFree(0);
    matrixMultiplyNoStreams();
    CHECK_CUDA(cudaDeviceSynchronize()); 

    cudaFree(0);
    matrixMultiplyWithStreams();
    CHECK_CUDA(cudaDeviceSynchronize()); 

    return EXIT_SUCCESS;
}
