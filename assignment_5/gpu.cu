#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#define N 128                // Grid size X
#define M 128                // Grid size Y
#define ITERATIONS 100000    // Number of iterations
#define DIFFUSION_FACTOR 0.5 // Diffusion factor
#define CELL_SIZE 0.01       // Cell size for the simulation
#define BLOCK_SIZE 16        // Thread block size (16x16 = 256 threads)

// Error checking call from previous exercises
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void initializeGrid(float *grid, int n, int m)
{
    for (int y = 0; y < m; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            // Initialize one quadrant to a high temp
            // and the rest to 0.
            if (y > m / 2 && x > n / 2)
            {
                grid[y * n + x] = 100.0f; // Temp in corner
            }
            else
            {
                grid[y * n + x] = 0.0f; // Temp in the rest
            }
        }
    }
}

/**
 * CUDA kernel for 2D heat diffusion simulation using shared memory.
 * 
 * Each thread block loads a tile of data into shared memory (including halos),
 * then computes the stencil operation for interior points only (matching CPU behavior).
 * Boundary cells are never updated and maintain their initial values as boundary conditions?
 * 
 * @param curr Current temperature grid
 * @param next Next temperature grid (output)
 * @param n Grid width
 * @param m Grid height
 * @param dt Time step
 * @param dx2 Cell size squared in x direction
 * @param dy2 Cell size squared in y direction
 * @param alpha Precomputed coefficient (DIFFUSION_FACTOR * dt)
 */
__global__ void heatDiffusionKernel(const float *curr, float *next, int n, int m, 
                                     float dt, float dx2, float dy2, float alpha)
{
    // Shared memory tile with halo cells (BLOCK_SIZE + 2 for each dimension)
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    // Global coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Local coordinates in shared memory (offset by 1 for halo)
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // Load center data into shared memory
    if (x < n && y < m)
    {
        tile[ty][tx] = curr[y * m + x];
    }
    
    // Load halo regions (boundary cells)
    // Left halo
    if (threadIdx.x == 0 && x > 0)
    {
        tile[ty][0] = curr[y * m + (x - 1)];
    }
    
    // Right halo
    if (threadIdx.x == blockDim.x - 1 && x < n - 1)
    {
        tile[ty][tx + 1] = curr[y * m + (x + 1)];
    }
    
    // Top halo
    if (threadIdx.y == 0 && y > 0)
    {
        tile[0][tx] = curr[(y - 1) * m + x];
    }
    
    // Bottom halo
    if (threadIdx.y == blockDim.y - 1 && y < m - 1)
    {
        tile[ty + 1][tx] = curr[(y + 1) * m + x];
    }
    
    // Wait for all threads to finish loading shared memory
    __syncthreads();
    
    // Compute heat diffusion for interior points only (1 to n-2, 1 to m-2)
    // This matches the CPU behavior which only updates interior cells
    if (x > 0 && x < n - 1 && y > 0 && y < m - 1)
    {
        float center = tile[ty][tx];
        float left = tile[ty][tx - 1];
        float right = tile[ty][tx + 1];
        float above = tile[ty + 1][tx];
        float below = tile[ty - 1][tx];
        
        // Apply 5-point stencil heat equation
        next[y * m + x] = center + alpha * 
                          ((left - 2.0f * center + right) / dy2 +
                           (above - 2.0f * center + below) / dx2);
    }
}

/**
 * Runs the heat simulation on GPU using CUDA.
 * 
 * @param h_grid Host grid to initialize GPU memory
 * @param n Grid width
 * @param m Grid height
 * @param iterations Number of simulation iterations
 * @param dt Time step
 * @return Final grid on host
 */
float *heatSimulationGPU(float *h_grid, int n, int m, int iterations, float dt)
{
    size_t size = n * m * sizeof(float);
    
    // Allocate device memory for current and next grids
    float *d_curr, *d_next;
    CUDA_CHECK(cudaMalloc(&d_curr, size));
    CUDA_CHECK(cudaMalloc(&d_next, size));
    
    // Copy initial grid to BOTH buffers (critical for boundary conditions)
    CUDA_CHECK(cudaMemcpy(d_curr, h_grid, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next, h_grid, size, cudaMemcpyHostToDevice));
    
    // Precompute constants
    float dx2 = CELL_SIZE * CELL_SIZE;
    float dy2 = CELL_SIZE * CELL_SIZE;
    float alpha = DIFFUSION_FACTOR * dt;
    
    // Configure kernel launch parameters
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Run iterations
    for (int iter = 0; iter < iterations; ++iter)
    {
        // Launch kernel
        heatDiffusionKernel<<<gridDim, blockDim>>>(d_curr, d_next, n, m, dt, dx2, dy2, alpha);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        
        // Swap pointers for double buffering
        float *temp = d_curr;
        d_curr = d_next;
        d_next = temp;
    }
    
    // Wait for all kernels to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy final result back to host 
    CUDA_CHECK(cudaMemcpy(h_grid, d_curr, size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_curr));
    CUDA_CHECK(cudaFree(d_next));
    
    return h_grid;
}

int main()
{
    // Allocate memory for the grid on host
    float *h_grid = (float *)malloc(N * M * sizeof(float));
    
    // Check for allocation failure
    if (h_grid == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Initialize the grid
    initializeGrid(h_grid, N, M);
    
    float dx2 = CELL_SIZE * CELL_SIZE;
    float dy2 = CELL_SIZE * CELL_SIZE;
    float dt = dx2 * dy2 / (2.0 * DIFFUSION_FACTOR * (dx2 + dy2));
    
    // Run the heat simulation on GPU
    float *final_grid = heatSimulationGPU(h_grid, N, M, ITERATIONS, dt);
    
    // Print a small section of the final grid for verification
    std::cout << "\nFinal grid values (top-left corner):" << std::endl;
    for (int y = 0; y < 16; ++y)
    {
        for (int x = 0; x < 16; ++x)
        {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) 
                      << final_grid[y * N + x] << " ";
        }
        std::cout << std::endl;
    }
    
    // Free allocated memory
    free(h_grid);
    
    return 0;
}