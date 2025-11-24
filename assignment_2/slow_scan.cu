#include <chrono>
#include <curand.h>
#include <iostream>
#include <stdlib.h>

#include "helper.cu"

void sequential_scan(size_t size, float *in_h, float *out_h) {
  out_h[0] = in_h[0];
  out_h[1] = in_h[1];
  for (auto i = 2; i < size; i += 2) {
    float real_prev = out_h[i - 2];
    float real_cur = in_h[i];
    float im_prev = out_h[i - 1];
    float im_cur = in_h[i + 1];

    out_h[i] = real_prev * real_cur - im_prev * im_cur;
    out_h[i + 1] = real_prev * im_cur + real_cur * im_prev;
  }
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
  sequential_scan(size, in_h, out_h);
  auto end = std::chrono::system_clock::now();

  std::cout << "First 3 entries of In Vec:" << std::endl;
  for (int32_t i = 0; i < 5 * 2; i += 2)
    std::cout << in_h[i] << "," << in_h[i + 1] << std::endl;
  std::cout << "First 3 entries of Out Vec:" << std::endl;
  for (int32_t i = 0; i < 5 * 2; i += 2)
    std::cout << out_h[i] << " + " << out_h[i + 1] << std::endl;

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

  CUDA_CALL(cudaFree(in_d));
  CUDA_CALL(cudaFree(out_d));
  free(in_h);
  free(out_h);
  return EXIT_SUCCESS;
}
