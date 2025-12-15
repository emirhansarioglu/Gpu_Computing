build_cpu: 
	clang++ -Wall -Wextra -Werror -O2 -o cpu cpu.cpp

build: gpu.cu
	nvcc -Werror all-warnings -lcurand -O2 -o gpu gpu.cu
