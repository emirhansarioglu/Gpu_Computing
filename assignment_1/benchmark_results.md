# CUDA Kernel Benchmark Results


| Threads per Block | Blocks per Grid | Tasks Solved / Execution Time (s) |
|-------------------:|----------------:|-----------------------------------:|
| 1  | 1  | 0.0016(does not solve the task) |
| 4  | 4  | 0.0017(does not solve the task) |
| 32 | 32 | 0.0035(does not solve the task) |
| 128 | 256 | 0.1620 |
| **256** | **128** | **0.1602** |
| 256 | 256 | 0.1611 |
| 512  | 64 | 0.1658 |
| 512 | 256 | 0.1648 |
| 512 | 512 | 0.1638 |
| 1024 | 32 | 0.1743 |
| 1024 | 256 | 0.1738 |
| 1024 | 1024 | 0.1731 |