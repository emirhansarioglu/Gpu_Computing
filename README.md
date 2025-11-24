# GPU Computing

This repository contains materials and assignments for the **GPU Computing** lecture at **TU Berlin**. The course focuses on parallel programming and GPU acceleration techniques.

---

## Assignment 1: Parallelizing a Basic Linear Algebra Program in CUDA

The goal of this assignment is to implement a parallel version of a basic linear algebra program using **CUDA**.

## Assignment 2: Implementing a parallel **inclusive scan** (prefix sum) for an array of complex numbers based on the Kogge-Stone algorithm

(see `fast_scan.cu` for implementation without optimiziation)

Implementing the optimizations in the following order: (see `optimized_scan.cu`)
1) Reducing Divergence
2) Shared Memory Utilization
3) Thread Coarsening
4) Memory Coalescing 

---

## How to run assignments

To compile and run any assignment:

1. Navigate to the corresponding assignment folder, e.g., `assignment1`.
2. Build the executable using the provided Makefile:

```bash
make build