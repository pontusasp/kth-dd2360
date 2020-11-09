#include <curand_kernel.h>
#include <curand.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PRECISION double


__global__ void monteCuda(PRECISION *counts, int num_iter, curandState *states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = idx;
    curand_init(seed, idx, 0, &states[idx]);

    int count = 0;
    PRECISION x, y, z;
    // Calculate PI following a Monte Carlo method
    for (int iter = 0; iter < num_iter; iter++)
    {
        // Generate random (X,Y) points
        x = curand_uniform(&states[idx]);
        y = curand_uniform(&states[idx]);
        z = sqrt((x*x) + (y*y));
        
        // Check if point is in unit circle
        if (z <= 1.0)
        {
            count++;
        }
    }

    counts[idx] = ((PRECISION)count / (PRECISION)num_iter);
}

// returns if values successfully read or not.
bool setValuesFromArgs(int argc, char **argv, unsigned int *block_size, unsigned int *num_threads, unsigned int *num_iter)
{
    if (argc != 4) {
        printf("Incorrect parameters!\nUsage: %s <block size> <num threads> <iterations per thread>\n", *argv);
        return false;
    }
    char *s;
    *block_size = strtoul(argv[1], &s, 10);
    *num_threads = strtoul(argv[2], &s, 10);
    *num_iter = strtoul(argv[3], &s, 10);
    return true;
}

int main(int argc, char* argv[])
{
    unsigned int block_size, num_threads, num_iter;
    if(!setValuesFromArgs(argc, argv, &block_size, &num_threads, &num_iter)) return 0;

    // Change num_threads to a multiple of block_size to prevent unexpected outcomes (memory size not matching up etc)
    num_threads = ((num_threads + block_size - 1) / block_size) * block_size; 

    PRECISION count = 0.0;
    PRECISION pi;

    PRECISION *counts = (PRECISION*)malloc(num_threads * sizeof(PRECISION));

    curandState *dev_random;
    cudaMalloc(&dev_random, num_threads*sizeof(curandState));

    PRECISION *p_counts = 0;
    cudaMalloc(&p_counts, num_threads * sizeof(PRECISION));
    monteCuda<<<(num_threads + block_size - 1) / block_size, block_size>>>(p_counts, num_iter, dev_random);
    
    cudaDeviceSynchronize();
    cudaMemcpy(counts, p_counts, num_threads * sizeof(PRECISION), cudaMemcpyDeviceToHost);

    // Estimate Pi and display the result

    for(int i = 0; i < num_threads; i++) {
        count += counts[i];
    }

    pi = (count / (PRECISION)num_threads) * 4.0;
    
    printf("The result is %f\n", pi);
    
    return 0;
}

