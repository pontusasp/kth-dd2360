#include <stdio.h>

#define N 64
#define TPB 32

// A scaling function to convert integers 0,1,...,N-1 to evenly spaced floats
__device__ float scale(int i, int n)
{
    return ((float)i) / (n - 1);
}

// Compute the distance between 2 points on a line.
__device__ float distance(float x1, float x2)
{
    return sqrt((x2 - x1) * (x2 - x1));
}

__global__ void distanceKernel(float* d_out, float ref, int len)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    const float x = scale(i, len);
    d_out[i] = distance(x, ref);

    printf("i = %2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i]);
}

int main()
{
    // Choose a reference value from which distances are measured.
    const float ref = 0.5;

    // Declare a pointer for an array of floats
    float* d_out = 0;

    // Allocate device memory for d_out
    cudaMalloc(&d_out, N * sizeof(float));

    // Launch kernel to compute, NOTE: it is advicable to replace N/TPB with
    // (N+TPB-1)/TPB to make sure the number of blocks needed is rounded up.
    distanceKernel<<<N/TPB, TPB>>>(d_out, ref, N);

    // Wait for device to finish before exiting
    cudaDeviceSynchronize();

    // Free the memory (Don't forget!!)
    cudaFree(d_out);

    return 0;
}