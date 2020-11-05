#include <stdio.h>
#include <random>
#include <cmath>

#define ARRAY_SIZE 10000
#define BLOCK_SIZE 256

__global__ void device_saxpy(float* x, float* y, const float a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    y[i] = a * x[i] + y[i];
}

void host_saxpy(float x[], float y[], const float a)
{
    for(int i = 0; i < ARRAY_SIZE; i++) {
        y[i] = a * x[i] + y[i];
    }
}


int main()
{
    std::default_random_engine rdmGen;
    std::uniform_real_distribution<float> dist(0.0, 5.0);

    const float a = 1.0;

    float* x = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float* y = (float*)malloc(ARRAY_SIZE * sizeof(float));

    for (int i = 0; i < ARRAY_SIZE; i++) {
        x[i] = dist(rdmGen);
        y[i] = dist(rdmGen);
    }

    // Create, allocate and copy array to device
    float* d_x = 0;
    float* d_y = 0;

    cudaMalloc(&d_x, ARRAY_SIZE * sizeof(float));
    cudaMalloc(&d_y, ARRAY_SIZE * sizeof(float));

    cudaMemcpy(d_x, x, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);


    // call host func
    printf("Computing SAXPY on the CPU... ");
    host_saxpy(x, y, a);
    printf("Done!\n\n");


    // call device func
    printf("Computing SAXPY on the GPU... ");
    device_saxpy<<<(ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE>>>(d_x, d_y, a);
    cudaDeviceSynchronize();
    printf("Done!\n\n");


    // Get results from device and store in d_res
    cudaMemcpy(x, d_y, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // compare d_res to y
    printf("Comparing the output for each implementation... ");

    bool correct = true;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        if(abs(x[i] - y[i]) > 0.0001) { // x is device result, y is host result
            correct = false;
            break;
        }
    }

    if(correct) printf("Correct!\n");
    else printf("Incorrect!\n");

    // Free up resources
    free(y);
    free(x);

    cudaFree(d_y);
    cudaFree(d_x);

    return 0;
}