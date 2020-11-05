#include<stdio.h>

__global__ void helloWorld()
{
    printf("Hello World! My threadId is %d\n", threadIdx.x);
}

int main()
{
    helloWorld<<<1, 256>>>();
    cudaDeviceSynchronize();
    return 0;
}