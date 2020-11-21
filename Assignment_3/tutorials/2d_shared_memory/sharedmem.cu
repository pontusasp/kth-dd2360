#include <stdio.h>

#define BDIMX 32
#define BDIMY 16


dim3 block (BDIMX, BDIMY);
dim3 grid (1,1);

__global__ void setRowReadRow(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x] ;
}

__global__ void setColReadCol(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMX][BDIMY];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadCol(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from 2D thread index to linear memory
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // shared memory store operation 
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[icol][irow];
}

__global__ void setRowReadColDyn(int *out)
{
    // dynamic shared memory
    extern __shared__ int tile[];
    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // convert back to smem idx to access the transposed element
    unsigned int col_idx = icol * blockDim.x + irow;

    // shared memory store operation
    tile[idx] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[col_idx]; 
}

int main()
{
    int *c;
    c = (int*)malloc(BDIMX * BDIMY * sizeof(int));

    int *d_C;
    cudaMalloc(&d_C, BDIMX * BDIMY * sizeof(int));

    setRowReadRow<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_C, BDIMX * BDIMY * sizeof(int), cudaMemcpyDeviceToHost);

    for (int y = 0; y < BDIMY; y++)
    {
        printf("[ ");
        for (int x = 0; x < BDIMX; x++)
            printf("% 4d ", c[y * BDIMX + x]);
        printf("]\n");
    }

    cudaFree(d_C);
    free(c);

    return 0;
}