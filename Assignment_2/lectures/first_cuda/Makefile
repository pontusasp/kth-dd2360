CC=g++
CC_FLAGS=
CC_LIBS=

NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

CUDA_FILES=kernel.cu
CPP_FILES=main.cpp

all:cuda

cuda:kernel.cu | bin
	$(NVCC) $(CUDA_FILES) -o ./bin/kernel

cpp:main.cpp | bin
	$(CC)  $(CPP_FILES) -o ./bin/main

bin:
	mkdir -p $@