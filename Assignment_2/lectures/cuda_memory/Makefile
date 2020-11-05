CC=g++
CC_FLAGS=
CC_LIBS=

NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

CUDA_FILES=kernel-1.cu
CPP_FILES=main.cpp

all:cuda

cuda: kernel-1.cu main.cpp | bin
	$(NVCC) $(CUDA_FILES) $(CPP_FILES) -o ./bin/kernel

bin:
	mkdir -p $@