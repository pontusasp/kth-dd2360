CC=g++
CC_FLAGS=
CC_LIBS=

NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

CUDA_FILES=exercise_2.cu
CPP_FILES=

all: $(CUDA_FILES) $(CPP_FILES) | bin
	$(NVCC) $(CUDA_FILES) $(CPP_FILES) -o ./bin/out

bin:
	mkdir -p $@