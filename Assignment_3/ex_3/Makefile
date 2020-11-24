CC=g++
CC_FLAGS=
CC_LIBS=

NVCC=nvcc
NVCC_FLAGS=-arch=sm_61 --default-stream per-thread
NVCC_LIBS=

CUDA_FILES=exercise_3.cu
CPP_FILES=
OUT_FILE=exercise_3.out

all: $(CUDA_FILES) $(CPP_FILES) | bin
	$(NVCC) $(CUDA_FILES) $(CPP_FILES) $(NVCC_FLAGS) -o ./bin/$(OUT_FILE)

bin:
	mkdir -p $@

debug: $(CUDA_FILES) $(CPP_FILES) | bin
	$(NVCC) $(CUDA_FILES) $(CPP_FILES) -lineinfo -Xcompiler -rdynamic $(NVCC_FLAGS) -o ./bin/$(OUT_FILE)