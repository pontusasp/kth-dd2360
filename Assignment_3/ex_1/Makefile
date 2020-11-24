CC=g++
CC_FLAGS=
CC_LIBS=

NVCC=nvcc
NVCC_FLAGS=-arch=sm_61
NVCC_LIBS=

CUDA_FILES=hw3_ex1.cu
CPP_FILES=
OUT_FILE=hw3_ex1.out

all: $(CUDA_FILES) $(CPP_FILES)
	$(NVCC) $(CUDA_FILES) $(CPP_FILES) $(NVCC_FLAGS) -o ./$(OUT_FILE)

bin:
	mkdir -p $@

debug: $(CUDA_FILES) $(CPP_FILES)
	$(NVCC) $(CUDA_FILES) $(CPP_FILES) -lineinfo -Xcompiler -rdynamic $(NVCC_FLAGS) -o ./$(OUT_FILE)