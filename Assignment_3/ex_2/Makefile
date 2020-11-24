CC=g++
CC_FLAGS=
CC_LIBS=

NVCC=nvcc
NVCC_FLAGS=-arch=sm_61
NVCC_LIBS=

CUDA_FILES=exercise_2a.cu
CPP_FILES=
OUT_FILE=exercise_2a.out

CUDA_FILES_B=exercise_2b.cu
OUT_FILE_B=exercise_2b.out

all: $(CUDA_FILES) $(CPP_FILES) | bin
	$(NVCC) $(CUDA_FILES) $(CPP_FILES) $(NVCC_FLAGS) -o ./bin/$(OUT_FILE)

bin:
	mkdir -p $@

debug: $(CUDA_FILES) $(CPP_FILES) | bin
	$(NVCC) $(CUDA_FILES) $(CPP_FILES) -lineinfo -Xcompiler -rdynamic $(NVCC_FLAGS) -o ./bin/$(OUT_FILE)

2b: $(CUDA_FILES) $(CPP_FILES) | bin
	$(NVCC) $(CUDA_FILES_B) $(CPP_FILES) $(NVCC_FLAGS) -o ./bin/$(OUT_FILE_B)