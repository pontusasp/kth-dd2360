CL_INCLUDE_PATH=/usr/local/cuda-11.1/targets/x86_64-linux/include
CL_LIBRARY_PATH=/usr/local/cuda-11.1/targets/x86_64-linux/lib
CL_LIBRARIES=OpenCL

FILES=exercise_3.c
OUT_FILE=bin/ex_3

main: $(FILES) | bin
	gcc $(FILES) -o $(OUT_FILE) -L$(CL_LIBRARY_PATH) -I$(CL_INCLUDE_PATH) -l$(CL_LIBRARIES)

bin:
	mkdir -p bin