#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <chrono>
#include <time.h>

// This is a macro for checking the error variable.
#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error (%d): %s\n", __LINE__,clGetErrorString(err));

#define BLOCK_SIZE 256
#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 100

#define PARTICLE_INIT_SEED 8362

// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);

typedef struct
{
    float x;
    float y;
    float z;
} f3;

typedef struct
{
    float x;
    float y;
    float z;
    float w;
} f4;

typedef struct
{
    f3 pos;
    f3 vel;
} Particle;


const char *kernel_timestep_src = "\
typedef struct\n\
{\n\
    float x;\n\
    float y;\n\
    float z;\n\
} f3;\n\
\n\
typedef struct\n\
{\n\
    float x;\n\
    float y;\n\
    float z;\n\
    float w;\n\
} f4;\n\
\n\
typedef struct\n\
{\n\
    f3 pos;\n\
    f3 vel;\n\
} Particle;\n\
\n\
__kernel\n\
void device_timestep(__global Particle* p, __global f4* f,\
__global int* num_particles, __global int* num_iterations)\n\
{\n\
    int i = get_global_id(0);\n\
    if(i >= *num_particles) return;\n\
\n\
    float dt = (*f).w;\n\
\n\
    for (int k = 0; k < *num_iterations; k++) {\n\
        // Update velocity\n\
        p[i].vel.x = p[i].vel.x + (*f).x * dt;\n\
        p[i].vel.y = p[i].vel.y + (*f).y * dt;\n\
        p[i].vel.z = p[i].vel.z + (*f).z * dt;\n\
\n\
        // Update position\n\
        p[i].pos.x = p[i].pos.x + p[i].vel.x * dt;\n\
        p[i].pos.y = p[i].pos.y + p[i].vel.y * dt;\n\
        p[i].pos.z = p[i].pos.z + p[i].vel.z * dt;\n\
    }\n\
}\n\
\n\
";

void host_timestep(Particle* p, f4 f, const int num_particles)
{
    float dt = f.w;
    for(int k = 0; k < NUM_ITERATIONS; k++) {
        for(int i = 0; i < num_particles; i++) {
            // Update velocity
            p[i].vel.x = p[i].vel.x + f.x * dt;
            p[i].vel.y = p[i].vel.y + f.y * dt;
            p[i].vel.z = p[i].vel.z + f.z * dt;

            // Update position
            p[i].pos.x = p[i].pos.x + p[i].vel.x * dt;
            p[i].pos.y = p[i].pos.y + p[i].vel.y * dt;
            p[i].pos.z = p[i].pos.z + p[i].vel.z * dt;
        }
    }
}


int main(int argc, char **argv) {
    cl_platform_id * platforms; cl_uint     n_platform;

    // Find OpenCL Platforms
    cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
    err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);

    printf("Number of OpenCL Devices: %d\n", n_platform);
    for (int i = 0; i < n_platform; i++)
    {
        char name[1000];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(char) * 1000, name, NULL);
        printf("Device %d: %s\n", i, name);
    }
    printf("\nUsing Device 0.\n\n");

    // Find and sort devices
    cl_device_id *device_list; cl_uint n_devices;
    err = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &n_devices);CHK_ERROR(err);
    device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
    err = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);CHK_ERROR(err);

    // Create and initialize an OpenCL context
    cl_context context = clCreateContext( NULL, n_devices, device_list, NULL, NULL, &err);CHK_ERROR(err);

    // Create a command queue
    cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);CHK_ERROR(err); 

    /* =========== Insert your own code here =========== */
    cl_program program = clCreateProgramWithSource(context, 1,(const char **)&kernel_timestep_src, NULL, &err); CHK_ERROR(err);


    err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];
    clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len); 
    fprintf(stderr,"Build error: %s\n", buffer); exit(0);
    }

    cl_kernel kernel = clCreateKernel(program, "device_timestep", &err); CHK_ERROR(err);

    /* Initialize host memory/data */
    int array_size = NUM_PARTICLES * sizeof(Particle);
    Particle *particles, *res_dev;

    particles = (Particle*) malloc(array_size);
    res_dev = (Particle*) malloc(array_size);

    int array_length = NUM_PARTICLES;
    srand(PARTICLE_INIT_SEED);
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].pos.x = (float) (rand() % 10);
        particles[i].pos.y = (float) (rand() % 10);
        particles[i].pos.z = (float) (rand() % 10);

        particles[i].vel.x = (float) (rand() % 10);
        particles[i].vel.y = (float) (rand() % 10);
        particles[i].vel.z = (float) (rand() % 10);
    }

    f4 forces;
    forces.x = 0.0f;
    forces.y = 9.82f;
    forces.z = 0.0f;
    forces.w = 1.0f;

    printf("Computing simulation on the GPU...\n");
    auto start1 = std::chrono::system_clock::now();

    /* Allocated device data */
    cl_mem particles_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, array_size, NULL, &err);CHK_ERROR(err);
    cl_mem forces_dev = clCreateBuffer(context, CL_MEM_READ_ONLY,sizeof(f4), NULL, &err);CHK_ERROR(err);
    cl_mem num_iter_dev = clCreateBuffer(context, CL_MEM_READ_ONLY,sizeof(int), NULL, &err);CHK_ERROR(err);
    cl_mem num_parti_dev = clCreateBuffer(context, CL_MEM_READ_ONLY,sizeof(int), NULL, &err);CHK_ERROR(err);

    {
        int num_iterations_dev = NUM_ITERATIONS;
        int num_particles_dev = NUM_PARTICLES;

        /* Send command to transfer host data to device */
        err = clEnqueueWriteBuffer(cmd_queue, particles_dev, CL_TRUE, 0, array_size, particles, 0, NULL, NULL);CHK_ERROR(err);
        err = clEnqueueWriteBuffer(cmd_queue, forces_dev, CL_TRUE, 0, sizeof(f4), &forces, 0, NULL, NULL);CHK_ERROR(err);
        err = clEnqueueWriteBuffer(cmd_queue, num_iter_dev, CL_TRUE, 0, sizeof(int), &num_iterations_dev, 0, NULL, NULL);CHK_ERROR(err);
        err = clEnqueueWriteBuffer(cmd_queue, num_parti_dev, CL_TRUE, 0, sizeof(int), &num_particles_dev, 0, NULL, NULL);CHK_ERROR(err);
    }

    /* Set the three kernel arguments */
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &particles_dev);CHK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &forces_dev);CHK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &num_iter_dev);CHK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &num_parti_dev);CHK_ERROR(err);
    
    size_t local_work_size = BLOCK_SIZE; // number of items in group 
    size_t global_work_size = (NUM_PARTICLES + local_work_size - 1) / local_work_size * local_work_size; 

    const size_t workItems[1] = { global_work_size }; // Global amount of items
    const size_t workGroups[1] = { local_work_size }; // Items to process in each local group i.e.
                                            // workItems/workGroups = num of groups in each dimension
                                            // e.g. 4 / 1 = 4 groups (x * y * z = total)
    auto end1 = std::chrono::system_clock::now();
    std::chrono::duration<double> dev_time1 = (end1-start1) * 1000;

    auto start2 = std::chrono::system_clock::now();
    err = clEnqueueNDRangeKernel(
        cmd_queue,   // command_queue
        kernel,      // kernel
        1,           // work_dim
        NULL,        // global_work_offset
        workItems,   // global_work_size
        workGroups,  // local_work_size
        0,           // num_events_in_wait_list
        NULL,        // event_wait_list
        NULL         // event
    ); CHK_ERROR(err);

    err = clFinish(cmd_queue); CHK_ERROR(err);

    auto end2 = std::chrono::system_clock::now();
    std::chrono::duration<double> dev_time2 = (end2-start2) * 1000;

    auto start3 = std::chrono::system_clock::now();

    err = clEnqueueReadBuffer(cmd_queue, particles_dev, CL_TRUE, 0, array_size, res_dev, 0, NULL, NULL);CHK_ERROR(err);
    err = clFinish(cmd_queue); CHK_ERROR(err);

    auto end3 = std::chrono::system_clock::now();
    std::chrono::duration<double> dev_time3 = (end3-start3) * 1000;

    printf("\tComputation done in\t%f ms\n\tMemory transfer:\n\t\tTo Device:\t%f ms\n\t\tFrom Device:\t%f ms\n\tTotal time:\t\t%f ms\n\n",
    dev_time2.count(), dev_time1.count(), dev_time3.count(),
    dev_time1.count() + dev_time2.count() + dev_time3.count());



    printf("Computing simulation on the CPU...\n");
    start1 = std::chrono::system_clock::now();
    host_timestep(particles, forces, NUM_PARTICLES);

    end1 = std::chrono::system_clock::now();
    std::chrono::duration<double> host_time = (end1-start1) * 1000;
    printf("\tComputation done in\t%f ms\n\n", host_time.count());
    {
        float margin = NUM_PARTICLES / 10000000.0;
        printf("Checking the output for each implementation with margin %f...\n", margin);
        int failed = 0;
        for (int i = 0; i < NUM_PARTICLES; i++) {
            int localFail = 0;

            localFail += abs(particles[i].pos.x - res_dev[i].pos.x) > margin;
            localFail += abs(particles[i].pos.y - res_dev[i].pos.y) > margin;
            localFail += abs(particles[i].pos.z - res_dev[i].pos.z) > margin;

            localFail += abs(particles[i].vel.x - res_dev[i].vel.x) > margin;
            localFail += abs(particles[i].vel.y - res_dev[i].vel.y) > margin;
            localFail += abs(particles[i].vel.z - res_dev[i].vel.z) > margin;

            if (localFail) {
                failed++;
                printf("WARN: Mismatch at %d: Host(%f) != Kernel(%f)\n", i, particles[i].pos.x, res_dev[i].pos.x);
            }
        }
        if (failed) {
            printf("\tDone! %d mismatches found. (%2d%% failed)\n", failed, failed * 100 / NUM_PARTICLES);
        } else {
            printf("\tDone!\n\n");
        }
    }

    /* =============== END ================ */  

    // Finally, release all that we have allocated.
    err = clReleaseCommandQueue(cmd_queue);CHK_ERROR(err);
    err = clReleaseContext(context);CHK_ERROR(err);

    err = clReleaseMemObject(particles_dev);CHK_ERROR(err);
    err = clReleaseMemObject(forces_dev);CHK_ERROR(err);
    err = clReleaseMemObject(num_iter_dev);CHK_ERROR(err);
    err = clReleaseMemObject(num_parti_dev);CHK_ERROR(err);
    free(platforms);
    free(device_list);

    free(particles);
    free(res_dev);

    return 0;
}


// The source for this particular version is from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* clGetErrorString(int errorCode) {
    switch (errorCode) {
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
    case -69: return "CL_INVALID_PIPE_SIZE";
    case -70: return "CL_INVALID_DEVICE_QUEUE";
    case -71: return "CL_INVALID_SPEC_ID";
    case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
    case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
    case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
    case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
    case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
    case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
    case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
    case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
    case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
    case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
    case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
    case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
    case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
    case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
    case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
    case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
    case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
    case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
    default: return "CL_UNKNOWN_ERROR";
    }
}
