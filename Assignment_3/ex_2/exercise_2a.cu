#include <stdio.h>
#include <random>
#include <chrono>

typedef struct
{
    float3 pos = {0.0};
    float3 vel = {0.0};
} Particle;

// Timestep for particles, f contains force to be applied to p.vel in x,y,z and w is time derivative
__global__ void device_timestep(Particle* p, float4 f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float dt = f.w;
    
    // Update velocity
    p[i].vel.x = p[i].vel.x + f.x * dt;
    p[i].vel.y = p[i].vel.y + f.y * dt;
    p[i].vel.z = p[i].vel.z + f.z * dt;

    // Update position
    p[i].pos.x = p[i].pos.x + p[i].vel.x * dt;
    p[i].pos.y = p[i].pos.y + p[i].vel.y * dt;
    p[i].pos.z = p[i].pos.z + p[i].vel.z * dt;
}

void host_timestep(Particle* p, float4 f, const int num_particles)
{
    float dt = f.w;
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

// returns if values successfully read or not.
bool setValuesFromArgs(int argc, char **argv, unsigned int *block_size, unsigned int *num_iterations, unsigned int *num_particles)
{
    if (argc < 4) {
        printf("Incorrect parameters!\nUsage: %s <block size> <num iterations>\
         <num particles> [1 extra arg for gpu benchmark output, 2 for cpu]\n", *argv);
        return false;
    }
    char *s;
    *block_size = strtoul(argv[1], &s, 10);
    *num_iterations = strtoul(argv[2], &s, 10);
    *num_particles = strtoul(argv[3], &s, 10);
    return true;
}

int main(int argc, char **argv)
{
    unsigned int block_size, num_iterations, num_particles;
    if(!setValuesFromArgs(argc, argv, &block_size, &num_iterations, &num_particles)) return 0;

    // Change num_threads to a multiple of block_size to prevent unexpected outcomes (memory size not matching up etc)
    num_particles = ((num_particles + block_size - 1) / block_size) * block_size; 

    bool gpuBench = argc == 5;
    bool cpuBench = argc == 6;

    if (!(gpuBench || cpuBench))
        printf("Starting simulation on %d particles with %d iterations, GPU set to use block size %d...\n\n", num_particles, num_iterations, block_size);
    
    Particle *particles = (Particle*)malloc(num_particles * sizeof(Particle));
    Particle *d_res;
    cudaMallocHost((void**)&d_res, num_particles * sizeof(Particle), cudaHostAllocDefault);

    std::default_random_engine rdmGen;
    std::uniform_real_distribution<float> posDist(-100.0, 100.0);
    std::uniform_real_distribution<float> velDist(-10.0, 10.0);

    for(int i = 0; i < num_particles; i++) {
        particles[i].pos.x = posDist(rdmGen);
        particles[i].pos.y = posDist(rdmGen);
        particles[i].pos.z = posDist(rdmGen);

        particles[i].vel.x = velDist(rdmGen);
        particles[i].vel.y = velDist(rdmGen);
        particles[i].vel.z = velDist(rdmGen);
    }

    float4 forces = {
         0.0,   // x
         0.0,   // y
        -9.82,  // z
         1.0    // dt
    };



    // ============= START COMPUTING ON DEVICE ============== //
    if (!cpuBench) {
        if (!gpuBench)
            printf("Simulating on the GPU...\n");

        auto start1 = std::chrono::system_clock::now();
        
        // Create, allocate and copy array to device
        Particle* d_particles = 0;
        cudaMalloc(&d_particles, num_particles * sizeof(Particle));

        for(int i = 0; i < num_iterations; i++) {
            cudaMemcpy(d_particles, particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);

            device_timestep<<<(num_particles + block_size - 1) / block_size,
                block_size>>>(d_particles, forces);
                
            cudaDeviceSynchronize();
            cudaMemcpy(d_res, d_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_particles);

        auto end1 = std::chrono::system_clock::now();
        std::chrono::duration<double> device_time = end1-start1;
        
        if (!gpuBench)
            printf("\tDone in %f s!\n\n", device_time.count());
        else
            printf("%f\n", device_time.count());
    }


    if (!gpuBench) {
        // ============= START COMPUTING ON HOST ============== //
        if (!cpuBench)
            printf("Simulating on the CPU...\n");

        auto start2 = std::chrono::system_clock::now();

        for(int i = 0; i < num_iterations; i++) {
            host_timestep(particles, forces, num_particles);
        }

        auto end2 = std::chrono::system_clock::now();
        std::chrono::duration<double> host_time = end2-start2;

        if (!cpuBench) {
            printf("\tDone in %f s!\n\n", host_time.count());
            printf("All done!\n");
        }
        else printf("%f\n", host_time.count());
    }

    cudaFree(d_res);
    free(particles);
    
    return 0;
}