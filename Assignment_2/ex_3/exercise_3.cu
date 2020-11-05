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
    if (argc != 4) {
        printf("Incorrect parameters!\nUsage: %s <block size> <num iterations> <num particles>\n", *argv);
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

    printf("Starting simulation on %d particles with %d iterations, GPU set to use block size %d...\n\n", num_particles, num_iterations, block_size);
    
    Particle *particles = (Particle*)malloc(num_particles * sizeof(Particle));
    Particle *d_res = (Particle*)malloc(num_particles * sizeof(Particle));

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
    printf("Simulating on the GPU...\n");

    auto start = std::chrono::system_clock::now();
    
    // Create, allocate and copy array to device
    Particle* d_particles = 0;
    cudaMalloc(&d_particles, num_particles * sizeof(Particle));
    cudaMemcpy(d_particles, particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);

    for(int i = 0; i < num_iterations; i++) {
        device_timestep<<<(num_particles + block_size - 1) / block_size,
            block_size>>>(d_particles, forces);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(d_res, d_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaFree(d_particles);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> host_time = end-start;

    printf("\tDone in %f s!\n\n", host_time.count());



    // ============= START COMPUTING ON HOST ============== //
    printf("Simulating on the CPU...\n");

    start = std::chrono::system_clock::now();

    for(int i = 0; i < num_iterations; i++) {
        host_timestep(particles, forces, num_particles);
    }

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> device_time = end-start;
    
    printf("\tDone in %f s!\n\n", device_time.count());
    printf("All done!\n");

    free(d_res);
    free(particles);
    
    return 0;
}