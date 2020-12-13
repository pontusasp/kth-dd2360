import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import csv

def readGpu(filename, convert=True):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        gpu = []
        ctr = 0
        for row in reader:
            gpu.append([])
            for elem in row:
                if convert:
                    gpu[ctr].append(float(elem.strip()))
                else:
                    gpu[ctr].append(elem.strip())
            ctr = ctr + 1
    return np.array(gpu).T.tolist()

def readCpu(filename, convert=True):
    with open(filename, newline='') as f:
        cpu = []
        ctr = 0
        for row in f.readlines():
            if convert:
                cpu.append(float(row.strip()))
            else:
                cpu.append(row.strip())
    return cpu

bl_sizes = ["16", "32", "64", "128", "256"]
pa_sizes = ["10", "100", "1000", "10000", "100000"]

gpu_16 = readGpu("gpu_16.dat")
gpu_32 = readGpu("gpu_32.dat")
gpu_64 = readGpu("gpu_64.dat")
gpu_128 = readGpu("gpu_128.dat")
gpu_256 = readGpu("gpu_256.dat")

gpu = [gpu_16, gpu_32, gpu_64, gpu_128, gpu_256]

cpu = readCpu("cpu.dat")

labels = pa_sizes

fig, ax = plt.subplots()

for i in range(0, len(bl_sizes)):
    plt.plot(pa_sizes, gpu[i][3], label = "GPU, block size " + bl_sizes[i])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time (ms) (Log scale)')
ax.set_xlabel('Number of Particles (Log scale)')
ax.set_title('Time measurements GPU')
ax.set_xticks(pa_sizes)
ax.set_xticklabels(labels)
ax.set_yscale('log', base=10)
plt.grid()
ax.legend()


plt.savefig("ex_3_gpu.png")

ax.set_title('Time measurements CPU & GPU')
plt.plot(pa_sizes, cpu, label = "CPU")
ax.legend()

plt.savefig("ex_3.png")

plt.close()