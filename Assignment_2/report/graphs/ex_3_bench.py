import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re

particleSizes = ["10k", "100k", "1000k", "10000k"]
blockSizes = ["16", "32", "64", "128", "256"]


#### GPU GRAPH ####

f = open('ex_3_gpu_output.dat')
raw = f.readlines()
f.close()

cntr = 0
for t in range(0, len(particleSizes)):
       deviceList = []

       for x in range(0, 5):
              deviceRun = 0
              hostRun = 0
              for i in range(0, 5):
                     deviceRun = deviceRun + float(raw[cntr])
                     print(cntr)
                     cntr = cntr + 1
              deviceList.append(deviceRun / 5)

       fig, ax = plt.subplots()

       ylevel = max(0, 2 * min(deviceList) - max(deviceList))

       ax.set(xlabel='Block Size', ylabel='Average time (s) (5 samples)',
              title=' CUDA GPU Simulation: 1k Iterations, ' + particleSizes[t] + ' Particles')
       ax.grid()
       ax.bar(blockSizes, deviceList)
       ax.set_ylim(bottom=ylevel)

       fig.savefig("ex_3_graph_gpu_" + particleSizes[t] + ".png")





#### CPU GRAPH ####

f = open('ex_3_cpu_output.dat')
raw = f.readlines()
f.close()

hostList = []

for t in range(0, len(particleSizes)):
       hostRun = 0
       for i in range(0, 5):
              hostRun = hostRun + float(raw[t * len(particleSizes) + i])
       hostList.append(str(round(hostRun / 5, 3)))

fig, ax = plt.subplots()

ax.set(xlabel='Particle count', ylabel='Average time (s) (5 samples)',
       title='CPU Simulation: 1k Iterations')
ax.grid()
ax.bar(particleSizes, hostList)

fig.savefig("ex_3_graph_cpu.png")

