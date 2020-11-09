import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re

particleSizes = ["10000", "100000", "1000000", "10000000"]
blockSizes = ["16", "32", "64", "128", "256"]


#### GPU GRAPH ####

f = open('ex_3_gpu_output.dat')
raw = f.readlines()
f.close()

for t in range(0, len(particleSizes)):
       deviceList = []

       for x in range(0, 5):
              deviceRun = 0
              hostRun = 0
              for i in range(0, 5):
                     deviceRun = deviceRun + float(raw[t*len(particleSizes) + x*5 + i])
              deviceList.append(deviceRun / 5)

       fig, ax = plt.subplots()

       ax.set(xlabel='Block Size', ylabel='Average time (s) (5 samples)',
              title=' CUDA GPU Simulation: 1k Iterations, ' + str(int(int(particleSizes[t])/1000)) + 'k Particles')
       ax.grid()
       ax.bar(blockSizes, deviceList)

       fig.savefig("ex_3_graph_gpu_" + str(int(int(particleSizes[t])/1000)) + "k.png")





#### CPU GRAPH ####

f = open('ex_3_cpu_output.dat')
raw = f.readlines()
f.close()

hostList = []

for t in range(0, len(particleSizes)):
       hostRun = 0
       for i in range(0, 5):
              hostRun = hostRun + float(raw[t * len(particleSizes) + i])
       hostList.append(hostRun / 5)

fig, ax = plt.subplots()

ax.set(xlabel='Particle count', ylabel='Average time (s) (5 samples)',
       title='CPU Simulation: 1k Iterations, ' + str(int(int(particleSizes[t])/1000)) + 'k Particles')
ax.grid()
ax.bar(particleSizes, hostList)

fig.savefig("ex_3_graph_cpu_" + str(int(int(particleSizes[t])/1000)) + "k.png")

