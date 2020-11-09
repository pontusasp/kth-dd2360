import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re

iterList = ["1000", "10000", "100000", "1000000"]
blockSizes = ["16", "32", "64", "128", "256"]

f = open('ex_bonus_double.dat')
doublef = f.readlines()
f.close()

f = open('ex_bonus_single.dat')
singlef = f.readlines()
f.close()

doubleTime = np.empty((len(blockSizes), len(iterList)), float)
doubleError = np.empty((len(blockSizes), len(iterList)), float)

singleTime = np.empty((len(blockSizes), len(iterList)), float)
singleError = np.empty((len(blockSizes), len(iterList)), float)

i = 0
for iter in range(0, len(iterList)):
       for block in range(0, len(blockSizes)):

              d_pi, d_time = re.findall(r"[-+]?\d*\.\d+|\d+", doublef[i])
              s_pi, s_time = re.findall(r"[-+]?\d*\.\d+|\d+", singlef[i])

              doubleTime[block, iter] = d_time
              doubleError[block, iter] = abs(np.pi - d_pi)

              singleTime[block, iter] = s_time
              singleError[block, iter] = abs(np.pi - s_pi)

              i = i + 1



'''

fig, ax = plt.subplots()

ax.set(xlabel='Particle count', ylabel='Average time (s) (5 samples)',
       title='CPU Simulation: 1k Iterations')
ax.grid()
ax.bar(particleSizes, hostList)

fig.savefig("ex_3_graph_cpu.png")

'''