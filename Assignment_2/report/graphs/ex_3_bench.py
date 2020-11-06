import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re

blockSizes = ["16", "32", "64", "128", "256"]

deviceList = []

f = open('ex_3_output.dat')
raw = f.readlines()
f.close()

for x in range(0, 5):
       deviceRun = 0
       hostRun = 0
       for i in range(0, 5):
              ds,hs = re.findall(r"[-+]?\d*\.\d+|\d+", raw[x*5 + i])
              deviceRun = deviceRun + float(ds)
              hostRun = hostRun + float(hs)
       deviceList.append(deviceRun / 5)
hostValue = hostRun / (5 * 5)


fig, ax = plt.subplots()

ax.set(xlabel='Block Size', ylabel='Average time (s) (5 samples)',
       title='CUDA Simulation: 10k Iterations, 100k Particles\n(Avg CPU time: ' + str(round(hostValue, 3)) + ' s)')
ax.grid()
ax.bar(blockSizes, deviceList)
ax.set_ylim(bottom=0.35)

fig.savefig("ex_3_graph.png")

