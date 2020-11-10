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
for it in range(0, len(iterList)):
       for block in range(0, len(blockSizes)):

              d_pi, d_time = re.findall(r"[-+]?\d*\.\d+|\d+", doublef[i])
              s_pi, s_time = re.findall(r"[-+]?\d*\.\d+|\d+", singlef[i])

              d_pi = float(d_pi)
              d_time = float(d_time)

              s_pi = float(s_pi)
              s_time = float(s_time)

              doubleTime[block, it] = d_time
              doubleError[block, it] = 100 * abs(np.pi - d_pi) / np.pi

              singleTime[block, it] = s_time
              singleError[block, it] = 100 * abs(np.pi - s_pi) / np.pi

              i = i + 1



def plot(mat, title, xlabel, ylabel, filename, logscale=False):

       for b in range(0, len(blockSizes)):
              plt.plot(iterList, mat[b])
       
       plt.legend(blockSizes, loc='upper left')
       plt.grid()
       plt.xlabel(xlabel)
       plt.ylabel(ylabel)
       if logscale:
              plt.yscale('log')
              plt.savefig(filename + "_log.png")
       else:
              plt.savefig(filename + ".png")
       plt.close()


plot(doubleTime, "Double Precision", "Iterations", "Time (s)", "ex_bonus_double")
plot(doubleTime, "Double Precision Logarithmic", "Iterations", "Time Logarithmic (s)", "ex_bonus_double", logscale=True)
plot(doubleError, "Double Precision", "Iterations", "Error (%)", "ex_bonus_double_error")

plot(doubleTime, "Single Precision", "Iterations", "Time (s)", "ex_bonus_single")
plot(doubleTime, "Single Precision Logarithmic", "Iterations", "Time Logarithmic (s)", "ex_bonus_single", logscale=True)
plot(doubleError, "Single Precision", "Iterations", "Error (%)", "ex_bonus_single_error")
