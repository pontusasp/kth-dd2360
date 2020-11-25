import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re

'''

64 128 256 512 1024 2048 4096
CPU-matmul: 0.046000 0.320000 2.440000 21.174000 144.353000 1552.880000 13544.008000
GPU-cuBLAS-matmul: 0.034000 0.028000 0.039000 0.104000 0.541000 3.512000 22.552000
GPU-matmul-(global-memory): 0.177000 0.188000 0.443000 3.181000 26.166000 191.489000 1394.062000 
GPU-matmul-(shared-memory): 0.008000 0.014000 0.054000 0.346000 2.829000 20.628000 151.366000 

'''

sizes = [64, 128, 256, 512, 1024, 2048, 4096]
cpu_matmul = [0.046000, 0.320000, 2.440000, 21.174000, 144.353000, 1552.880000, 13544.008000]
gpu_cublas = [0.034000, 0.028000, 0.039000, 0.104000, 0.541000, 3.512000, 22.552000]
gpu_global = [0.177000, 0.188000, 0.443000, 3.181000, 26.166000, 191.489000, 1394.062000]
gpu_shared = [0.008000, 0.014000, 0.054000, 0.346000, 2.829000, 20.628000, 151.366000]


labels = sizes

x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, gpu_cublas, width, label='GPU-cuBLAS-matmul')
rects2 = ax.bar(x, gpu_global, width, label='GPU-matmul (global memory)')
rects3 = ax.bar(x + width, gpu_shared, width, label='GPU-matmul (shared memory)')
rects4 = ax.bar(x + 2 * width, cpu_matmul, width, label='CPU-matmul')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Log Time (ms)')
ax.set_title('Time measurements GPU')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yscale('log', basey=2)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{} ms'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

plt.savefig("blas_graph.png")
plt.close()
