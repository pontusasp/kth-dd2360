import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re

f = open('bandwidthTest-shmoo.txt')
raw = f.readlines()

####################
## Host to Device ##
####################

htdx = []
htdy = []

for i in range(10, 91):
       x,y = re.findall(r"[-+]?\d*\.\d+|\d+", raw[i])
       htdx.append(int(x))
       htdy.append(float(y))


fig, ax = plt.subplots()

ax.set(xlabel='Transfer Size (Bytes)', ylabel='Bandwidth(GB/s)',
       title='Host to Device Bandwidth')
ax.grid()
ax.plot(htdx, htdy)
ax.set_xscale('log')
plt.yticks(np.arange(0, max(htdy)+1, 1.0))

fig.savefig("host-to-device.png")

####################
## Device to Host ##
####################

dthx = []
dthy = []

for i in range(96, 177):
       x,y = re.findall(r"[-+]?\d*\.\d+|\d+", raw[i])
       dthx.append(int(x))
       dthy.append(float(y))


fig, ax = plt.subplots()

ax.set(xlabel='Transfer Size (Bytes)', ylabel='Bandwidth(GB/s)',
       title='Device to Host Bandwidth')
ax.grid()
ax.plot(dthx, dthy)
ax.set_xscale('log')
plt.yticks(np.arange(0, max(dthy)+1, 1.0))

fig.savefig("device-to-host.png")


######################
## Device to Device ##
######################

dtdx = []
dtdy = []

for i in range(182, 236):
       x,y = re.findall(r"[-+]?\d*\.\d+|\d+", raw[i])
       dtdx.append(int(x))
       dtdy.append(float(y))


fig, ax = plt.subplots()

ax.set(xlabel='Transfer Size (Bytes)', ylabel='Bandwidth(GB/s)',
       title='Device to Device Bandwidth')
ax.grid()
ax.plot(dtdx, dtdy)
ax.set_xscale('log')

fig.savefig("device-to-device.png")

