#example of brownian bridges

import math
import matplotlib.pyplot as plt
import numpy as np

def brownian_bridge(Traj, Pas, T, x, y):
    #time step
    DeltaT = T/Pas
    
    #Increments between each step; Standard Wiener Process
    col_zero = np.zeros((Traj, 1))
    rand_nums = math.sqrt(DeltaT) * np.random.normal(0, 1, size=(Traj, Pas))
    dW = np.append(col_zero, rand_nums, axis=1)
    W = np.cumsum(dW, axis=1)

    #Generate bridges
    rep = np.tile((DeltaT/T) * np.add(y-x, -1 * W[:, Pas]), (Pas, 1)).transpose()
    rep_mat = np.append(col_zero, rep, axis=1)

    #formulating the bridge
    bridge = W + x + np.cumsum(rep_mat, axis=1)

    return bridge
    
a = brownian_bridge(40, 100, 1, 100, 105)
DeltaT = 1/100

#graph the trajectories
time_intervals = np.arange(0, 1+DeltaT, DeltaT)
plt.plot(time_intervals, a.transpose())

#labels
plt.xlabel("Time")
plt.ylabel("W(t)")
plt.title("Brownian Bridges Example: From y = 100 to y = 105")
plt.show()