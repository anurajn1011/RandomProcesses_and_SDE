#example of brownian bridges

import math
import statistics
import matplotlib as plt

def brownian_bridge(Traj, Pas, T, x, y):
    DeltaT = T/Pas
    
