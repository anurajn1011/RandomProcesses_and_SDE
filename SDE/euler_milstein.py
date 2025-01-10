#simulation of a SDE via Euler and Milstein discretization schemes
import math
import numpy as np

def euler_scheme(NbTraj, NbStep, X0, a, b, T = 1):
    #time step
    DeltaT = 1/NbStep

    #compute gaussian variable
    DeltaW = math.sqrt(DeltaT) * np.random.normal(0, 1, size=(NbTraj, NbStep))

    #define matrix of steps
    Traj = np.zeros((NbTraj, NbStep + 1))
    Traj[:, [0]] = X0 * np.ones((NbTraj, 1))

    #simulate all time steps and update matrix
    for i in range(0, NbStep):
        Traj[:, i+1] = Traj[:, i] * (1 + a + DeltaT + b * DeltaW[:, i])
    
    return Traj

def milstein_scheme(NbTraj, NbStep, X0, a, b, T = 1):
    #time step
    DeltaT = 1/NbStep

    #compute gaussian variable
    DeltaW = math.sqrt(DeltaT) * np.random.normal(0, 1, size=(NbTraj, NbStep))
    DeltaW2 = np.multiply(DeltaW, DeltaW) #for the milstein approximation

    #define matrix of steps
    Traj = np.zeros((NbTraj, NbStep + 1))
    Traj[:, [0]] = X0 * np.ones((NbTraj, 1))

    #simulate all time steps and update matrix
    for i in range(0, NbStep):
        Traj[:, [i+1]] = Traj[:, [i]] * (1 + a + DeltaT + b * DeltaW[:, [i]]) + (b**2/2) * (DeltaW2[:, [i]] - DeltaT * np.ones((NbTraj, 1)))
    
    return Traj

result1 = euler_scheme(1000, 10, 100, 0.05, 0.2)
result2 = milstein_scheme(1000, 10, 100, 0.05, 0.2)

print(result2)
