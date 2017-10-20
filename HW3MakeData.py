import numpy as np
import scipy as sp
import sys
import numpy.random as rnd
import matplotlib.pyplot as plt
import numpy.linalg as la
import pickle
from numpy.random import multivariate_normal as mvn
from scipy.sparse import diags
from HW3fns import fnRK4, fnL96, fnL, fnLcirc, fnPOEnKF, fnSqrtEnKF


# To find an initial condition on (or close to) the L96 attractor do the following. 
# Solve the equations starting with an arbitrary initial condition for “a long time”, 
# e.g., start with a uniformly distributed initial condition and simulate for 100 
# dimensionless time units. This should bring you close to the attractor. Use the 
# final state of this simulation as your initial condition for the simulations below.

if "--nD" in sys.argv:
    nD = sys.argv[ sys.argv.index( "--nD" ) + 1 ]        
else:
    nD = 40
if "--nE" in sys.argv:
    nE = sys.argv[ sys.argv.index( "--nE" ) + 1 ]        
else:
    nE = 20
if "--F" in sys.argv:
    F = sys.argv[ sys.argv.index( "--F" ) + 1 ]
else:
    F = 8
if "--strSaveFile" in sys.argv:
    strSaveFile = sys.argv[ sys.argv.index( "--strSaveFile" ) + 1 ] 
else:
    strSaveFile = "HW3Data.dat"

t0 = 0.
tMax = 100.
dt = 0.01

nStep = int( np.floor( ( tMax - t0 ) / dt ) )

x0 = F + rnd.randn( nD )
# print( x0.shape )

rgx = fnL96( x0, t0, tMax, dt )
rgt = np.arange( t0, tMax+dt, dt )

plt.plot( rgt, rgx.T )

# Create a synthetic data set by simulating the L96 model starting from an initial 
# condition on the attractor (see above) for 10 dimensionless time units. Observe 
# every other state variable every δt=0.1 dimensionless time units, perturbed by 
# Gaussian noise with mean zero and covariance matrix R=I (here I is the 20×20 
# identity matrix).

x0 = rgx[ :, -1 ]
tMax = 100.
nStep = int( np.floor( ( tMax - t0 ) / dt ) )

rgxNew = fnL96( x0, t0, tMax, dt )

# We need to sparsify the data. The time step used for RK4 to create the L96 
# data was dt = 0.01, but we want our data to have dt = 0.1. We also want to
# observe every other state variable, starting with the 0th. Thus we generate
# index arrays for the observations that are every 2, starting from 0 to the 
# dimension nD, and for the time steps getting every 10th, starting from 10 and 

# going to the end of the array.
rg1 = np.arange( 0, nD, 2 )
rg2 = np.arange( 10, nStep+1, 10 )

# nO is the number of observations, nSamp is the number of samples in each of 
# the observations.
nO = rg1.size
nSamp = rg2.size

# Create data perturbed by N( 0, 1 ) noise.
xTrue = rgxNew[ :, rg2 ]
xData = xTrue[ rg1, : ] + rnd.randn( nO, nSamp )

rgt = np.arange( t0, tMax+dt, dt )

# plt.plot( rgt, rgxNew[ 2, : ], 'k' )
# plt.plot( rgt[ rg2 ], xData[ 1, : ], 'r' )

# Assimilate the synthetic data by an EnKF in perturbed obs. implementation. For 
# you initial ensemble, use randomly chosen states from a “long” simulation of L96 
# (1000 or more dimensionless time units). Do not use localization or inflation. 
# Plot RMSE and spread to check if the algorithm works, i.e., RMSE \aprox spread. 
# Disregard the first half of your similation/assimilation as “spin-up”. Increase 
# the ensemble size until the algorithm works. How large must your ensemble be?

tMax = 1e3
nStep = int( np.floor( ( tMax - t0 ) / dt ) )

xEnsTot = fnL96( x0, t0, tMax, dt )

rgi = rnd.randint( 1, nStep+2, nE )

xEns = xEnsTot[ :, rgi ]

H = np.eye( nD )
H = H[ rg1, : ]
R = np.eye( nO )

dictSave = { "x0": x0, "xTrue": xTrue, "xData": xData, "xEnsTot": xEnsTot, 
             "F": F, "nE": nE, "nD": nD, "H": H, "R": R, "xEns": xEns }

with open( strSaveFile, 'wb' ) as fiSave:
    pickle.dump( dictSave, fiSave )
