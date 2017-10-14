import numpy as np
import scipy as sp
import numpy.random as rnd
import matplotlib.pyplot as plt
import numpy.linalg as la
from numpy.random import multivariate_normal as mvn
from scipy.sparse import diags

# Function definitions

## RK4
def fnRK4( x0, t0, tMax, dt, f ):
    ti = t0;
    
    if np.size( x0.shape ) == 1:
        x0 = x0.reshape( x0.size, 1 )
    
    nStep = int( np.floor( ( tMax - t0 ) / dt ) )
    rgx = np.zeros( np.append( x0.shape, nStep+1 ) )
    rgx[ :, :, 0 ] = x0
    
    for i in range( nStep ):
        k1 = f( rgx[ :, :, i ], ti )
        k2 = f( rgx[ :, :, i ] + dt*k1/2., ti + dt/2. )
        k3 = f( rgx[ :, :, i ] + dt*k2/2., ti + dt/2. )
        k4 = f( rgx[ :, :, i ] + dt*k3, ti + dt )
        
        rgx[ :, :, i+1 ] = rgx[ :, :, i ] + dt * ( k1 + 2*k2 + 2*k3 + k4 ) / 6.
        ti = ti + dt
        
    return np.squeeze( rgx )

## Lorenz 96 model
def fnL96( x0, t0, tMax, dt, F = 8. ):
    nD = np.shape( x0 )[ 0 ]
    A = diags( [ 1, -1, 1, -1 ], [ 1-nD, -2, 1, nD-2 ] )
    
    def f( x, t ):
        return A.dot( x ) * np.roll( x, 1, axis = 0 ) - x + F
    
    # return odeint( f, x0, np.arange( t0, tMax, dt ) )
    return fnRK4( x0, t0, tMax, dt, f )

def fnL( n, r, p = 1 ):
    L = np.tile( np.arange( n ), ( n, 1 ) )
    L = np.exp( -np.abs( L - L.T )**p / r )
    return L
    
def fnLcirc( n, r, p = 2 ):
    L = np.tile( np.arange( n ), ( n, 1 ) )
    Lc = np.cos( 2*np.pi*L/n )
    Ls = np.sin( 2*np.pi*L/n )
    L = np.exp( -( np.abs( Lc - Lc.T )**p + np.abs( Ls - Ls.T )**p ) / r )
    return L

## Perturbed Observation EnKF
def fnPOEnKF( x, y, M, H, R, dtx = 0.01, dty = 0.1, a = 0, L = 1 ):
    # Inputs:
    #     x: ensemble up to time k, should be nD x nE
    #     y: data, should be nO x nSamp
    #     M: model function, should take a N x nE matrix (i.e. ensembles at 
    #        specific time) as input
    #     H: observation matrix, y = Hx so should be nO x N
    #     R: Prior covariance matrix, should be nO x nO
    #     a: inflation paramter. defaults to 0 for no inflation
    #     L: localization matrix. defaults to 1 for no localization
    
    # The input x is the initial ensemble, which has nE column vectors of 
    # length nD, which is the dimension of the problem.
    # The input y is the data, which has size nO (number of observations) by
    # nSamp (the number of samples).
    [ nD, nE ] = x.shape
    [ nO, nSamp ] = y.shape
    
    I = np.eye( nD )
    
    # Initialize output, with the first state being the input ensemble.
    rgx = np.zeros( ( nD, nE, nSamp+1 ) )
    rgx[ :, :, 0 ] = x
    
    # Initialize spread variable
    spread = np.zeros( nSamp ) 
    
    # number of iterations of model in x between data observations
    ndt = dty / dtx
    
    for k in range( nSamp ):
        # 1) Calculate the forecast ensemble:
        #        x_f^j = M x_k^j
        # 2) Calculate the forecast mean:
        #        mu_f^j = sum( x_f^j ) / nE
        # 3) Calculate the (possibly inflated) forecast perturbations:
        #        x_f^j = mu_f + sqrt( 1 + a ) ( x_f^j - mu_f )
        #        X_f = [ x_f^j - mu_f ]
        #    where [] indicates performing this observation over each column
        #    j of x_f^j, so this matrix should be nD x nE
        # 4) Calculate the (possibly localized) forecast covariance:
        #        P_f = L .* cov( X_f )
        # 5) Compute the Kalman Gain:
        #        K = P_f H^\top ( H P_f H^\top + R )^{-1}
        # 6) Compute the analysis mean:
        #        mu_a = mu_f + K( y - H mu_f )
        # 7) Perturb the observation:
        #        \tilde{y}^j = y^j + \xi^j,   \xi^j \sim N( 0, R )
        # 8) Generate the analysis ensemble:
        #        x_a^j = x_f^j + K[] \tilde{y}^j - H x_f^j ]
        
        xf = M( rgx[ :, :, k ], 0, ndt*dtx, dtx )[ :, :, -1 ]
        muf = np.mean( xf, axis = 1 )
        Xf = np.sqrt( 1+a ) * ( xf - np.tile( muf, ( nE, 1 ) ).T )
        xf = Xf + np.tile( muf, ( nE, 1 ) ).T
        Pf = L * np.cov( Xf )
        K = Pf.dot( la.solve( ( R + H.dot( Pf.dot( H.T ) ) ).T, H ).T )
        
        mua = muf + K.dot( y[ :, k ] - H.dot( muf ) )
        Pa = ( I - K.dot( H ) ).dot( Pf )
        
        yt = np.tile( y[ :, k ], ( nE, 1 ) ).T + mvn( np.zeros( nO ), R, nE ).T
        rgx[ :, :, k+1 ] = xf + K.dot( yt - H.dot( xf ) )
        
        spread[ k ] = np.sqrt( np.trace( Pa ) / nD )
    
    return rgx, spread

## Square root EnKF
def fnSqrtEnKF( x, y, M, H, R, dtx = 0.01, dty = 0.1, a = 0, L = 1 ):
    # Inputs:
    #     x: ensemble up to time k, should be nD x nE
    #     y: data, should be nO x nSamp
    #     M: model function, should take a N x nE matrix (i.e. ensembles at specific time) as input
    #     H: observation matrix, y = Hx so should be nO x N
    #     R: Prior covariance matrix, should be nO x nO
    #     a: inflation paramter. defaults to 0 for no inflation
    #     L: localization matrix. defaults to 1 for no localization
    
    [ nD, nE ] = x.shape
    [ nO, nSamp ] = y.shape
    
    I = np.eye( nD )
    
    rgx = np.zeros( ( nD, nE, nSamp+1 ) )
    rgx[ :, :, 0 ] = x
    
    # Initialize spread variable
    spread = np.zeros( nSamp ) 
    
    # number of iterations of model in x between data observations
    ndt = dty / dtx
    
    for k in range( nSamp ):
        xf = M( rgx[ :, :, k ], 0, ndt*dtx, dtx )[ :, :, -1 ]
        muf = np.mean( xf, axis = 1 )
        xf = np.sqrt( 1+a ) * ( xf - np.tile( muf, ( nE, 1 ) ).T ) + np.tile( muf, ( nE, 1 ) ).T 
        Xf = ( xf - np.tile( muf, ( nE, 1 ) ).T ) / np.sqrt( nE - 1 )
        Pf = L * Xf.dot( Xf.T )
        K = Pf.dot( la.solve( ( R + H.dot( Pf.dot( H.T ) ) ).T, H ).T )
        
        mua = muf + K.dot( y[ :, k ] - H.dot( muf ) )
        Pa = ( I - K.dot( H ) ).dot( Pf )
        
        V = H.dot( Xf ).T
        
        # print( 'V.shape: ', V.shape )
        # print( 'H.shape: ', H.shape )
        # print( 'Xf.shape: ', Xf.shape )
        Mtmp = V.dot( la.solve( R, V.T ) ) 
        Mtmp = ( Mtmp + Mtmp.T ) / 2
        [ D, U ] = la.eigh( Mtmp )
        # print( 'D.shape: ', D.shape )
        # D[ D < 0 ] = 0
        # print( 'D: ', D )
        Z = U.dot( np.diag( 1 / np.sqrt( 1+D ) ) ).dot( U.T )
        Xa = Xf.dot( Z )
        
        rgx[ :, :, k+1 ] = np.tile( mua, ( nE, 1 ) ).T + np.sqrt( nE - 1 ) * Xa
        
        spread[ k ] = np.sqrt( np.trace( Pa ) / nD )
        
    return rgx, spread



# To find an initial condition on (or close to) the L96 attractor do the following. 
# Solve the equations starting with an arbitrary initial condition for “a long time”, 
# e.g., start with a uniformly distributed initial condition and simulate for 100 
# dimensionless time units. This should bring you close to the attractor. Use the 
# final state of this simulation as your initial condition for the simulations below.

nD = 40
F = 8
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
tMax = 10.
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
dt = 0.01
nStep = int( np.floor( ( tMax - t0 ) / dt ) )

xEnsTot = fnL96( x0, t0, tMax, dt )

nE = 25
rgi = rnd.randint( 0, nStep+1, nE )

xEns = xEnsTot[ :, rgi ]

## Apply the EnKF to the data. Note that we have:
#    M: model is nonlinear Lorenz 96
#    H: observation matrix which gives every other sample, i.e. x1, x3, ...
#    R: prior covariance matrix 

nD = x0.size 

H = np.eye( nD )
H = H[ rg1, : ]
R = np.eye( nO )

xOutPO, spread = fnPOEnKF( xEns, xData, fnL96, H, R, a = 0.1, L = fnLcirc( nD, 1 ) )
xOutPOMean = np.mean( xOutPO, axis = 1 )
RMSEPO = np.sqrt( np.mean( ( xTrue - xOutPOMean[ :, 1: ] ) ** 2, 0 ) )

xOutSqrt, spreadSqrt = fnSqrtEnKF( xEns, xData, fnL96, H, R, a = 0.1, L = fnLcirc( nD, 1 ) )
xOutSqrtMean = np.mean( xOutSqrt, axis = 1 )
RMSESqrt = np.sqrt( np.mean( ( xTrue - xOutSqrtMean[ :, 1: ] ) ** 2, 0 ) )
