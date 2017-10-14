## Lorenz 96 model
def fnL96( x0, t0, tMax, dt, F = 8. ):
    nD = np.shape( x0 )[ 0 ]
    A = diags( [ 1, -1, 1, -1 ], [ 1-nD, -2, 1, nD-2 ] )
    
    def f( x, t ):
        return A.dot( x ) * np.roll( x, 1, axis = 0 ) - x + F
    
    # return odeint( f, x0, np.arange( t0, tMax, dt ) )
    return fnRK4( x0, t0, tMax, dt, f )