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