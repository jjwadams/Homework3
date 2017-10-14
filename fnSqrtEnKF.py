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