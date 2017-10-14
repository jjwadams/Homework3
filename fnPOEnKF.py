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