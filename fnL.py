def fnL( n, r, p = 1 ):
    L = np.tile( np.arange( n ), ( n, 1 ) )
    L = np.exp( -np.abs( L - L.T )**p / r )
    return L
    