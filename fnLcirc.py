def fnLcirc( n, r, p = 2 ):
    L = np.tile( np.arange( n ), ( n, 1 ) )
    Lc = np.cos( 2*np.pi*L/n )
    Ls = np.sin( 2*np.pi*L/n )
    L = np.exp( -( np.abs( Lc - Lc.T )**p + np.abs( Ls - Ls.T )**p ) / r )
    return L
