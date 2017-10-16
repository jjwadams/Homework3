import numpy as np
import Homework3

for a in np.arange( 0, 2, 0.05 ):
	for r in 10 ** np.arange( -1, 2, 0.1 ):
		Homework3.main( a, r )
