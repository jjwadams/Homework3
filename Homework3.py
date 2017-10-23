import numpy as np
import scipy as sp
import sys
import numpy.random as rnd
import matplotlib.pyplot as plt
import numpy.linalg as la
import pickle
import re
from numpy.random import multivariate_normal as mvn
from scipy.sparse import diags
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from HW3fns import fnL96, fnL, fnLcirc, fnPOEnKF, fnSqrtEnKF

def test( a = 0, r = 1 ):
    strTextFile = "test.txt"
    with open( strTextFile, 'a' ) as fiText:
        fiText.write( "Finished a = %f, r = %f\n"%( a, r ) ) 
    

def main( a = 0, r = 1, strSaveDir = "./outFiles", strLoadFile = "HW3Data.dat" ):
    # if "--a" in sys.argv:
    #     a = sys.argv[ sys.argv.index( "--a" ) + 1 ]
    # else:
    #     a = 0
    # if "--r" in sys.argv:
    #     r = sys.argv[ sys.argv.index( "--r" ) + 1 ]
    # else:
    #     r = 1
    # if "--strLoadFile" in sys.argv:
    #     strLoadFile = sys.argv[ sys.argv.index( "--strLoadFile" ) + 1 ] 
    # else:
    #     strLoadFile = "HW3Data.dat"

    with open( strLoadFile, 'rb' ) as fiLoad:
        dictLoad = pickle.load( fiLoad )

        F = dictLoad[ "F" ]
        xData = dictLoad[ "xData" ]
        xEnsTot = dictLoad[ "xEnsTot" ]
        xEns = dictLoad[ "xEns" ]
        xTrue = dictLoad[ "xTrue" ]
        H = dictLoad[ "H" ]
        R = dictLoad[ "R" ]

        [ nD, nETot ] = xEnsTot.shape
        nE = xEns.shape[ 1 ]

        rgi = rnd.randint( 1, nETot+1, nE )
        xEns = xEnsTot[ :, rgi ]

    print( "Starting a = %s, r = %s"%( a, r ) )

    xOutPO, spreadPO = fnPOEnKF( xEns, xData, fnL96, H, R, a = a, L = fnLcirc( nD, r ) )
    xOutPOMean = np.mean( xOutPO, axis = 1 )
    RMSEPO = np.sqrt( np.mean( ( xTrue - xOutPOMean[ :, 1: ] ) ** 2, 0 ) )

    xOutSqrt, spreadSqrt = fnSqrtEnKF( xEns, xData, fnL96, H, R, a = a, L = fnLcirc( nD, r ) )
    xOutSqrtMean = np.mean( xOutSqrt, axis = 1 )
    RMSESqrt = np.sqrt( np.mean( ( xTrue - xOutSqrtMean[ :, 1: ] ) ** 2, 0 ) )

    strSaveFile = "{}/a{}r{}.dat"%( strSaveDir, a, r )
    dictSave = { "spreadPO": spreadPO, "RMSEPO": RMSEPO, "spreadSqrt": spreadSqrt, 
                 "RMSESqrt": RMSESqrt }

    with open( strSaveFile, 'wb' ) as fiSave:
        pickle.dump( dictSave, fiSave )

    strTextFile = "{}/test.txt"%( strSaveDir )
    with open( strTextFile, 'a' ) as fiText:
        fiText.write( "Finished a = %f, r = %f\n"%( a, r ) ) 

def fnGetAll( strSaveDir = "./Outfiles", strRegEx = r"a([\d.\-\+]+)r([\d.\-\+]+)" ):
    regex = re.compile( strRegEx )
    rga = []
    rgr = []
    rgSpread = []
    rgRMSE = []
    

    rgPath = Path( strSaveDir ).glob( '**/*.dat' )
    for path in rgPath:
        strPath = str( path )

        rgMatch = regex.findall( strPath )
        for match in rgMatch:
            # print( "appending %s to rga" % ( match[0] ) )
            rga.append( float( match[ 0 ] ) )
            rgr.append( float( match[ 1 ][ :-1 ] ) )

        with open( strPath, 'rb' ) as fiPath:
            dictPath = pickle.load( fiPath )
            rgSpread.append( [ dictPath[ "spreadPO" ], dictPath[ "spreadSqrt" ] ] )
            rgRMSE.append( [ dictPath[ "RMSEPO" ], dictPath[ "RMSESqrt" ] ] )


    rga = np.array( rga )
    rgr = np.array( rgr )
    rgSpread = np.array( rgSpread )
    rgRMSE = np.array( rgRMSE )

    iSort = np.lexsort( ( rgr, rga ) )
    rga = rga[ iSort ]
    rgr = rgr[ iSort ]
    rgSpread = rgSpread[ iSort ]
    rgRMSE = rgRMSE[ iSort ]

    return rga, rgr, rgSpread, rgRMSE

def fnPlotVals( rgVals, strTitle, stryLabel = r'$\alpha$ from 0 to 1.95 by 0.5',
				strxLabel = r'$r$ from $10^{-1}$ to $10^{1.9}$ with the exponent by 0.1' ):
	fig = plt.figure()
	plt.imshow( rgVals )
	plt.ylabel( stryLabel )
	plt.xlabel( strxLabel )
	plt.title( strTitle )
	plt.colorbar()
	plt.show()
