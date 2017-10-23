import numpy as np
import Homework3

def main( rga = np.linspace( 0, 1, 11 ), rgr = np.linspace( 0, 3, 4 ), 
          strSaveDir = "./outFiles", strLoadFile = "HW3Data.dat" ):

    for a in rga:
        for r in rgr:
            Homework3.main( a, r, strSaveDir, strLoadFile )
