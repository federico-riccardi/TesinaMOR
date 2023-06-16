#!/usr/bin/env python3
#%%
import sys
from scipy.sparse.linalg import splu
import numpy as np
sys.path.append("CppToPython")
sys.path.insert(0, '../Utilities/')
import GeDiM4Py as gedim
import scipy

def FEM_funct(problemData, lib):
        
    def Poisson_a(numPoints, points):
        values = np.ones(numPoints)
        return values.ctypes.data

    def Poisson_b(numPoints, points):
        values = np.zeros((2, numPoints))
        matPoints = gedim.make_nd_matrix(points, (3, numPoints), np.double)
        values[0,:] = matPoints[1,:]*(1.-matPoints[1,:]) #campo di velocità [y(1-y), 0]
        return values.ctypes.data

    def Poisson_c(numPoints, points):
        values = np.ones(numPoints)
        return values.ctypes.data

    def Poisson_weakTerm_down(numPoints, points):
        values = np.ones(numPoints)
        return values.ctypes.data

    [stiffness, stiffnessStrong] = gedim.AssembleStiffnessMatrix(Poisson_a, problemData, lib)

    [advection, advectionStrong] = gedim.AssembleAdvectionMatrix(Poisson_b, problemData, lib)

    [mass, massStrong] = gedim.AssembleReactionMatrix(Poisson_c, problemData, lib) # serve solo per calcolare la costante di Poincaré   q

    weakTerm_down = gedim.AssembleWeakTerm(Poisson_weakTerm_down, 2, problemData, lib)

    return stiffness, advection, mass, weakTerm_down