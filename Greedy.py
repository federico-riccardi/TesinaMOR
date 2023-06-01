#!/usr/bin/env python3
#%%

#import stuff
import numpy as np
import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
import warnings
import csv
from scipy.sparse.linalg import splu
import time

#figuwarnings.filterwarnings('ignore')

if not os.path.exists("CppToPython"):
    sys.exit("CppToPython does not exist, please run mesh.sh.") 

sys.path.append("CppToPython")
sys.path.insert(0, '../Utilities/')

import GeDiM4Py as gedim
os.chdir("CppToPython")
lib = gedim.ImportLibrary("./release/GeDiM4Py.so")

config = { 'GeometricTolerance': 1.0e-8 }
gedim.Initialize(config, lib)

order = 1
meshSize = 0.001

domain = { 'SquareEdge': 1.0, 'VerticesBoundaryCondition': [1,0,1,1], 'EdgesBoundaryCondition': [2,3,1,1], 'DiscretizationType': 1, 'MeshCellsMaximumArea': meshSize }
[meshInfo, mesh] = gedim.CreateDomainSquare(domain, lib)

discreteSpace = { 'Order': order, 'Type': 1, 'BoundaryConditionsType': [1, 2, 3, 3] }
[problemData, dofs, strongs] = gedim.Discretize(discreteSpace, lib)

gedim.PlotMesh(mesh)
gedim.PlotDofs(mesh, dofs, strongs)

def Poisson_a(numPoints, points):
	values = np.ones(numPoints)
	return values.ctypes.data

def Poisson_b(numPoints, points):
	values = np.zeros((2, numPoints))
	matPoints = gedim.make_nd_matrix(points, (3, numPoints), np.double)
	values[0,:] = matPoints[1,:]*(1.-matPoints[1,:])
	return values.ctypes.data

def Poisson_weakTerm_down(numPoints, points):
	values = np.ones(numPoints)
	return values.ctypes.data

[stiffness, stiffnessStrong] = gedim.AssembleStiffnessMatrix(Poisson_a, problemData, lib)

[advection, advectionStrong] = gedim.AssembleAdvectionMatrix(Poisson_b, problemData, lib)

weakTerm_down = gedim.AssembleWeakTerm(Poisson_weakTerm_down, 2, problemData, lib)

solution = gedim.LUSolver(stiffness+advection, weakTerm_down, lib)
gedim.PlotSolution(mesh, dofs, strongs, solution, np.zeros(problemData['NumberStrongs']))


### define the training set
M = 100
mu1_range = [0.1, 10.]
mu2_range = [-1., 1.]
P = np.array([mu1_range, mu2_range])

training_set = np.random.uniform(low=P[:, 0], high=P[:, 1], size=(M, P.shape[0]))
N_max = 20
tol  = 1.e-2
X = stiffness

def normX(v, X):
	return np.sqrt(np.transpose(v) @ X @ v)

def ProjectSystem(AQH, fQH, B):
    AQN = []
    fQN = []
    for AH in AQH:
        AQN.append(np.copy(np.transpose(B) @ AH @ B))
    for fH in fQH:
        fQN.append(np.copy(np.transpose(B) @ fH))
    return [AQN, fQN]

def Solve_reduced_order(AQN, fQN, thetaA_mu, thetaF_mu):
    A = thetaA_mu[0] * AQN[0]
    f = thetaF_mu[0] * fQN[0]
    for i in range(1, len(AQN)):
        A += thetaA_mu[i] * AQN[i]
    for i in range(1, len(fQN)):
        f += thetaF_mu[i] * fQN[i]
    return np.linalg.solve(A, f)

def GramSchmidt(V, u, X):
    z = u
    if np.size(V) > 0:
        z = u - V @ (np.transpose(V) @ (X @ u))
    return z / normX(z, X)

def ErrorEstimate():
     return

def InfSupConstant(mu):
     return

def greedy(X,N_max, tol):
    N = 0
    basis_functions = []
    B = np.empty((0,0))
    delta_N = tol + 1.
    training_set_list = training_set.tolist()
    initial_muN = np.random.choice(len(training_set_list) - 1, 1)[0]
    mu_N = training_set_list.pop(initial_muN)
    invX = splu(X)

    print('Perfom greedy algorithm...')
    while len(training_set_list) > 0 and N < N_max and delta_N > tol:
        N += 1
        snapshot = gedim.LUsolver(mu_N[0]*X+advection,mu_N[1]*weakTerm_down,lib)
        basis_function = GramSchmidt(B, snapshot, X)
        basis_functions.append(np.copy(basis_function))
        B = np.transpose(np.array(basis_functions))
        BX = np.transpose(B) @ X @ B

        [AQN, fQN] = ProjectSystem(X, weakTerm_down, B) # applica il cambio base
        [C_11, d_11, d_12, E_11, E_12, E_22] = OfflineResidual(B, invX)

        counter = 0
        mu_selected_index = -1
        max_deltaN = -1.
        for mu in training_set_list:
            solN_mu = Solve_reduced_order(AQN, fQN, thetaA(mu), thetaF(mu))
            betaN_mu = InfSupConstant(mu)
            deltaN_mu = ErrorEstimate(Cq1q2, dq1q2, Eq1q2, thetaA(mu), thetaF(mu), solN_mu, betaN_mu) / normX(solN_mu, BX)
	    
            if deltaN_mu > max_deltaN:
                max_deltaN = deltaN_mu
                mu_selected_index = counter

            counter += 1

        if mu_selected_index == -1:
            raise Exception('ERROR, parameter not found')

        muN = training_set_list.pop(mu_selected_index)
        deltaN = max_deltaN

    return [N, np.transpose(np.array(basis_functions))]
