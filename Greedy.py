#!/usr/bin/env python3
#%%

#import stuff
import numpy as np
import shutil
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
dir = "greedyResults"
if not os.path.exists(dir):
    os.makedirs(dir)

complete_dir = os.getcwd()+"/"+dir

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
invX = splu(X)

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

def GramSchmidt(V, u):
    z = u
    if np.size(V) > 0:
        z = u - V @ (np.transpose(V) @ (X @ u))
    return z / normX(z, X)

def OfflineResidual(B):
     C_11 = weakTerm_down.T @ (invX.solve(weakTerm_down)) # Xu = f_1 -> u = X^{-1} f_1 = invX.solve(f_1)

     d_11 = B.T @ stiffness.T @ (invX.solve(weakTerm_down))
     d_12 = B.T @ advection.T @ (invX.solve(weakTerm_down))

     E_11 = B.T @ stiffness.T @ (invX.solve(stiffness @ B))
     E_12 = B.T @ stiffness.T @ (invX.solve(advection @ B))
     E_22 = B.T @ advection.T @ (invX.solve(advection @ B))
     return [C_11, d_11, d_12, E_11, E_12, E_22]
     

def ErrorEstimate(mu, solN_mu, off_res, beta_mu):
    pre_mult = [mu[1]**2, -2*mu[0]*mu[1]*solN_mu.T, -2*mu[1]*solN_mu.T, mu[0]**2*solN_mu.T, 2*mu[0]*solN_mu.T, solN_mu.T]
    post_mult = [1., 1., 1., solN_mu, solN_mu, solN_mu]
    error = 0.0
    for i in range(len(off_res)):
        error += pre_mult[i] @ off_res[i] @ post_mult[i]
    return np.sqrt(error)/beta_mu

eigs, vecs = scipy.linalg.eig(stiffness.todense(), mass.todense())
min_eig = np.min(eigs.real)
C_omega =  1 / np.sqrt(min_eig) #costante di Poincaré
def InfSupConstant(mu):
    return mu[0]/(1+C_omega**2)

def greedy(N_max, tol):
    N = 0
    basis_functions = []
    B = np.empty((0,0))
    delta_N = tol + 1.
    training_set_list = training_set.tolist()
    initial_muN = np.random.choice(len(training_set_list) - 1, 1)[0]
    mu_N = training_set_list.pop(initial_muN)
    beta_mu = InfSupConstant(mu_N)

    print('Perfom greedy algorithm...')
    while len(training_set_list) > 0 and N < N_max and delta_N > tol:
        N += 1
        snapshot = gedim.LUsolver(mu_N[0]*X+advection,mu_N[1]*weakTerm_down,lib)
        basis_function = GramSchmidt(B, snapshot, X)
        basis_functions.append(np.copy(basis_function))
        B = np.transpose(np.array(basis_functions))
        BX = np.transpose(B) @ X @ B
        stiffness_RB = B.T @ stiffness @ B
        advection_RB = B.T @ advection @ B
        weakTerm_down_RB = B.T @ weakTerm_down
        off_res = OfflineResidual(B)
        counter = 0
        mu_selected_index = -1
        max_deltaN = -1.
        for mu in training_set_list:
            solN_mu =  np.linalg.solve(mu[0]*stiffness_RB + advection_RB, mu[1]*weakTerm_down_RB)
            deltaN_mu = ErrorEstimate(mu, solN_mu, off_res, beta_mu) / normX(solN_mu, BX)

            if deltaN_mu > max_deltaN:
                max_deltaN = deltaN_mu
                mu_selected_index = counter

            counter += 1

        if mu_selected_index == -1:
            raise Exception('ERROR, parameter not found')

        mu_N = training_set_list.pop(mu_selected_index)
        delta_N = max_deltaN

    return [N, np.transpose(np.array(basis_functions))]

pic_dir = "Images"
shutil.copytree(pic_dir, complete_dir, dirs_exist_ok=True)
