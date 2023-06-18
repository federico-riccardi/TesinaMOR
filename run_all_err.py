#!/usr/bin/env python3
#%%

#import stuff
import argparse
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
from numpy import linalg as LA
import csv
import time

if not os.path.exists("CppToPython"):
    sys.exit("CppToPython does not exist, please run mesh.sh.") 

sys.path.append("CppToPython")
sys.path.insert(0, '../Utilities/')

import GeDiM4Py as gedim
import PINN_funct
import Greedy_funct
import FEM_funct

#### Parametri mesh
meshSize = 0.0005
order = 1

#### Parameters PINN
iterations= 8000
coeff = [3, 2, 4, 4] #trovato coi tentativi
n_points = 15000
delta = 0.1 #parametro per funzione cutoff

### PARAM GREEDY

M = 1000
N_max = 20
tol  = 1.e-8

## PARAM SOL
mu1_range = [0.1, 10.]
mu2_range = [-1., 1.]
P = np.array([mu1_range, mu2_range])

training_set = np.random.uniform(low=P[:, 0], high=P[:, 1], size=(M, P.shape[0]))
training_set_list = training_set.tolist()

###Creazione mesh e spazio discreto
os.chdir("CppToPython")
lib = gedim.ImportLibrary("./release/GeDiM4Py.so")

config = { 'GeometricTolerance': 1.0e-8 }
gedim.Initialize(config, lib)

domain = { 'SquareEdge': 1.0, 'VerticesBoundaryCondition': [1,0,1,1], 'EdgesBoundaryCondition': [2,3,1,1], 'DiscretizationType': 1, 'MeshCellsMaximumArea': meshSize }
[meshInfo, mesh] = gedim.CreateDomainSquare(domain, lib)

discreteSpace = { 'Order': order, 'Type': 1, 'BoundaryConditionsType': [1, 2, 3, 3] }
[problemData, dofs, strongs] = gedim.Discretize(discreteSpace, lib)

###Tempo addestramento PINN
mse_table, net = PINN_funct.PINN_funct(iterations, coeff, n_points, delta)
stiffness, advection, mass, weakTerm_down = FEM_funct.FEM_funct(problemData, lib)
B, stiffness_RB, advection_RB, weakTerm_down_RB = Greedy_funct.Greedy_funct(problemData, lib, M, tol, N_max)

### Plot PINN
pt_x = Variable(torch.from_numpy(np.array([dofs[0]]).T).float(), requires_grad=True)
pt_y = Variable(torch.from_numpy(np.array([dofs[1]]).T).float(), requires_grad=True)

err_2_PINN = []
err_inf_PINN = []
err_H1_PINN = []
err_2_Greedy = []
err_inf_Greedy = []
err_H1_Greedy = []

for mu in training_set_list:
    mu_1 = mu[0]
    mu_2 = mu[1]
    ###Calcolo valore PINN nodi mesh
    pt_u = net(pt_x, pt_y, mu_1*torch.ones((pt_x.shape[0],1)), mu_2*torch.ones((pt_y.shape[0],1)))
    solution_PINN = net(pt_x, pt_y, mu_1*torch.ones((pt_x.shape[0],1)), mu_2*torch.ones((pt_y.shape[0],1))).detach().numpy().T[0]

    ###Tempo Greedy
    u_RB = np.linalg.solve(mu_1*(stiffness_RB) + (advection_RB), mu_2*(weakTerm_down_RB))
    solution_Greedy = B @ u_RB

    ###Tempo FEM
    solution_FEM = gedim.LUSolver(mu_1*stiffness+advection, mu_2*weakTerm_down, lib)

    ##calcolo errore L^inf
    error_PINN = solution_FEM - solution_PINN
    l_inf_PINN = LA.norm(error_PINN, np.inf)
    l_2_PINN = np.sqrt(np.abs(error_PINN.T @ mass @ error_PINN))
    semi_h_1_PINN = np.sqrt(np.abs(error_PINN.T @ stiffness @ error_PINN))
    err_2_PINN.append(l_2_PINN)
    err_inf_PINN.append(l_inf_PINN)
    err_H1_PINN.append(semi_h_1_PINN)
    error_Greedy = solution_FEM - solution_Greedy
    l_inf_Greedy = LA.norm(error_Greedy, np.inf)
    l_2_Greedy = np.sqrt(np.abs(error_Greedy.T @ mass @ error_Greedy))
    semi_h_1_Greedy = np.sqrt(np.abs(error_Greedy.T @ stiffness @ error_Greedy))
    err_2_Greedy.append(l_2_Greedy)
    err_inf_Greedy.append(l_inf_Greedy)
    err_H1_Greedy.append(semi_h_1_Greedy)

print("mean of the L2 error PINN = {}".format(np.mean(np.array(err_2_PINN))))
print("std of the L2 error PINN = {}".format(np.std(np.array(err_2_PINN))))

print("mean of the Linf error PINN = {}".format(np.mean(np.array(err_inf_PINN))))
print("std of the Linf error PINN = {}".format(np.std(np.array(err_inf_PINN))))

print("mean of the H1 error PINN = {}".format(np.mean(np.array(err_H1_PINN))))
print("std of the H1 error PINN = {}".format(np.std(np.array(err_H1_PINN))))

print("mean of the L2 error Greedy = {}".format(np.mean(np.array(err_2_Greedy))))
print("std of the L2 error Greedy = {}".format(np.std(np.array(err_2_Greedy))))

print("mean of the Linf error Greedy = {}".format(np.mean(np.array(err_inf_Greedy))))
print("std of the Linf error Greedy = {}".format(np.std(np.array(err_inf_Greedy))))

print("mean of the H1 error Greedy = {}".format(np.mean(np.array(err_H1_Greedy))))
print("std of the H1 error Greedy = {}".format(np.std(np.array(err_H1_Greedy))))