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



#### Parameters PINN
meshSize = 0.01
order = 2
iterations= 10000
#coeff = [10, 10, 500, 500]
coeff = [5, 3, 50, 100]
n_points = 200
delta = 0.1 #parametro per funzione cutoff

### PARAM GREEDY

M = 1000
N_max = 20
tol  = 1.e-8

## PARAM SOL
mu_1 = 1.
mu_2 = 1.

start_time_PINN_offline = time.time()
mse_table, net = PINN_funct.PINN_funct(iterations, coeff, n_points, delta)
end_time_PINN_offline = time.time()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x = np.arange(0,1,0.02)
y = np.arange(0,1,0.02)
ms_x, ms_y = np.meshgrid(x,y)
x = np.ravel(ms_x).reshape(-1,1)
y = np.ravel(ms_y).reshape(-1,1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True)

start_time_PINN_online = time.time()
pt_u = net(pt_x, pt_y, mu_1*torch.ones((pt_x.shape[0],1)), mu_2*torch.ones((pt_y.shape[0],1)))
start_time_PINN_online = time.time()
#u = pt_u.data.cpu().numpy()
u = pt_u.numpy(force=True)
ms_u = u.reshape(ms_x.shape)
surf = ax.plot_surface(ms_x, ms_y, ms_u, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.savefig('fotoPINN.png')


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

start_time_Greedy_offline = time.time()
B, stiffness_RB, advection_RB, weakTerm_down_RB = Greedy_funct.Greedy_funct(problemData, lib, M, tol, N_max)
end_time_Greedy_offline = time.time()

start_time_Greedy_online = time.time()
u_RB = np.linalg.solve(mu_1*(stiffness_RB) + (advection_RB), mu_2*(weakTerm_down_RB))
solution_RB = B @ u_RB
end_time_Greedy_online = time.time()
gedim.PlotSolution(mesh, dofs, strongs, solution_RB, np.zeros(problemData['NumberStrongs']), title='Solution_RB')

## FEM
stiffness, advection, weakTerm_down = FEM_funct.FEM_funct(problemData, lib)
solution = gedim.LUSolver(mu_1*stiffness+advection, mu_2*weakTerm_down, lib)
gedim.PlotSolution(mesh, dofs, strongs, solution, np.zeros(problemData['NumberStrongs']), title='Solution_FEM')