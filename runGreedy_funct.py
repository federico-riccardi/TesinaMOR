#!/usr/bin/env python3
#%%

#sys.path.append("PINN_funct.py")
import Greedy_funct
import os
import sys

M = 100
N_max = 20
tol  = 1.e-7
import numpy as np

if not os.path.exists("CppToPython"):
    sys.exit("CppToPython does not exist, please run mesh.sh.") 

sys.path.append("CppToPython")
sys.path.insert(0, '../Utilities/')

import GeDiM4Py as gedim
dir = "greedyResults"

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

B, stiffness_RB, advection_RB, weakTerm_down_RB = Greedy_funct.Greedy_funct(problemData, lib, M, tol, N_max)

mu_1 = 1.
mu_2 = 1.
u_RB = np.linalg.solve(mu_1*(stiffness_RB) + (advection_RB), mu_2*(weakTerm_down_RB))
solution_RB = B @ u_RB
gedim.PlotSolution(mesh, dofs, strongs, solution_RB, np.zeros(problemData['NumberStrongs']), title='Solution_RB')

