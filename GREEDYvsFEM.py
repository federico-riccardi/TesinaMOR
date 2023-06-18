#!/usr/bin/env python3

import yaml
from yaml.loader import SafeLoader
import subprocess
import FEM_funct
import sys
import os
import Greedy_funct
import numpy as np
import torch
from torch.autograd import Variable
from numpy import linalg as LA
import csv

## Importazione GeDiM4Py, se necessario
if not os.path.exists("CppToPython"):
    sys.exit("CppToPython does not exist, please run mesh.sh.") 

sys.path.append("CppToPython")
sys.path.insert(0, '../Utilities/')

import GeDiM4Py as gedim
this_dir = os.getcwd()
os.chdir("CppToPython")
lib = gedim.ImportLibrary("./release/GeDiM4Py.so")

## Creazione mesh e spazio discreto
config = { 'GeometricTolerance': 1.0e-8 }
gedim.Initialize(config, lib)
order = 1
meshSize = 0.001

domain = { 'SquareEdge': 1.0, 'VerticesBoundaryCondition': [1,0,1,1], 'EdgesBoundaryCondition': [2,3,1,1], 'DiscretizationType': 1, 'MeshCellsMaximumArea': meshSize }
[meshInfo, mesh] = gedim.CreateDomainSquare(domain, lib)

discreteSpace = { 'Order': order, 'Type': 1, 'BoundaryConditionsType': [1, 2, 3, 3] }
[problemData, dofs, strongs] = gedim.Discretize(discreteSpace, lib)

## Creazioni matrici sistema FEM (posso farlo prima perché è affine) e versione tensor dei nodi della mesh
stiffness, advection, mass, weakTerm_down = FEM_funct.FEM_funct(problemData, lib)

## Costruzione ROM space con algoritmo Greedy e confronto con soluzione FEM su un sottoinsieme dello spazio dei parametri
parameters = [[1, 1], [7, 0.7], [5, 0], [3, -0.5], [0.1, 1], [.1, -1], [10, -1], [.5, .5], [6, -0.7]]
tol = 1.e-8
N_max = 20
M_val = [100, 250, 500, 1000, 2000]
for M in M_val:
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("\n")
    print("Greedy solution with {} points in the parametric space".format(M))
    print("\n")
    dir = this_dir+"/results_plot_Greedy/{}".format(M)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir+'/error.csv', 'w', newline='') as csvfile:
        fieldnames = ['parameters', 'error_inf', 'error_2', 'error_H1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    B, stiffness_RB, advection_RB, weakTerm_down_RB = Greedy_funct.Greedy_funct(problemData, lib, M, tol, N_max)
    # Open the file and load the file
    for combo in parameters:
        print("combo = {}".format(combo))
        mu_1 = float(combo[0])
        mu_2 = float(combo[1])

        solution_FEM = gedim.LUSolver(mu_1*stiffness+advection, mu_2*weakTerm_down, lib)
        gedim.PlotSolution(mesh, dofs, strongs, solution_FEM, np.zeros(problemData['NumberStrongs']), title='Solution_FEM')
        solution_Greedy_RB = LA.solve(mu_1*(B.T @ stiffness @ B) + (B.T @ advection @ B), mu_2*(B.T @ weakTerm_down))
        solution_Greedy = B @ solution_Greedy_RB
        gedim.PlotSolution(mesh, dofs, strongs, solution_Greedy, np.zeros(problemData['NumberStrongs']), title='Solution_RB')

        ##calcolo errore L^inf, L^2 e seminorma H^1
        error = solution_FEM - solution_Greedy
        l_inf = LA.norm(error, np.inf)
        l_2 = np.sqrt(np.abs(error.T @ mass @ error))
        semi_h_1 = np.sqrt(np.abs(error.T @ stiffness @ error))
        with open(dir+'/error.csv', 'a', newline='') as csvfile:
            fieldnames = ['parameters', 'error_inf', 'error_2', 'error_H1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({"parameters": str(combo), "error_inf": l_inf, "error_2": l_2, "error_H1": semi_h_1})


print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("\n")