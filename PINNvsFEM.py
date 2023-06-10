#!/usr/bin/env python3

import yaml
from yaml.loader import SafeLoader
import subprocess
import FEM_funct
import sys
import os
import PINN_funct
import numpy as np
import torch
from torch.autograd import Variable
from numpy import linalg as LA
import csv

if not os.path.exists("CppToPython"):
    sys.exit("CppToPython does not exist, please run mesh.sh.") 

sys.path.append("CppToPython")
sys.path.insert(0, '../Utilities/')

import GeDiM4Py as gedim
this_dir = os.getcwd()
os.chdir("CppToPython")
lib = gedim.ImportLibrary("./release/GeDiM4Py.so")

config = { 'GeometricTolerance': 1.0e-8 }
gedim.Initialize(config, lib)
order = 1
meshSize = 0.001

domain = { 'SquareEdge': 1.0, 'VerticesBoundaryCondition': [1,0,1,1], 'EdgesBoundaryCondition': [2,3,1,1], 'DiscretizationType': 1, 'MeshCellsMaximumArea': meshSize }
[meshInfo, mesh] = gedim.CreateDomainSquare(domain, lib)
print(meshInfo)

discreteSpace = { 'Order': order, 'Type': 1, 'BoundaryConditionsType': [1, 2, 3, 3] }
[problemData, dofs, strongs] = gedim.Discretize(discreteSpace, lib)
print(dofs.shape)
print(dofs[0])



## FEM (posso farlo prima perché è affine)
stiffness, advection, weakTerm_down = FEM_funct.FEM_funct(problemData, lib)
pt_x = Variable(torch.from_numpy(np.array([dofs[0]]).T).float(), requires_grad=True)
pt_y = Variable(torch.from_numpy(np.array([dofs[1]]).T).float(), requires_grad=True)
pt_x_s = Variable(torch.from_numpy(np.array([strongs[0]]).T).float(), requires_grad=True)
pt_y_s = Variable(torch.from_numpy(np.array([strongs[1]]).T).float(), requires_grad=True)
#solution = gedim.LUSolver(mu_1*stiffness+advection, mu_2*weakTerm_down, lib)
#gedim.PlotSolution(mesh, dofs, strongs, solution, np.zeros(problemData['NumberStrongs']), title='Solution_FEM')

##RUN PINN [.1, 10]x[-1, 1]
delta = 0.1
parameters = [[1, 1], [7, 0.7], [5, 0], [3, -0.5], [0.1, 1], [.1, -1], [10, -1], [10, -1], [.5, .5], [6, -0.7]]
# Open the file and load the file
with open('/root/TesinaMOR/Configurazioni.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
    for iter in data['iterations']:
        for l in data['coeff']:
            coeff = [float(l[1]), float(l[2]), float(l[3]), float(l[4])]
            dir = this_dir+"/results_plot1/{}/{}".format(iter, coeff)
            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(dir+'/error.csv', 'w', newline='') as csvfile:
                fieldnames = ['parameters', 'error_inf', 'error_2']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for p in data['points']:
                    mse_table, net = PINN_funct.PINN_funct(iter, coeff, p, delta)

                    ## scelgo mu_1 mu_2
                    for combo in parameters:
                        print("combo = {}".format(combo))
                        mu_1 = float(combo[0])
                        mu_2 = float(combo[1])
                        solution_FEM = gedim.LUSolver(mu_1*stiffness+advection, mu_2*weakTerm_down, lib)
                        gedim.PlotSolution(mesh, dofs, strongs, solution_FEM, np.zeros(problemData['NumberStrongs']), title='Solution_FEM')
                        solution_PINN = net(pt_x, pt_y, mu_1*torch.ones((pt_x.shape[0],1)), mu_2*torch.ones((pt_y.shape[0],1))).detach().numpy().T[0]
                        solution_PINN_strong = net(pt_x_s, pt_y_s, mu_1*torch.ones((pt_x_s.shape[0],1)), mu_2*torch.ones((pt_y_s.shape[0],1))).detach().numpy().T[0]

                        ##calcolo errore L^inf
                        int_error = solution_FEM - solution_PINN
                        error = np.concatenate([int_error, solution_PINN_strong])
                        l_inf = LA.norm(error, np.inf)
                        l_2 = LA.norm(error, 2)
                        writer.writerow({"parameters": [mu_1, mu_2], "error_inf": l_inf, "error_2": l_2})


print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("\n")