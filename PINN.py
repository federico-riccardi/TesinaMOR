#!/usr/bin/env python3

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
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')

if not os.path.exists("CppToPython"):
    sys.exit("CppToPython does not exist, please run mesh.sh.") 


sys.path.append("CppToPython")
sys.path.insert(0, '../Utilities/')

import GeDiM4Py as gedim

#Parameters
meshSize = 0.01
order = 2
plotMesh = True

os.chdir("CppToPython")
lib = gedim.ImportLibrary("./release/GeDiM4Py.so")
config = { 'GeometricTolerance': 1.0e-8 }
gedim.Initialize(config, lib)

#Dirichlet sui lati e in alto, Neumann sotto
#EdgesBoundaryCondition: 2 se Neumann, 1 se Dirichlet, a partire da sotto
domain = { 'SquareEdge': 1.0, 'VerticesBoundaryCondition': [1,1,1,1], 'EdgesBoundaryCondition': [2,1,1,1], 'DiscretizationType': 1, 'MeshCellsMaximumArea': meshSize }
[meshInfo, mesh] = gedim.CreateDomainSquare(domain, lib)

if plotMesh:
    gedim.PlotMesh(mesh)

#BoundaryConditionsType:
discreteSpace = { 'Order': order, 'Type': 1, 'BoundaryConditionsType': [1,2,3] }
[problemData, dofs, strongs] = gedim.Discretize(discreteSpace, lib)

if plotMesh:
    gedim.PlotDofs(mesh, dofs, strongs)

np.random.seed(1234)

mu_1_range = [0.1,10.]
mu_2_range = [-1., 1.]
P = np.array([mu_1_range,mu_2_range])
input_dim = 2 + P.shape[0]
output_dim = 1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(input_dim,5)
        self.hidden_layer1 = nn.Linear(5,5)
        self.hidden_layer2 = nn.Linear(5,5)
        self.hidden_layer3 = nn.Linear(5,5)
        self.hidden_layer4 = nn.Linear(5,5)
        self.output_layer = nn.Linear(5,output_dim)

    def forward(self, x,y,mu_1,mu_2):
        input = torch.cat([x,y,mu_1,mu_2],axis=1) # combines the column array
        layer1_out = torch.sigmoid(self.input_layer(input))
        layer2_out = torch.sigmoid(self.hidden_layer1(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer2(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer3(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer4(layer4_out))
        output = self.output_layer(layer5_out)
        return output
    
def beta(x,y):
    return np.array([y*(1-y),0])
    
def R(x,y,mu_1,mu_2,net): #residuo pde
    u = net(x,y,mu_1,mu_2)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    pde = mu_1*(u_xx + u_yy) + beta(x,y)[0]*u_x + beta(x,y)[1]*u_y
    return pde



net = Net()
mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters())

