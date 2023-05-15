#!/usr/bin/env python3


import numpy as np
import sys
import os
sys.path.append("CppToPython")

import GeDiM4Py as gedim


os.chdir("CppToPython")
lib = gedim.ImportLibrary("./release/GeDiM4Py.so")
config = { 'GeometricTolerance': 1.0e-8 }
gedim.Initialize(config, lib)
meshSize = 0.01
order = 2
domain = { 'SquareEdge': 1.0, 'VerticesBoundaryCondition': [1,1,1,1], 'EdgesBoundaryCondition': [2,1,1,1], 'DiscretizationType': 1, 'MeshCellsMaximumArea': meshSize }
[meshInfo, mesh] = gedim.CreateDomainSquare(domain, lib)
gedim.PlotMesh(mesh)
discreteSpace = { 'Order': order, 'Type': 1, 'BoundaryConditionsType': [1,2,3] }
[problemData, dofs, strongs] = gedim.Discretize(discreteSpace, lib)

#Dirichlet sui lati e in alto, Neumann sotto
gedim.PlotDofs(mesh, dofs, strongs)

import sys
sys.path.insert(0, '../Utilities/')

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')

np.random.seed(1234)