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

#figuwarnings.filterwarnings('ignore')

if not os.path.exists("CppToPython"):
    sys.exit("CppToPython does not exist, please run mesh.sh.") 

sys.path.append("CppToPython")
sys.path.insert(0, '../Utilities/')

import GeDiM4Py as gedim

#Parameters
meshSize = 0.01
order = 2
plotMesh = True

#sys.path.append("PINN_funct.py")
import PINN_funct

iterations= 10000
coeff = [10, 10, 500, 500]
n_points = 20000
delta = 0.1 #parametro per funzione cutoff

mse_table, net = PINN_funct.PINN_funct(iterations, coeff, n_points, delta)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x = np.arange(0,1,0.02)
y = np.arange(0,1,0.02)
ms_x, ms_y = np.meshgrid(x,y)
x = np.ravel(ms_x).reshape(-1,1)
y = np.ravel(ms_y).reshape(-1,1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True)
pt_u = net(pt_x, pt_y, .1*torch.ones((pt_x.shape[0],1)), -1.*torch.ones((pt_y.shape[0],1)))
#u = pt_u.data.cpu().numpy()
u = pt_u.numpy(force=True)
ms_u = u.reshape(ms_x.shape)
surf = ax.plot_surface(ms_x, ms_y, ms_u, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.savefig('fotoPINN.png')