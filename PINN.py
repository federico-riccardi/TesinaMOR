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
iterations = 1000
#iterations = int(input("please insert iteration:"))

#creo la cartella dove salvare i risultati
dir = str(iterations)
if not os.path.exists(dir):
    os.mkdir(dir)
complete_dir = os.getcwd()+"/"+dir
with open(complete_dir+'/loss.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ')

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
    input_dim = 2 + P.shape[0] #x, y and parameters
    output_dim = 1

    class Net(nn.Module):
        def __init__(self):
            '''
            It defines the structure: the number of layers and the number of nodes for each layer
            '''
            super(Net, self).__init__()
            self.input_layer = nn.Linear(input_dim,10)
            self.hidden_layer1 = nn.Linear(10,10)
            self.hidden_layer2 = nn.Linear(10,5)
            self.hidden_layer3 = nn.Linear(5,5)
            self.hidden_layer4 = nn.Linear(5,5)
            self.output_layer = nn.Linear(5,output_dim)

        def forward(self, x,y,mu_1,mu_2):
            '''
            It defines the advancing method
            '''
            input = torch.cat([x,y,mu_1,mu_2],axis=1) # combines the column array
            layer1_out = torch.sigmoid(self.input_layer(input))
            layer2_out = torch.sigmoid(self.hidden_layer1(layer1_out))
            layer3_out = torch.sigmoid(self.hidden_layer2(layer2_out))
            layer4_out = torch.sigmoid(self.hidden_layer3(layer3_out))
            layer5_out = torch.sigmoid(self.hidden_layer4(layer4_out))
            output = self.output_layer(layer5_out)
            return output
        
    def beta(x,y):
        beta_1 = y*(1-y)
        beta_2 = torch.zeros((500,1))
        return torch.cat((beta_1,beta_2), dim=1)

    def beta_1(x,y):
        return y*(1-y)

    def beta_2(x,y):
        return torch.zeros((x.shape[0],1))
        
    def R_pde(x, y, mu_1, mu_2, net): 
        '''residuo pde'''
        u = net(x,y,mu_1,mu_2)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        pde = mu_1*(u_xx + u_yy) + beta_1(x,y)*u_x + beta_2(x,y)*u_y
        return pde

    def R_bc_1(x, y, mu_1, mu_2, net): 
        '''residuo Neumann bordo 1'''
        u = net(x,y,mu_1,mu_2)
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        normal_der = u_y
        return normal_der

    def R_bc_2(x, y, mu_1, mu_2, net): 
        '''residuo Neumann bordo 1'''
        u = net(x,y,mu_1,mu_2)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        normal_der = u_x
        return normal_der

    net = Net()
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(net.parameters())

    #PRIMO TENTATIVO: facciamo finta che il problema sia in 4 dimensioni e alleniamo la rete esattamente come nel caso 2d, cioè mu_1 e mu_2 
    #si considerano delle variabili esattamente come x e y

    #Data from boundary condition 1:bordo inferiore 2:bordo destro 3: bordo superiore 4: bordo sinistro

    #I nodi di Dirichlet non vanno ricreati perché non sono dof, i Neumann invece, come i punti nel dominio, sono dof e quindi per allenare
    #la rete vanno cambiati ad ogni iterazione

    mu_1_bc_dir = np.random.uniform(low = mu_1_range[0], high = mu_1_range[1], size=(1,1))*np.ones((500,1))
    mu_2_bc_dir = np.random.uniform(low = mu_2_range[0], high = mu_2_range[1], size=(1,1))*np.ones((500,1))

    x_bc_3 = np.random.uniform(low=0.0, high=1.0, size=(500,1))
    y_bc_3 = np.ones((500,1))
    u_bc_3 = np.zeros((500,1))

    x_bc_4 = np.zeros((500,1))
    y_bc_4 = np.random.uniform(low=0.0, high=1.0, size=(500,1))
    u_bc_4 = np.zeros((500,1))


    for epoch in range(iterations):
        optimizer.zero_grad()
        #Loss condizioni al contorno di Dirichlet
        pt_mu_1_bc_dir = Variable(torch.from_numpy(mu_1_bc_dir).float(), requires_grad = False)
        pt_mu_2_bc_dir = Variable(torch.from_numpy(mu_2_bc_dir).float(), requires_grad = False)

        pt_x_bc_3 = Variable(torch.from_numpy(x_bc_3).float(), requires_grad = False)
        pt_y_bc_3 = Variable(torch.from_numpy(y_bc_3).float(), requires_grad = False)
        pt_u_bc_3 = Variable(torch.from_numpy(u_bc_3).float(), requires_grad = False)
        net_bc_out_3 = net(pt_x_bc_3, pt_y_bc_3, pt_mu_1_bc_dir, pt_mu_2_bc_dir)
        mse_u_bc_3 = mse_cost_function(net_bc_out_3, pt_u_bc_3)

        pt_x_bc_4 = Variable(torch.from_numpy(x_bc_4).float(), requires_grad = False)
        pt_y_bc_4 = Variable(torch.from_numpy(y_bc_4).float(), requires_grad = False)
        pt_u_bc_4 = Variable(torch.from_numpy(u_bc_4).float(), requires_grad = False)
        net_bc_out_4 = net(pt_x_bc_4, pt_y_bc_4, pt_mu_1_bc_dir, pt_mu_2_bc_dir)
        mse_u_bc_4 = mse_cost_function(net_bc_out_4, pt_u_bc_4)
        

        #Loss nei dof
        #mu_1 = np.random.uniform(low = mu_1_range[0], high = mu_1_range[1], size=(1,1)) * np.ones((500,1))
        #mu_2 = np.random.uniform(low = mu_2_range[0], high = mu_2_range[1], size=(1,1)) *np.ones((500,1))
        mu_1 = mu_1_bc_dir
        mu_2 = mu_2_bc_dir
        pt_mu_1 = Variable(torch.from_numpy(mu_1).float(), requires_grad = False)
        pt_mu_2 = Variable(torch.from_numpy(mu_2).float(), requires_grad = False)
    
        x_collocation = np.random.uniform(low=0.0, high=1.0, size=(500,1))
        y_collocation = np.random.uniform(low=0.0, high=1.0, size=(500,1))
        y_bc_1 = np.zeros((500,1))
        x_bc_2 = np.ones((500,1))
        u_y_bc_1 = -np.divide(mu_2,mu_1)
        u_x_bc_2 = np.zeros((500,1))
        all_zeros = np.zeros((500,1))

        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad = True)
        pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad = True)
        pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad = False)

        pt_y_bc_1 = Variable(torch.from_numpy(y_bc_1).float(), requires_grad = True)
        pt_u_y_bc_1 = Variable(torch.from_numpy(u_y_bc_1).float(), requires_grad = False)

        pt_x_bc_2 = Variable(torch.from_numpy(x_bc_2).float(), requires_grad = True)    
        pt_u_x_bc_2 = Variable(torch.from_numpy(u_x_bc_2).float(), requires_grad = False)


        #Loss condizioni al contorno di Neumann

        f_out_bc_1 = R_bc_1(pt_x_collocation, pt_y_bc_1, pt_mu_1, pt_mu_2, net)
        mse_f_bc_1 = mse_cost_function(f_out_bc_1, pt_u_y_bc_1)

        f_out_bc_2 = R_bc_2(pt_x_bc_2, pt_y_collocation, pt_mu_1, pt_mu_2, net)
        mse_f_bc_2 = mse_cost_function(f_out_bc_2, pt_u_x_bc_2)

        f_out = R_pde(pt_x_collocation, pt_y_collocation, pt_mu_1, pt_mu_2, net)
        mse_f = mse_cost_function(f_out, pt_all_zeros)

        loss = mse_f + mse_f_bc_1 + mse_f_bc_2 + mse_u_bc_3 + mse_u_bc_4
        spamwriter.writerow([epoch, loss.item()])
        loss.backward()
        optimizer.step()

        with torch.autograd.no_grad():
            #print(epoch, "Loss:",loss.item())
            print(mse_f)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x = np.arange(0,1,0.02)
y = np.arange(0,1,0.02)
ms_x, ms_y = np.meshgrid(x,y)
x = np.ravel(ms_x).reshape(-1,1)
y = np.ravel(ms_y).reshape(-1,1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True)
pt_u = net(pt_x, pt_y, 7*torch.ones((pt_x.shape[0],1)), 0.4*torch.ones((pt_y.shape[0],1)))
u = pt_u.data.cpu().numpy()
ms_u = u.reshape(ms_x.shape)
surf = ax.plot_surface(ms_x, ms_y, ms_u, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.savefig(complete_dir+'/foto.png')
plt.show()