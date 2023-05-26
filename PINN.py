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

import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("--lam")
ap.add_argument("--iterations")
ap.add_argument("--points")
args = vars(ap.parse_args())
#iterations = int(args['iterations'])
iterations= 10000
#lam = float(args['lam']) #tra 0 e 1
lam = 1.e-2
#n_points = int(args['points'])
n_points = 20000
theta_dir = 1.e3
theta_neu = 1.e1
bc_dict = dict(zip([1,2,3,4],[1,1,0,0])) #La chiave è il bordo, il valore è il tipo di condizione. 0 sta per Dirichlet, 1 sta per Neumann
tol = 1e-4
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("\n")
print("Mandando una simulazione con {} iterazioni e parametro lambda che vale {}, valutata in {} punti.".format(iterations, lam, n_points))
print("\n")

#creo la cartella dove salvare i risultati
dir = "results/{}/{}/{}".format(iterations, lam, n_points)
if not os.path.exists(dir):
    os.makedirs(dir)

complete_dir = os.getcwd()+"/"+dir
with open(complete_dir+'/loss.csv', 'w', newline='') as csvfile:
    fieldnames = ['epoch', 'loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

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

    mu_1_range = [1.,1.]
    mu_2_range = [-1., -1.]
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
        
    def beta_1(x,y):
        return y*(1-y)

    def beta_2(x,y):
        return torch.zeros((x.shape[0],1))
    
    def f_pde(x,y,mu_1,mu_2):
        #f = 32.*mu_1*(y*(1-y) + x*(1-x)) + (1-x)*(16.*y**4 - 32.*y**3 + 16.*y**2)
        #return f
        return torch.zeros((x.shape[0],1))
    
    def f_bc_1(x,y,mu_1,mu_2):
        return mu_2
        #return torch.zeros((x.shape[0],1))

    def f_bc_2(x,y,mu_1,mu_2):
        return torch.zeros((x.shape[0],1))
    
    def f_bc_3(x,y,mu_1,mu_2):
        #return torch.zeros((x.shape[0],1))
        return x**2
    
    def f_bc_4(x,y,mu_1,mu_2):
        return torch.zeros((x.shape[0],1))

        
    def R_pde(x, y, mu_1, mu_2, net): 
        '''residuo pde'''
        u = net(x,y,mu_1,mu_2)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        pde = -mu_1*(u_xx + u_yy) + beta_1(x,y)*u_x + beta_2(x,y)*u_y
        return pde

    def R_dir(x,y,mu_1,mu_2,net):
        return net(x,y,mu_1,mu_2)
    
    def R_neu(x,y,mu_1,mu_2,net,n_x,n_y):
        u = net(x,y,mu_1,mu_2)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        normal_der = mu_1*(u_x*n_x + u_y*n_y)
        return normal_der
    
    net = Net()
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.ASGD(net.parameters())

    #PRIMO TENTATIVO: facciamo finta che il problema sia in 4 dimensioni e alleniamo la rete esattamente come nel caso 2d, cioè mu_1 e mu_2 
    #si considerano delle variabili esattamente come x e y

    #Data from boundary condition 1:bordo inferiore 2:bordo destro 3: bordo superiore 4: bordo sinistro

    #I nodi di Dirichlet non vanno ricreati perché non sono dof, i Neumann invece, come i punti nel dominio, sono dof e quindi per allenare
    #la rete vanno cambiati ad ogni iterazione
    loss = 5
    mse_table = np.zeros([iterations,5])
    epoch = 0
    while epoch < iterations and loss > tol:

        optimizer.zero_grad()

        ## LOSS NEI DOF
        # Genero i parametri
        mu_1 = np.random.uniform(low = mu_1_range[0], high = mu_1_range[1], size=(n_points,1))
        pt_mu_1 = Variable(torch.from_numpy(mu_1).float(), requires_grad = False)
        mu_2 = np.random.uniform(low = mu_2_range[0], high = mu_2_range[1], size=(n_points,1))
        pt_mu_2 = Variable(torch.from_numpy(mu_2).float(), requires_grad = False)
        #Genero i punti nel dominio
        x_collocation = np.random.uniform(low=0.0, high=1.0, size=(n_points,1))
        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad = True)
        y_collocation = np.random.uniform(low=0.0, high=1.0, size=(n_points,1))
        pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad = True)
        #Genero i valori obiettivo
        res_obj = f_pde(x_collocation,y_collocation,pt_mu_1,pt_mu_2)
        #Calcolo il residuo
        res_out = R_pde(pt_x_collocation, pt_y_collocation, pt_mu_1, pt_mu_2, net)
        mse_pde = mse_cost_function(res_out, res_obj)
        mse_table[epoch, 0] = mse_pde
        
        ##LOSS BORDO 1 ([0,1]x{0}), NEUMANN NON OMOGENEO
        # Genero i parametri
        mu_1 = np.random.uniform(low = mu_1_range[0], high = mu_1_range[1], size=(n_points,1))
        pt_mu_1 = Variable(torch.from_numpy(mu_1).float(), requires_grad = False)
        mu_2 = np.random.uniform(low = mu_2_range[0], high = mu_2_range[1], size=(n_points,1))
        pt_mu_2 = Variable(torch.from_numpy(mu_2).float(), requires_grad = False)
        #Genero i punti sul bordo
        x_collocation = np.random.uniform(low=0.0, high=1.0, size=(n_points,1))
        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad = True)
        y_collocation = np.zeros((n_points,1))
        pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad = True)
        #Genero i valori obiettivo
        res_obj = f_bc_1(x_collocation,y_collocation,pt_mu_1,pt_mu_2)
        #Calcolo il residuo
        if bc_dict[1] == 0:
            res_out = R_dir(pt_x_collocation, pt_y_collocation, pt_mu_1, pt_mu_2, net)
        else:
            res_out = R_neu(pt_x_collocation, pt_y_collocation, pt_mu_1, pt_mu_2, net, 0, -1)
        #Calcolo MSE
        mse_bc_1 = mse_cost_function(res_out, res_obj)
        mse_table[epoch, 1] = mse_bc_1

        ##LOSS BORDO 2 ({1}x[0,1]), NEUMANN OMOGENEO
        # Genero i parametri
        mu_1 = np.random.uniform(low = mu_1_range[0], high = mu_1_range[1], size=(n_points,1))
        pt_mu_1 = Variable(torch.from_numpy(mu_1).float(), requires_grad = False)
        mu_2 = np.random.uniform(low = mu_2_range[0], high = mu_2_range[1], size=(n_points,1))
        pt_mu_2 = Variable(torch.from_numpy(mu_2).float(), requires_grad = False)
        #Genero i punti sul bordo
        x_collocation = np.ones((n_points,1))
        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad = True)
        y_collocation = np.random.uniform(low=0.0, high=1.0, size=(n_points,1))
        pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad = True)
        #Genero i valori obiettivo
        res_obj = f_bc_2(x_collocation,y_collocation,pt_mu_1,pt_mu_2)
        #Calcolo il residuo
        if bc_dict[2] == 0:
            res_out = R_dir(pt_x_collocation, pt_y_collocation, pt_mu_1, pt_mu_2, net)
        else:
            res_out = R_neu(pt_x_collocation, pt_y_collocation, pt_mu_1, pt_mu_2, net, 1, 0)
        #Calcolo MSE
        mse_bc_2 = mse_cost_function(res_out, res_obj)
        mse_table[epoch, 2] = mse_bc_2

        ##LOSS BORDO 3 ([0,1]x{1}), DIRICHLET OMOGENEO
        # Genero i parametri
        mu_1 = np.random.uniform(low = mu_1_range[0], high = mu_1_range[1], size=(n_points,1))
        pt_mu_1 = Variable(torch.from_numpy(mu_1).float(), requires_grad = False)
        mu_2 = np.random.uniform(low = mu_2_range[0], high = mu_2_range[1], size=(n_points,1))
        pt_mu_2 = Variable(torch.from_numpy(mu_2).float(), requires_grad = False)
        #Genero i punti sul bordo
        x_collocation = np.random.uniform(low=0.0, high=1.0, size=(n_points,1))
        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad = True)
        y_collocation = np.ones((n_points,1))
        pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad = True)
        #Genero i valori obiettivo
        res_obj = f_bc_3(pt_x_collocation,pt_y_collocation,pt_mu_1,pt_mu_2)
        #Calcolo il residuo
        if bc_dict[3] == 0:
            res_out = R_dir(pt_x_collocation, pt_y_collocation, pt_mu_1, pt_mu_2, net)
        else:
            res_out = R_neu(pt_x_collocation, pt_y_collocation, pt_mu_1, pt_mu_2, net, 0, 1)
        #Calcolo MSE
        mse_bc_3 = mse_cost_function(res_out, res_obj)
        mse_table[epoch, 3] = mse_bc_3

        ##LOSS BORDO 4 ({0}x[0,1]), DIRICHLET OMOGENEO
        # Genero i parametri
        mu_1 = np.random.uniform(low = mu_1_range[0], high = mu_1_range[1], size=(n_points,1))
        pt_mu_1 = Variable(torch.from_numpy(mu_1).float(), requires_grad = False)
        mu_2 = np.random.uniform(low = mu_2_range[0], high = mu_2_range[1], size=(n_points,1))
        pt_mu_2 = Variable(torch.from_numpy(mu_2).float(), requires_grad = False)
        #Genero i punti sul bordo
        x_collocation = np.zeros((n_points,1))
        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad = True)
        y_collocation = np.random.uniform(low=0.0, high=1.0, size=(n_points,1))
        pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad = True)
        #Genero i valori obiettivo
        res_obj = f_bc_4(pt_x_collocation,pt_y_collocation,pt_mu_1,pt_mu_2)
        #Calcolo il residuo
        if bc_dict[4] == 0:
            res_out = R_dir(pt_x_collocation, pt_y_collocation, pt_mu_1, pt_mu_2, net)
        else:
            res_out = R_neu(pt_x_collocation, pt_y_collocation, pt_mu_1, pt_mu_2, net, -1, 0)
        #Calcolo MSE
        mse_bc_4 = mse_cost_function(res_out, res_obj)
        mse_table[epoch, 4] = mse_bc_4

        ##CALCOLO LOSS TOTALE
        #mse_bc = mse_bc_1+mse_bc_2+mse_bc_3+mse_bc_4
        if lam == 0.0:
            max_value = max(mse_pde, mse_bc_1, mse_bc_2, mse_bc_3, mse_bc_4)
            a = 1.0
            b = 1.0
            c = 1.0
            d = 1.0
            e = 1.0
            while max_value/(a*mse_pde) > 10:
                a *= 10
                print("a={}".format(a))
            while max_value/(b*mse_bc_1) > 10:
                b *= 10
                print("b={}".format(b))
            while max_value/(c*mse_bc_2) > 10:
                c *= 10
                print("c={}".format(c))
            while max_value/(d*mse_bc_3) > 10:
                d *= 10
                print("d={}".format(d))
            while max_value/(e*mse_bc_4) > 10:
                e *= 10
                print("e={}".format(e))
            a = min(a, 10e3)
            b = min(b, 10e3)
            c = min(c, 10e3)
            d = min(d, 10e5)
            e = min(e, 10e5)
            print(a)
            print(b)
            print(c)
            print(d)
            print(e)
            loss = a*mse_pde + b*mse_bc_1 + c*mse_bc_2 + d*mse_bc_3 + e*mse_bc_4
        else:
            loss = lam*mse_pde + (1-lam)*mse_bc_1 + (1-lam)*mse_bc_2 + (1-lam)*mse_bc_3 + (1-lam)*mse_bc_4
        loss = mse_pde + theta_neu*(mse_bc_1 + mse_bc_2) + theta_dir*(mse_bc_3 + mse_bc_4)
        writer.writerow({"epoch":epoch, "loss":loss.item()})
        loss.backward()
        optimizer.step()
        print('epoch:',epoch,' mse',mse_table[epoch,:])
        epoch += 1
        torch.autograd.no_grad()
        


np.savetxt('mse_table.csv',mse_table,delimiter=',')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x = np.arange(0,1,0.02)
y = np.arange(0,1,0.02)
ms_x, ms_y = np.meshgrid(x,y)
x = np.ravel(ms_x).reshape(-1,1)
y = np.ravel(ms_y).reshape(-1,1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True)
pt_u = net(pt_x, pt_y, 1.*torch.ones((pt_x.shape[0],1)), -1.*torch.ones((pt_y.shape[0],1)))
#u = pt_u.data.cpu().numpy()
u = pt_u.numpy(force=True)
ms_u = u.reshape(ms_x.shape)
surf = ax.plot_surface(ms_x, ms_y, ms_u, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.savefig(complete_dir+'/foto.png')
plt.show()