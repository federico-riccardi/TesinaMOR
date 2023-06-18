#!/usr/bin/env python3
#%%
import sys
from scipy.sparse.linalg import splu
import numpy as np
sys.path.append("CppToPython")
sys.path.insert(0, '../Utilities/')
import GeDiM4Py as gedim
import scipy

def Greedy_funct(problemData, lib, M, tol, N_max):
        
    def Poisson_a(numPoints, points):
        values = np.ones(numPoints)
        return values.ctypes.data

    def Poisson_b(numPoints, points):
        values = np.zeros((2, numPoints))
        matPoints = gedim.make_nd_matrix(points, (3, numPoints), np.double)
        values[0,:] = matPoints[1,:]*(1.-matPoints[1,:])
        return values.ctypes.data

    def Poisson_c(numPoints, points):
        values = np.ones(numPoints)
        return values.ctypes.data

    def Poisson_weakTerm_down(numPoints, points):
        values = np.ones(numPoints)
        return values.ctypes.data

    [stiffness, stiffnessStrong] = gedim.AssembleStiffnessMatrix(Poisson_a, problemData, lib)

    [advection, advectionStrong] = gedim.AssembleAdvectionMatrix(Poisson_b, problemData, lib)

    [mass, massStrong] = gedim.AssembleReactionMatrix(Poisson_c, problemData, lib) # serve solo per calcolare la costante di Poincaré

    weakTerm_down = gedim.AssembleWeakTerm(Poisson_weakTerm_down, 2, problemData, lib)


    ### Spazio dei parametri e training set
    mu1_range = [0.1, 10.]
    mu2_range = [-1., 1.]
    P = np.array([mu1_range, mu2_range])
    training_set = np.random.uniform(low=P[:, 0], high=P[:, 1], size=(M, P.shape[0]))

    # Definizione prodotti scalari, ortonormalizzazione, residui ed errori
    X = stiffness
    invX = splu(X)

    def normX(v, X):
        return np.sqrt(np.transpose(v) @ X @ v)

    def GramSchmidt(V, u):
        z = u
        if np.size(V) > 0:
            z = u - V @ (np.transpose(V) @ (X @ u))
        return z / normX(z, X)

    def OfflineResidual(B):
        N = B.shape[1] # numero di elementi nella base
        C_11 = np.array(weakTerm_down.T @ (invX.solve(weakTerm_down)), ndmin=2) # Xu = f_1 -> u = X^{-1} f_1 = invX.solve(f_1)
        d_11 = (np.array(B.T @ stiffness.T @ (invX.solve(weakTerm_down)), ndmin=2)).reshape((N,1)) #il cambio di shape è necessario perché altrimenti viene un vettore riga
        d_12 = (np.array(B.T @ advection.T @ (invX.solve(weakTerm_down)), ndmin=2)).reshape((N,1))
        E_11 = B.T @ stiffness.T @ (invX.solve(stiffness @ B))
        E_12 = B.T @ stiffness.T @ (invX.solve(advection @ B))
        E_22 = B.T @ advection.T @ (invX.solve(advection @ B))
        return [C_11, d_11, d_12, E_11, E_12, E_22]
        
    def ErrorEstimate(mu, solN_mu, off_res, beta_mu):
        pre_mult = [np.array([mu[1]**2]), -2*mu[0]*mu[1]*solN_mu.T, -2*mu[1]*solN_mu.T, mu[0]**2*solN_mu.T, 2*mu[0]*solN_mu.T, solN_mu.T]
        post_mult = [np.array([1.]), np.array([1.]), np.array([1.]), solN_mu, solN_mu, solN_mu]
        error = 0.0
        for i in range(len(off_res)):
            error += pre_mult[i] @ off_res[i] @ post_mult[i]
        return np.sqrt(np.abs(error))/beta_mu

    eigs, vecs = scipy.linalg.eig(stiffness.todense(), mass.todense())
    min_eig = np.min(eigs.real)
    C_omega =  1 / np.sqrt(min_eig) #costante di Poincaré
    def InfSupConstant(mu):
        return mu[0]/(1+C_omega**2)

    #Inizio costruzione ROM
    N = 0
    basis_functions = []
    B = np.empty((0,0))
    delta_N = tol + 1.
    training_set_list = training_set.tolist()
    initial_muN = np.random.choice(len(training_set_list) - 1, 1)[0]
    mu_N = training_set_list.pop(initial_muN)
    beta_mu = InfSupConstant(mu_N)
    print('Perfom greedy algorithm...')
    flag = 0
    while len(training_set_list) > 0 and N < N_max and delta_N > tol and flag == 0:
        N += 1
        snapshot = gedim.LUSolver(mu_N[0]*X+advection,mu_N[1]*weakTerm_down,lib)
        basis_function = GramSchmidt(B, snapshot)
        basis_functions.append(np.copy(basis_function))
        B = np.transpose(np.array(basis_functions))
        BX = np.transpose(B) @ X @ B
        stiffness_RB = B.T @ stiffness @ B
        advection_RB = B.T @ advection @ B
        weakTerm_down_RB = B.T @ weakTerm_down
        off_res = OfflineResidual(B)
        counter = 0
        mu_selected_index = -1
        max_deltaN = -1.
        for mu in training_set_list:
            if not np.linalg.det(mu[0]*stiffness_RB + advection_RB) < tol:
                solN_mu =  np.linalg.solve(mu[0]*stiffness_RB + advection_RB, mu[1]*weakTerm_down_RB)
                deltaN_mu = ErrorEstimate(mu, solN_mu, off_res, beta_mu) / normX(solN_mu, BX)

                if deltaN_mu > max_deltaN:
                    max_deltaN = deltaN_mu
                    mu_selected_index = counter

            counter += 1

        if mu_selected_index == -1:
            flag = 1

        mu_N = training_set_list.pop(mu_selected_index)
        delta_N = max_deltaN  

    print('We used {} bases'.format(N))

    return np.transpose(np.array(basis_functions)), stiffness_RB, advection_RB, weakTerm_down_RB