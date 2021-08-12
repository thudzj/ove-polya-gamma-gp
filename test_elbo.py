####
#
import sys
import copy
from collections import namedtuple
import warnings

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal

# from gpytorch.utils.cholesky import psd_safe_cholesky

# import pypolyagamma

# from gpytorch.utils.quadrature import GaussHermiteQuadrature1D

from sacred import Ingredient

############
from torch.autograd import Variable
import tqdm
# import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import expit
from scipy.special import psi
from scipy.stats import multinomial
from scipy.optimize import minimize
import gpytorch
from time import gmtime, strftime
import random
from configs import kernel_type
from scipy.special import gamma
from scipy.stats import multinomial
############


np.random.seed(0)
x=np.array(list(np.linspace(0+25/350/2,25-25/350/2,350))+list(np.linspace(35+25/350/2,60-25/350/2,350))+list(np.linspace(70+25/350/2,95-25/350/2,350)))
x=x.reshape(len(x),1)  #(1050, 1)
y=np.zeros((len(x),3)) #(1050, 3)
for i in range(len(x)):
    if i<=350:
        y[i]=multinomial(n=1,p=np.array([0.9,0.05,0.05])).rvs()[0]
    elif i<=700:
        y[i]=multinomial(n=1,p=np.array([0.05,0.9,0.05])).rvs()[0]
    else:
        y[i]=multinomial(n=1,p=np.array([0.05,0.05,0.9])).rvs()[0]

# y_test=np.zeros((len(x),3))
# for i in range(len(x)):
#     if i<=350:
#         y_test[i]=multinomial(n=1,p=np.array([0.9,0.05,0.05])).rvs()[0]
#     elif i<=700:
#         y_test[i]=multinomial(n=1,p=np.array([0.05,0.9,0.05])).rvs()[0]
#     else:
#         y_test[i]=multinomial(n=1,p=np.array([0.05,0.05,0.9])).rvs()[0]

X = torch.from_numpy(x)
Y = torch.from_numpy(y)
N = X.shape[0]
C = Y.shape[-1]
K = X.mm(X.t())+torch.eye(N, dtype=X.dtype, device=X.device)


# mu = torch.zeros(N, C, dtype=K.dtype, device=K.device)
# mu_n_c = torch.ones(N, C, dtype=K.dtype, device=K.device)  # N*C
# Sigma_n_c = (K+torch.eye(N, N, dtype=K.dtype, device=K.device)).repeat(C,1).reshape(C, N, N) # C*N*N
# f_n_c= torch.sqrt(torch.diag(K)).view(-1,1).repeat(1,C) # N*C
# alpha_n = torch.ones(N, dtype=K.dtype, device=K.device)
# gnc = torch.tensor(psi(alpha_n.data.cpu().numpy()),device=Y.device).reshape(-1,1).expand(N,C)
# gamma_n_c = torch.exp(gnc-mu_n_c/2)/2/C/torch.cosh(f_n_c/2)
# w_n_c = (gamma_n_c + Y)/2/f_n_c*torch.tanh(f_n_c/2)
# L = torch.slogdet(K)[1]
# K_inv = torch.inverse(K)

# ELBO = torch.sum(-(Y+gamma_n_c)*torch.log(torch.tensor([2], dtype=K.dtype, device=K.device))+(Y-gamma_n_c)/2*mu_n_c-w_n_c/2*(f_n_c)**2)\
# +torch.sum(alpha_n-torch.log(torch.tensor([C], dtype=K.dtype, device=K.device))+torch.log(torch.tensor(gamma(alpha_n.data.cpu().numpy()),device=K.device))+(1-alpha_n)*torch.digamma(alpha_n))\
# -torch.sum(gamma_n_c*(torch.log(gamma_n_c)-1)-gamma_n_c*((torch.digamma(alpha_n)-torch.log(torch.tensor([C], dtype=K.dtype, device=K.device))).reshape(-1,1)))-torch.sum(alpha_n/C)\
# -torch.sum(-f_n_c**2/2*w_n_c+(gamma_n_c+Y)*torch.log(torch.cosh(f_n_c/2)))

# # print(ELBO)

# for c in range(C):
#     ELBO-=0.5*(L-torch.slogdet(Sigma_n_c[c])[1]- N + torch.trace(torch.mm(K_inv,Sigma_n_c[c]))+ torch.mul(torch.mv(K_inv, mu[:,c]-mu_n_c[:,c]).t(),mu[:,c]-mu_n_c[:,c]).sum())
#     # print("****:", -ELBO)

# print("****:", -ELBO)



mu = torch.zeros(N, C, dtype=K.dtype, device=K.device)
mu_n_c = torch.ones(N, C, dtype=K.dtype, device=K.device)  # N*C
Sigma_n_c = (K+torch.eye(N, N, dtype=K.dtype, device=K.device)).repeat(C,1).reshape(C, N, N) # C*N*N
f_n_c= torch.sqrt(torch.diag(K)).view(-1,1).repeat(1,C) # N*C
alpha_n = torch.ones(N, dtype=K.dtype, device=K.device)
gnc = torch.tensor(psi(alpha_n.data.cpu().numpy()),device=Y.device).reshape(-1,1).expand(N,C)
gamma_n_c = torch.exp(gnc-mu_n_c/2)/2/C/torch.cosh(f_n_c/2)
w_n_c = (gamma_n_c + Y)/2/f_n_c*torch.tanh(f_n_c/2)
L = torch.slogdet(K)[1]
K_inv = torch.inverse(K)

def next_iter(mu, mu_n_c, Sigma_n_c, f_n_c, alpha_n, gnc, gamma_n_c, w_n_c, L, K_inv):
    temp_sigma = torch.inverse(torch.diag(w_n_c[:,0]) + K_inv)
    # print(temp_sigma.shape)
    for c in range(C - 1):
        temp_sigma = torch.cat((temp_sigma, torch.inverse(torch.diag(w_n_c[:,c+1]) + K_inv)), 0)
        # Sigma_n_c_new[c,:,:] = torch.inverse(torch.diag(pg_state.w_n_c[:,c]) + model_state.K_inv)
        # mu_n_c_new[:,c] = Sigma_n_c_new[c].mv(model_state.Y[:,c]-pg_state.gamma_n_c[:,c])/2
    Sigma_n_c_new = temp_sigma.reshape(C, N, N)


    temp_mu = Sigma_n_c_new[0].mv(Y[:,0]-gamma_n_c[:,0])/2
    for c in range(C - 1):
        temp_mu = torch.cat((temp_mu, Sigma_n_c_new[c+1].mv(Y[:,c+1]-gamma_n_c[:,c+1])/2), 0)
    mu_n_c_new = temp_mu.reshape(N, C)

    # for c in range(model_state.C):
    #     Sigma_n_c_new_c = torch.inverse(torch.diag(pg_state.w_n_c[:,c]) + model_state.K_inv)
    #     mu_n_c_new_c = Sigma_n_c_new_c.mv(model_state.Y[:,c]-pg_state.gamma_n_c[:,c])/2

    f_n_c_new = torch.sqrt(mu_n_c_new**2+torch.tensor([np.diag(Sigma_n_c_new[c].data.cpu().numpy()) for c in range(C)], device=K.device).T) ############

    gnc = torch.tensor(psi(alpha_n.data.cpu().numpy()),device=K.device).reshape(-1,1).expand(N,C)
    gamma_n_c_new = torch.exp(gnc-mu_n_c_new/2)/2/C/torch.cosh(f_n_c_new/2)  # element-wise div

    alpha_n_new = torch.sum(gamma_n_c_new,axis=1)+1

    w_n_c_new = (gamma_n_c_new + Y)/2/f_n_c_new*torch.tanh(f_n_c_new/2)
    return mu_n_c_new, Sigma_n_c_new, f_n_c_new, alpha_n_new, w_n_c_new, gamma_n_c


for i in range(2):
    ELBO = torch.sum(-(Y+gamma_n_c)*torch.log(torch.tensor([2], dtype=K.dtype, device=K.device))+(Y-gamma_n_c)/2*mu_n_c-w_n_c/2*(f_n_c)**2)\
    +torch.sum(alpha_n-torch.log(torch.tensor([C], dtype=K.dtype, device=K.device))+torch.log(torch.tensor(gamma(alpha_n.data.cpu().numpy()),device=K.device))+(1-alpha_n)*torch.digamma(alpha_n))\
    -torch.sum(gamma_n_c*(torch.log(gamma_n_c)-1)-gamma_n_c*((torch.digamma(alpha_n)-torch.log(torch.tensor([C], dtype=K.dtype, device=K.device))).reshape(-1,1)))-torch.sum(alpha_n/C)\
    -torch.sum(-f_n_c**2/2*w_n_c+(gamma_n_c+Y)*torch.log(torch.cosh(f_n_c/2)))
    for c in range(C):
        ELBO-=0.5*(L-torch.slogdet(Sigma_n_c[c])[1]- N + torch.trace(torch.mm(K_inv,Sigma_n_c[c]))+ torch.mul(torch.mv(K_inv, mu[:,c]-mu_n_c[:,c]).t(),mu[:,c]-mu_n_c[:,c]).sum())
    # print("****:", -ELBO)

    print("**********ELBO is**********", -ELBO)

    mu_n_c, Sigma_n_c, f_n_c, alpha_n, w_n_c, gamma_n_c =  next_iter(mu, mu_n_c, Sigma_n_c, f_n_c, alpha_n, gnc, gamma_n_c, w_n_c, L, K_inv)



# print("****:", -ELBO)
