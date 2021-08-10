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

from methods.meta_template import MetaTemplate
from methods.ove_polya_gamma_gp import psd_safe_cholesky

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
############

kernel_ingredient = Ingredient("kernel")


@kernel_ingredient.config
def get_config():
    name = "L2LinearKernel"
    learn_params = True


@kernel_ingredient.capture
def load_kernel(num_classes, name, learn_params):
    return getattr(sys.modules[__name__], name)(
        num_classes=num_classes, learn_params=learn_params
    )


######################################################################################
# linear kernel: LinearKernel  (output_scale_raw + nn param.)
# cosine lernel: L2LinearKernel  (output_scale_raw + nn param.)
# RBF kernel: RBFKernel  (output_scale_raw, lengthscale_raw, nn.param.)
# normalized RBF kernel: L2RBFKernel  (output_scale_raw, lengthscale_raw, nn.param.)
class Kernel(nn.Module):   
    def __init__(self, num_classes, learn_params):
        self.num_classes = num_classes
        self.learn_params = learn_params
        super(Kernel, self).__init__()

    def mean_function(self, X):
        raise NotImplementedError("mean_function not yet implemented")

    def cov_block(self, x1, x2):
        raise NotImplementedError("cov_block not yet implemented")

    def cov_block_diag(self, x1, x2):
        # naive implementation
        return torch.diag(self.cov_block_wrapper(x1, x2))

    def cov_block_wrapper(self, x1, x2=None):
        # x1: N x D
        # x2: N x D (or None)
        if x2 is None:
            x2 = x1
        return self.cov_block(x1, x2)

    def batch_mean_function(self, X):
        # X: N x D
        return self.mean_function(X).reshape(X.size(0), self.num_classes)

    # def cov_function(self, x1, x2=None):
    #     # aa=block_matrix(self.cov_block_wrapper(x1, x2), self.num_classes)
    #     # print(aa.shape) #425*425
    #     # print("block_matrix is", aa) 
    #     # print(self.cov_block_wrapper(x1, x2).shape) #85*85
    #     # print("cov_block_wrapper is", self.cov_block_wrapper(x1, x2))
    #     # return block_matrix(self.cov_block_wrapper(x1, x2), self.num_classes)
    #     return self.cov_block_wrapper(x1, x2)  # different from ove

    def cov_function(self, x1, x2=None):
        return block_matrix(self.cov_block_wrapper(x1, x2), self.num_classes)

    def batch_cov_function(self, x1, x2=None):
        return batch_block_matrix(self.cov_block_wrapper(x1, x2), self.num_classes)

    def batch_cov_function_diag(self, x1, x2=None):
        ret = self.cov_block_diag(x1, x2).unsqueeze(-1)
        ret = ret.expand(ret.size(0), self.num_classes)
        return torch.diag_embed(ret)

class LinearKernel(Kernel):   # linear kernel  (output_scale_raw + nn param.)
    def __init__(self, *args, **kwargs):
        super(LinearKernel, self).__init__(*args, **kwargs)
        # if self.learn_params:
        #     self.register_parameter("output_scale_raw", nn.Parameter(torch.tensor([1.0])))
        # else:
        #     self.register_buffer("output_scale_raw", torch.tensor([1.0]))
        ###########################
        if self.learn_params:
            self.register_parameter("output_scale_raw", nn.Parameter(torch.tensor([1.0])))
            self.register_parameter("mean_value", nn.Parameter(torch.zeros(1)))###################
        else:
            self.register_buffer("output_scale_raw", torch.tensor([1.0]))
            self.register_parameter("mean_value", torch.zeros(1))

    def mean_function(self, X):
        # return torch.zeros(X.size(0) * self.num_classes, dtype=X.dtype, device=X.device)
        return self.mean_value * torch.ones(
            X.size(0) * self.num_classes, dtype=X.dtype, device=X.device
        )


    def normalize(self, X):
        D = X.size(-1)
        return X / math.sqrt(D)

    def cov_block(self, x1, x2=None):  # Linear Kernel
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)
        # aaa=torch.exp(self.output_scale_raw) * (x1.mm(x2.t()))
        # print(aaa.shape) #85*85
        # print("cov_block is", aaa)
        return torch.exp(self.output_scale_raw) * (x1.mm(x2.t()))  # Linear Kernel Eq.48 


class ConstantMeanLinearKernel(LinearKernel):
    def __init__(self, *args, **kwargs):
        super(ConstantMeanLinearKernel, self).__init__(*args, **kwargs)
        assert self.learn_params == True, "if not learning, just use LinearKernel"
        self.register_parameter("mean_value", nn.Parameter(torch.zeros(1)))

    def mean_function(self, X):
        return self.mean_value * torch.ones(
            X.size(0) * self.num_classes, dtype=X.dtype, device=X.device
        )


class L2LinearKernel(LinearKernel):  # cosine kernel
    def normalize(self, X):
        return F.normalize(X)   # cosine kernel

class ConstantMeanL2LinearKernel(ConstantMeanLinearKernel):
    def normalize(self, X):
        return F.normalize(X)  # 2nd kernel: ConstantMean L2 LinearKernel


class QuadraticKernel(Kernel):
    def __init__(self, *args, **kwargs):
        super(QuadraticKernel, self).__init__(*args, **kwargs)
        if self.learn_params:
            self.register_parameter("output_scale_raw", nn.Parameter(torch.zeros(1)))
        else:
            self.register_buffer("output_scale_raw", torch.zeros(1))

    def mean_function(self, X):
        # return torch.zeros(X.size(0) * self.num_classes, dtype=X.dtype, device=X.device)
        return torch.ones(X.size(0) * self.num_classes, dtype=X.dtype, device=X.device)

    def normalize(self, X):
        D = X.size(-1)
        return X / math.sqrt(D)

    def cov_block(self, x1, x2=None):
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)
        return torch.exp(self.output_scale_raw) * (x1.mm(x2.t()) ** 2)


class L2QuadraticKernel(QuadraticKernel):
    def normalize(self, X):
        return F.normalize(X)    # 3rd kernel: L2 Quadratic Kernel


class RBFKernel(Kernel):  # RBF Kernel: output_scale_raw, lengthscale_raw, nn.param.
    def __init__(self, *args, **kwargs):
        super(RBFKernel, self).__init__(*args, **kwargs)
        if self.learn_params:
            self.register_parameter("output_scale_raw", nn.Parameter(torch.zeros(1)))
            self.register_parameter("lengthscale_raw", nn.Parameter(torch.zeros(1)))
        else:
            self.register_buffer("output_scale_raw", torch.zeros(1))
            self.register_buffer("lengthscale_raw", torch.zeros(1))

    def mean_function(self, X):
        return torch.zeros(X.size(0) * self.num_classes, dtype=X.dtype, device=X.device)

    def normalize(self, X):
        D = X.size(-1)
        return X / math.sqrt(D)

    def cov_block(self, x1, x2=None):
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)
        dists = (
            (x1 ** 2).sum(-1).view(-1, 1)
            + (x2 ** 2).sum(-1).view(1, -1)
            - 2 * x1.mm(x2.t())
        )
        print()
        return torch.exp(self.output_scale_raw) * torch.exp(
            -0.5 * dists / torch.exp(2.0 * self.lengthscale_raw)
        )


class L2RBFKernel(RBFKernel):  # normalized RBF Kernel
    def normalize(self, X):
        return F.normalize(X)  # 4st kernel: L2 RBFKernel  

def block_matrix(block, num_blocks):
    arr = []
    for i in range(num_blocks):
        row = []
        for j in range(num_blocks):
            if i == j:
                row.append(block)
            else:
                row.append(torch.zeros_like(block))
        row = torch.cat(row, 1)
        arr.append(row)
    return torch.cat(arr, 0)


def batch_block_matrix(block, num_blocks):
    # block: N x M
    # num_blocks = C
    # ret: N x C x M
    block = block.unsqueeze(1)
    arr = []
    for i in range(num_blocks):
        row = []
        for j in range(num_blocks):
            if i == j:
                row.append(block)
            else:
                row.append(torch.zeros_like(block))
        row = torch.cat(row, 2)
        arr.append(row)
    return torch.cat(arr, 1)
#############################################################################

# MDKTpgModelState = namedtuple(
#     "MDKTpgModelState",
#     ["mu_n_c_list", "Sigma_n_c_list", "logl_train_list","kernel","K","K_inv","x"],
# )

MDKTpgModelState = namedtuple(
    "MDKTpgModelState",
    ["N", "C", "mu", "K_block", "K", "L", "K_inv", "X", "Y", "kernel"],
)
MDKTPGState = namedtuple(
    "MDKTPGState", ["mu_n_c", "Sigma_n_c", "f_n_c", "alpha_n", "gamma_n_c","w_n_c"]
)

#MDKTPGState(mu_n_c_init, Sigma_n_c_init, f_n_c_init, alpha_n_init, gamma_n_c_init, w_n_c_init)


def block_matrix(block, num_blocks):
    arr = []
    for i in range(num_blocks):
        row = []
        for j in range(num_blocks):
            if i == j:
                row.append(block)
            else:
                row.append(torch.zeros_like(block))
        row = torch.cat(row, 1)
        arr.append(row)
    return torch.cat(arr, 0)


def batch_block_matrix(block, num_blocks):
    # block: N x M
    # num_blocks = C
    # ret: N x C x M
    block = block.unsqueeze(1)
    arr = []
    for i in range(num_blocks):
        row = []
        for j in range(num_blocks):
            if i == j:
                row.append(block)
            else:
                row.append(torch.zeros_like(block))
        row = torch.cat(row, 2)
        arr.append(row)
    return torch.cat(arr, 1)



################################################################################
class MDKT(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, fast_inference=False):
        super(MDKT, self).__init__(model_func, n_way, n_support)
        self.n_way = n_way
        self.kernel = load_kernel(n_way)
        self.noise = 0.001
        # self.ppg = pypolyagamma.PyPolyaGamma()
        self.num_steps = 100
        self.num_draws = 50
        # self.quad = GaussHermiteQuadrature1D()

        self.loss_fn = nn.CrossEntropyLoss()

        self.fast_inference = fast_inference

    ############################################

    def extract_dataset(self, X):
        # X: C x shot x D
        C = X.size(0)
        shot = X.size(1)

        Y = torch.repeat_interleave(torch.eye(C, device=X.device), shot, 0)#(C x shot) x C
        X = X.reshape(-1, X.size(-1)) #(C x shot) x D

        return X, Y

    def encode(self, x, is_feature=False):
        return self.parse_feature(x, is_feature)

    def merged_encode(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature=is_feature) 
        return torch.cat([z_support, z_query], 1)


    ############################################
    def multi_class_logl(self, y, f): # y and f are N*C matrixes
        # return sum(torch.log(expit(f[torch.where(y==1)[0],torch.where(y==1)[1]])))-sum(torch.log(torch.sum(expit(f),axis=1)))
        return sum(torch.log(torch.sigmoid(f[torch.where(y==1)[0],torch.where(y==1)[1]])))-sum(torch.log(torch.sum(torch.sigmoid(f),axis=1)))

    

    # def ELBO(self, model_state):
    #     x = model_state.x
    #     mu_n_c = model_state.mu_n_c_list[-1]
    #     Sigma_n_c = model_state.Sigma_n_c_list[-1]
    #     K = model_state.K
    #     K_inv = model_state.K_inv
    #     N=mu_n_c.shape[0]
    #     C=mu_n_c.shape[1]
    #     L=torch.slogdet(K)[1]*C
    #     # print()
    #     for c in range(C):
    #         L= L + torch.trace(torch.mm(K_inv,Sigma_n_c[c]+torch.outer(mu_n_c[:,c],mu_n_c[:,c])))
    #     return L


    def ELBO(self, model_state, pg_state):
        # X = model_state.X
        mu_n_c = pg_state.mu_n_c.detach()
        Sigma_n_c = pg_state.Sigma_n_c.detach()
        K = model_state.K[0]
        K_inv = model_state.K_inv
        C = model_state.C
        mu = model_state.mu
        L = torch.slogdet(K)[1]*C
        for c in range(C):
            ELBO= L + torch.trace(torch.mm(K_inv,Sigma_n_c[c]))+ torch.mul(torch.mv(K_inv, mu[:,c]-mu_n_c[:,c]).t(),mu[:,c]-mu_n_c[:,c]).sum()
        return -ELBO

    # def ELBO(self, model_state, pg_state):
    #     X = model_state.X
    #     Y = model_state.Y
    #     mu_n_c = pg_state.mu_n_c
    #     Sigma_n_c = pg_state.Sigma_n_c
    #     gamma_n_c = pg_state.gamma_n_c
    #     f_n_c = pg_state.f_n_c
    #     w_n_c = pg_state.w_n_c
    #     alpha_n = pg_state.alpha_n
    #     K_inv = model_state.K_inv
    #     K = model_state.K[0]
    #     N = model_state.N
    #     C = model_state.C
    #     mu = model_state.mu
    #     L = torch.slogdet(K)[1]

    #     ELBO = torch.sum(-(Y+gamma_n_c)*torch.log(torch.tensor([2], dtype=K.dtype, device=K.device))+(Y-gamma_n_c)/2*mu_n_c-w_n_c/2*(f_n_c)**2)\
    #     +torch.sum(alpha_n-torch.log(torch.tensor([C], dtype=K.dtype, device=K.device))+torch.log(torch.tensor(gamma(alpha_n.data.cpu().numpy()),device=K.device))+(1-alpha_n)*torch.digamma(alpha_n))\
    #     -torch.sum(gamma_n_c*(torch.log(gamma_n_c)-1)-gamma_n_c*((torch.digamma(alpha_n)-torch.log(torch.tensor([C], dtype=K.dtype, device=K.device))).reshape(-1,1)))-torch.sum(alpha_n/C)\
    #     -torch.sum(-f_n_c**2/2*w_n_c+(gamma_n_c+Y)*torch.log(torch.cosh(f_n_c/2)))

    #     for c in range(C):
    #         ELBO-=0.5*(L-torch.slogdet(Sigma_n_c[c])[1]- N + torch.trace(torch.mm(K_inv,Sigma_n_c[c]))+ torch.mul(torch.mv(K_inv, mu[:,c]-mu_n_c[:,c]).t(),mu[:,c]-mu_n_c[:,c]).sum())

    #     return -ELBO


    # def polya_gamma_conjugate_vb(self, x, y, num_iter=100): ## hyperparameters are theta0,theta1,noise_var
    #     # torch.set_default_tensor_type('torch.cuda')
    #     N=y.shape[0]

    #     C=self.n_way
    #     ## kernel matrix
    #     # K=expo_quad_kernel(theta0,theta1,x)
    #     mu = self.kernel.mean_function(x).reshape(x.size(0), self.n_way)
    #     K = self.kernel.cov_function(x)
    #     # K+=np.eye(N)*noise_var
    #     K = K + self.noise * torch.eye(K.size(-1), dtype=K.dtype, device=K.device)  # 85*85
    #     K_inv=torch.inverse(K)
    #     # print(K.shape, mu.shape) #torch.Size([85, 85]) torch.Size([85, 5])
    
    #     ## initial state
    #     # mu_n_c=torch.cuda.FloatTensor(N,C).uniform_(-4,4)
    #     mu_n_c = mu
    #     # mu_n_c=np.random.uniform(-4,4,size=(N,C))
    #     Sigma_n_c=torch.zeros((C,N,N), device=y.device)
    #     f_n_c=torch.zeros((N,C),device=y.device)
    #     for c in range(C):
    #         Sigma_n_c[c]=K
    #         f_n_c[:,c]=torch.sqrt(torch.diag(Sigma_n_c[c]))
    #     alpha_n=torch.ones(N, device=y.device)
    #     gnc=torch.tensor(psi(alpha_n.data.cpu().numpy()),device=K.device).reshape(N,1).expand(N,C)
    #     gamma_n_c=torch.exp(gnc-mu_n_c/2)/2/C/torch.cosh(f_n_c/2)    
    #     w_n_c=(gamma_n_c+y)/2/f_n_c*torch.tanh(f_n_c/2)

    #     w_n_c_list=[]
    #     alpha_n_list=[]
    #     gamma_n_c_list=[]
    #     f_n_c_list=[]
    #     mu_n_c_list=[]
    #     Sigma_n_c_list=[]
    #     logl_train_list=[]

    #     for _ in range(num_iter):
    #         ## update Sigma_c
    #         for c in range(C):
    #             Sigma_n_c[c]=torch.inverse(torch.diag(w_n_c[:,c])+K_inv)
            
    #         ## update mu_c
    #         for c in range(C):
    #             mu_n_c[:,c]=Sigma_n_c[c].mv(y[:,c]-gamma_n_c[:,c])/2

    #         ## update f_n_c
    #         f_n_c=torch.sqrt(mu_n_c**2+torch.tensor([np.diag(Sigma_n_c[c].data.cpu().numpy()) for c in range(C)], device=K.device).T)

    #         ## update gamma_n_c
    #         gnc=torch.tensor(psi(alpha_n.data.cpu().numpy()),device=K.device).reshape(N,1).expand(N,C)
    #         gamma_n_c=torch.exp(gnc-mu_n_c/2)/2/C/torch.cosh(f_n_c/2)  
            
    #         ## update alpha_n
    #         alpha_n=torch.sum(gamma_n_c,axis=1)+1
            
    #         ## update w_n_c
    #         w_n_c=(gamma_n_c+y)/2/f_n_c*torch.tanh(f_n_c/2)
            
    #         ## log-likelihood
    #         logl_train=self.multi_class_logl(y,mu_n_c)
            
    #         ## record all
    #         Sigma_n_c_list.append(Sigma_n_c)
    #         mu_n_c_list.append(mu_n_c)
    #         f_n_c_list.append(f_n_c)
    #         gamma_n_c_list.append(gamma_n_c)
    #         alpha_n_list.append(alpha_n)
    #         w_n_c_list.append(w_n_c)
    #         logl_train_list.append(logl_train)
    #     return MDKTpgModelState(
    #         mu_n_c_list=mu_n_c_list,Sigma_n_c_list=Sigma_n_c_list,logl_train_list=logl_train_list,kernel=self.kernel,K=K,K_inv=K_inv,x=x)

    def fit(self, X, Y): ## hyperparameters are theta0,theta1,noise_var
        # torch.set_default_tensor_type('torch.cuda')
        N = Y.shape[0]
        C = self.n_way
        ## kernel matrix
        # K=expo_quad_kernel(theta0,theta1,x)
        mu = self.kernel.mean_function(X).view(N, C) # N * C
        # K = self.kernel.cov_function(x)
        # # K+=np.eye(N)*noise_var
        # K = K + self.noise * torch.eye(K.size(-1), dtype=K.dtype, device=K.device)  # 85*85
        K_block = self.kernel.cov_function(X) # (NC*NC)
        K_block = K_block + self.noise * torch.eye(
            K_block.size(-1), dtype=K_block.dtype, device=K_block.device
        )
        K = K_block[:N, :N]
        L = psd_safe_cholesky(K)
        K_inv = torch.cholesky_solve(torch.eye(N, dtype=K_block.dtype, device=K_block.device), L) ######

        # K_inv=torch.inverse(K)
        # print(K.shape, mu.shape) #torch.Size([85, 85]) torch.Size([85, 5])
        K = K.unsqueeze(0).expand(C, N, N) # C*N*N

        return MDKTpgModelState(
            N=N,
            C=C,
            mu=mu,
            K_block=K_block,
            K=K,
            L=L,
            K_inv=K_inv,
            X=X,
            Y=Y,
            kernel=self.kernel,
        )

    def pg_update(self, model_state):
        pg_state = self.initial_pg_state(model_state)

        for _ in range(self.num_steps):
            pg_state = self.next_pg_state(model_state, pg_state)

            # print("************NEGELBO IS:************", self.ELBO(model_state, pg_state))

        return pg_state

    def initial_pg_state(self, model_state):
        mu_n_c_init = model_state.mu  # N*C
        Sigma_n_c_init = model_state.K # C*N*N
        f_n_c_init = torch.sqrt(torch.diag(model_state.K[0])).view(-1,1).repeat(1,model_state.C) # N*C
        alpha_n_init = torch.ones(model_state.N, device=model_state.Y.device)
        gnc = torch.tensor(psi(alpha_n_init.data.cpu().numpy()),device=model_state.Y.device).reshape(-1,1).expand(model_state.N,model_state.C)
        gamma_n_c_init = torch.exp(gnc-mu_n_c_init/2)/2/model_state.C/torch.cosh(f_n_c_init/2)    
        w_n_c_init = (gamma_n_c_init + model_state.Y)/2/f_n_c_init*torch.tanh(f_n_c_init/2)

        return MDKTPGState(mu_n_c_init, Sigma_n_c_init, f_n_c_init, alpha_n_init, gamma_n_c_init, w_n_c_init)


    def next_pg_state(self, model_state, pg_state):
        # Sigma_n_c = pg_state.Sigma_n_c
        # mu_n_c = pg_state.mu_n_c
        # Sigma_n_c_new = Sigma_n_c
        # mu_n_c_new = mu_n_c
        temp_sigma = torch.inverse(torch.diag(pg_state.w_n_c[:,0]) + model_state.K_inv)
        # print(temp_sigma.shape)
        for c in range(model_state.C - 1):
            temp_sigma = torch.cat((temp_sigma, torch.inverse(torch.diag(pg_state.w_n_c[:,c+1]) + model_state.K_inv)), 0)
            # Sigma_n_c_new[c,:,:] = torch.inverse(torch.diag(pg_state.w_n_c[:,c]) + model_state.K_inv)
            # mu_n_c_new[:,c] = Sigma_n_c_new[c].mv(model_state.Y[:,c]-pg_state.gamma_n_c[:,c])/2
        Sigma_n_c_new = temp_sigma.reshape(model_state.C, model_state.N, model_state.N)


        temp_mu = Sigma_n_c_new[0].mv(model_state.Y[:,0]-pg_state.gamma_n_c[:,0])/2
        for c in range(model_state.C - 1):
            temp_mu = torch.cat((temp_mu, Sigma_n_c_new[c+1].mv(model_state.Y[:,c+1]-pg_state.gamma_n_c[:,c+1])/2), 0)
        mu_n_c_new = temp_mu.reshape(model_state.C, model_state.N).T

        # for c in range(model_state.C):
        #     Sigma_n_c_new_c = torch.inverse(torch.diag(pg_state.w_n_c[:,c]) + model_state.K_inv)
        #     mu_n_c_new_c = Sigma_n_c_new_c.mv(model_state.Y[:,c]-pg_state.gamma_n_c[:,c])/2

        f_n_c_new = torch.sqrt(mu_n_c_new**2+torch.tensor([np.diag(Sigma_n_c_new[c].data.cpu().numpy()) for c in range(model_state.C)], device=model_state.K.device).T) ############

        gnc = torch.tensor(psi(pg_state.alpha_n.data.cpu().numpy()),device=model_state.K.device).reshape(-1,1).expand(model_state.N,model_state.C)
        gamma_n_c_new = torch.exp(gnc-mu_n_c_new/2)/2/model_state.C/torch.cosh(f_n_c_new/2)  # element-wise div

        alpha_n_new = torch.sum(gamma_n_c_new,axis=1)+1

        w_n_c_new = (gamma_n_c_new + model_state.Y)/2/f_n_c_new*torch.tanh(f_n_c_new/2)

        logl_train_new = self.multi_class_logl(model_state.Y, mu_n_c_new)

        return MDKTPGState(mu_n_c_new, Sigma_n_c_new, f_n_c_new, alpha_n_new, gamma_n_c_new, w_n_c_new)


    #############################################
    def predict_mu_sigma(self, z_query, model_state, pg_state):
        z_train = model_state.X
        # print(z_query.shape)
        # print(z_train.shape)
        k_qq = model_state.kernel.cov_function(z_query)[:z_query.shape[0], :z_query.shape[0]] #torch.Size([80, 80])
        k_qs = model_state.kernel.cov_function(z_query, z_train)[:z_query.shape[0], :z_train.shape[0]] #torch.Size([80, 25])
        k_inv_ss = model_state.K_inv  #torch.Size([25, 25])
        # k_sq = model_state.kernel.cov_function(z_train, z_query)
        k_sq = k_qs.T  #torch.Size([25, 80])
        sigma_s = pg_state.Sigma_n_c  # 5* 25*25
        mu_s = pg_state.mu_n_c  # 25*5
        # k_qq = model_state.kernel.cov_function(z_query) #torch.Size([80, 80])
        # k_qs = model_state.kernel.cov_function(z_query, z_train) #torch.Size([80, 25])
        # k_inv_ss = model_state.K_inv  #torch.Size([25, 25])
        # # k_sq = model_state.kernel.cov_function(z_train, z_query)
        # k_sq = k_qs.T  #torch.Size([25, 80])
        # sigma_s = model_state.Sigma_n_c_list[-1]  # 5* 25*25
        # mu_s = model_state.mu_n_c_list[-1]  # 25*5

        mu_pre = torch.matmul(k_qs, k_inv_ss).matmul(mu_s) #80*5 #len(z), self.n_way
        k_qq = torch.diag(k_qq) # 80*1

        sigma_pre = torch.zeros(len(z_query), self.n_way, device = k_inv_ss.device)
        for i in range(len(z_query)):
            for c in range(self.n_way):
                sigma_pre[i,c] = k_qq[i]+torch.matmul(k_qs[i,:], k_inv_ss).matmul(sigma_s[c].mm(k_inv_ss)-torch.eye(len(z_train), device=z_query.device)).matmul(k_sq[:,i])
        return mu_pre, sigma_pre  #80*5,  80*5

    def predict_mu_sigma_class(self, z_query, model_state, pg_state):
        z_train = model_state.X
        k_qq = model_state.kernel.batch_cov_function(z_query) #torch.Size([80, 80])
        k_qs = model_state.kernel.batch_cov_function(z_query, z_train) #torch.Size([80, 25])
        k_inv_ss = model_state.K_inv  #torch.Size([25, 25])
        # k_sq = model_state.kernel.cov_function(z_train, z_query)
        k_sq = k_qs.T  #torch.Size([25, 80])
        sigma_s = pg_state.Sigma_n_c  # 5* 25*25
        mu_s = pg_state.mu_n_c  # 25*5

        mu_pre = torch.matmul(k_qs, k_inv_ss).matmul(mu_s) #80*5 #len(z), self.n_way
        k_qq = torch.diag(k_qq) # 80*1

        sigma_pre = torch.zeros(len(z_query), self.n_way, device = k_inv_ss.device)
        for i in range(len(z_query)):
            for c in range(self.n_way):
                sigma_pre[i,c] = k_qq[i]+torch.matmul(k_qs[i,:], k_inv_ss).matmul(sigma_s[c].mm(k_inv_ss)-torch.eye(len(z_train), device=z_query.device)).matmul(k_sq[:,i])
        return mu_pre, sigma_pre  #80*5,  80*5

    
    def logistic_softmax(self, f_new):#f_new 80*5
        ff = torch.sum(torch.sigmoid(f_new),dim=1).reshape(-1,1).repeat(1,self.n_way)#self.n_way
        prob = torch.sigmoid(f_new).div(ff)
        return prob

    ########################################################

    def set_forward(self, X, is_feature=False, verbose=False):
        X_support, X_query = self.encode(X, is_feature=is_feature)

        X_support, Y_support = self.extract_dataset(X_support)
        X_query, Y_query = self.extract_dataset(X_query)

        # model_state = self.polya_gamma_conjugate_vb(X, Y, num_iter=self.num_steps)
        model_state = self.fit(X_support, Y_support)

        pg_state = self.pg_update(model_state)

        # model_state = self.polya_gamma_conjugate_vb(X_support, Y_support, num_iter=self.num_steps)

        mu_pre, sigma_pre = self.predict_mu_sigma(X_query, model_state, pg_state)
        # print(sigma_pre) #mu_pre: 85*5,  sigma_pre: 85*5

        
        # # 1st form (our sampling + our logistic softmax)
        # prob = []
        # for i in range(self.num_draws):
        #     # print(mu_pre.shape)
        #     # print(sigma_pre.shape)
        #     mu_new = mu_pre.contiguous().view(1,-1)
        #     sigma_new = sigma_pre.contiguous().view(1,-1).squeeze(0).diag()
        #     # f_new = MultivariateNormal(mu_new, sigma_new).sample()  #85*5  sampling 50 samples
        #     f_new = MultivariateNormal(mu_new, scale_tril=psd_safe_cholesky(sigma_new)).sample()  #85*5  sampling 50 samples
        #     prob_i = self.logistic_softmax(f_new.contiguous().view(-1,self.n_way)) #85*5
        #     prob.append(prob_i.cpu().detach().numpy())
        # # y_pred = np.mean(prob,axis=0).argmax(axis=1) #85*5
        # y_pred = np.mean(prob,axis=0)
        # y_pred = torch.from_numpy(y_pred).cuda()


        ## 2nd form (ove sampling + logistic softmax)
        sigma_final = torch.zeros(sigma_pre.shape[0],sigma_pre.shape[1],sigma_pre.shape[1], dtype = sigma_pre.dtype, device = sigma_pre.device)
        for i in range(sigma_pre.shape[0]):
            sigma_final[i] = sigma_pre[i].diag()
        f_post = MultivariateNormal(mu_pre, scale_tril=psd_safe_cholesky(sigma_final)) #mu_pre: 85*5,  sigma_pre: 85*5*5
        f_samples = f_post.rsample((self.num_draws,)) #20*85*5
        # f_samples = f_samples.reshape(-1, *Y_query.size())
        y_pred = F.log_softmax(F.logsigmoid(f_samples).mean(0), -1)


        # # 3rd form (our sampling + logistic softmax)
        # mu_new = mu_pre.contiguous().view(1,-1).squeeze(0)
        # sigma_new = sigma_pre.contiguous().view(1,-1).squeeze(0).diag()
        # f_new = MultivariateNormal(mu_new, scale_tril=psd_safe_cholesky(sigma_new)).rsample((self.num_draws,)) # mu_new: (85*5); sigma_new: (85*5)*(85*5); f_new:20*425
        # f_samples = f_new.contiguous().view(self.num_draws, -1, self.n_way)
        # y_pred = F.log_softmax(F.logsigmoid(f_samples).mean(0), -1)

        # # 4th form (ove sampling + our logistic softmax)
        # sigma_final = torch.zeros(sigma_pre.shape[0],sigma_pre.shape[1],sigma_pre.shape[1], dtype = sigma_pre.dtype, device = sigma_pre.device)
        # for i in range(sigma_pre.shape[0]):
        #     sigma_final[i] = sigma_pre[i].diag()
        # f_post = MultivariateNormal(mu_pre, scale_tril=psd_safe_cholesky(sigma_final)) #mu_pre: 85*5,  sigma_pre: 85*5*5
        # f_samples = f_post.rsample((self.num_draws,)) #20*85*5
        # y_pred_all = f_samples
        # for i in range(self.num_draws):
        #     y_pred_all[i] = self.logistic_softmax(f_samples[i])
        # y_pred = y_pred_all.mean(0) # 85*5

        return y_pred

    def set_forward_loss(self, X): #x:tensor, y:tensor
        #x: torch.Size([85, 1600]) torch.float32
        #y: torch.Size([85]) torch.int64
        X, Y = self.extract_dataset(self.merged_encode(X)) # torch.Size([105, 1600]) torch.Size([105, 5])

        # model_state = self.polya_gamma_conjugate_vb(X, Y, num_iter=self.num_steps)
        model_state = self.fit(X, Y)

        pg_state = self.pg_update(model_state)
        #####
        # max prediction
        y_pred = pg_state.mu_n_c.argmax(axis=1) # 1*85   array([0, 1, 2, 3, 4])
        y_support = Y.argmax(axis=1)
        accuracy = (torch.sum(y_pred==y_support) / float(len(y_support))) * 100.0
        print("accuracy is:", accuracy)

        # f prediction
        # mu_pre, sigma_pre = self.predict_mu_sigma(X, model_state)
        # prob = []
        # for i in range(self.num_draws):
        #     mu_new = mu_pre.contiguous().view(1,-1)
        #     sigma_new = sigma_pre.contiguous().view(1,-1).squeeze(0).diag()
        #     # f_new = MultivariateNormal(mu_new, sigma_new).sample()  #85*5  sampling 50 samples
        #     f_new = MultivariateNormal(mu_new, scale_tril=psd_safe_cholesky(sigma_new)).sample()  #85*5  sampling 50 samples
        #     prob_i = self.logistic_softmax(f_new.contiguous().view(-1,self.n_way)) #85*5
        #     prob.append(prob_i.cpu().detach().numpy())
        # # y_pred = np.mean(prob,axis=0).argmax(axis=1) #85*5
        # y_pred = np.mean(prob,axis=0)
        # y_pred = torch.from_numpy(y_pred).cuda().argmax(axis=1)
        # y_support = Y.argmax(axis=1)
        # accuracy = (torch.sum(y_pred==y_support) / float(len(y_support))) * 100.0
        # print("accuracy is:", accuracy)
        # ####
        return self.ELBO(model_state, pg_state)

class PredictiveMDKT(MDKT):
    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()
        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query)



        


