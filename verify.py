import sys
import copy
from collections import namedtuple
import warnings

import math
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import expit
from scipy.special import psi
from scipy.special import gamma
from scipy.stats import multinomial
from scipy.optimize import minimize
from autograd import grad
from autograd import jacobian
import torch
from sacred import Ingredient

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal

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
            self.register_parameter("output_scale_raw", nn.Parameter(torch.tensor([0.])))
            self.register_parameter("mean_value", nn.Parameter(torch.zeros(1)))###################
        else:
            self.register_buffer("output_scale_raw", torch.tensor([0.]))
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
        return torch.exp(self.output_scale_raw) * (x1 @ x2.T)  # Linear Kernel Eq.48


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

MDKTpgModelState = namedtuple(
    "MDKTpgModelState",
    ["N", "C", "K", "K_inv", "X", "Y", "kernel"],
)
MDKTPGState = namedtuple(
    "MDKTPGState", ["mu_n_c", "Sigma_n_c", "f_n_c", "alpha_n", "gamma_n_c","w_n_c"]
)

np.random.seed(0)
x=np.array(list(np.linspace(0+25/350/2,25-25/350/2,350))+list(np.linspace(35+25/350/2,60-25/350/2,350))+list(np.linspace(70+25/350/2,95-25/350/2,350)))
x=x.reshape(len(x),1)
y=np.zeros((len(x),3))
for i in range(len(x)):
    if i<=350:
        y[i]=multinomial(n=1,p=np.array([0.9,0.05,0.05])).rvs()[0]
    elif i<=700:
        y[i]=multinomial(n=1,p=np.array([0.05,0.9,0.05])).rvs()[0]
    else:
        y[i]=multinomial(n=1,p=np.array([0.05,0.05,0.9])).rvs()[0]

y_test=np.zeros((len(x),3))
for i in range(len(x)):
    if i<=350:
        y_test[i]=multinomial(n=1,p=np.array([0.9,0.05,0.05])).rvs()[0]
    elif i<=700:
        y_test[i]=multinomial(n=1,p=np.array([0.05,0.9,0.05])).rvs()[0]
    else:
        y_test[i]=multinomial(n=1,p=np.array([0.05,0.05,0.9])).rvs()[0]

def neg_ELBO(K,a,x,y,mu_n_c,Sigma_n_c,f_n_c,gamma_n_c,alpha_n,w_n_c):
    assert len(a)==len(x)
    N=mu_n_c.shape[0]
    C=mu_n_c.shape[1]
    K_inv=np.linalg.inv(K)
    ELBO=np.sum(-(y+gamma_n_c)*np.log(2)+(y-gamma_n_c)/2*mu_n_c-w_n_c/2*(f_n_c)**2)\
    +np.sum(alpha_n-np.log(C)+np.log(gamma(alpha_n))+(1-alpha_n)*psi(alpha_n))\
    -np.sum(gamma_n_c*(np.log(gamma_n_c)-1)-gamma_n_c*((psi(alpha_n)-np.log(C)).reshape(N,1)))-np.sum(alpha_n/C)\
    -np.sum(-f_n_c**2/2*w_n_c+(gamma_n_c+y)*np.log(np.cosh(f_n_c/2)))
    for c in range(C):
        ELBO-=0.5*(np.linalg.slogdet(K)[1]-np.linalg.slogdet(Sigma_n_c[c])[1]-N+np.trace(np.dot(K_inv,Sigma_n_c[c]))+np.dot(np.dot(a-mu_n_c[:,c],K_inv),a-mu_n_c[:,c]))
    return -ELBO


def polya_gamma_conjugate_vb(x,y,theta0,theta1,noise_var,a,num_iter): ## hyperparameters are theta0,theta1,noise_var, a is mean
    N=y.shape[0]
    C=y.shape[1]
    ## kernel matrix
    kernel = LinearKernel(C, learn_params=True)
    K=kernel.cov_block_wrapper(torch.from_numpy(x)).detach().numpy() + np.eye(N)*0.001 ##### NOTICE
    K_inv=np.linalg.inv(K)

    model_state = MDKTpgModelState(
        N=N,
        C=C,
        K=torch.from_numpy(K),
        K_inv=torch.from_numpy(K).inverse(),
        X=x,
        Y=torch.from_numpy(y),
        kernel=kernel,
    )

    ## initial state
    mu_n_c=np.zeros((N,C))
    Sigma_n_c=np.zeros((C,N,N))
    f_n_c=np.zeros((N,C))
    for c in range(C):
        Sigma_n_c[c]=K
        f_n_c[:,c]=np.sqrt(np.diag(Sigma_n_c[c]))
    alpha_n=np.ones(N)
    gamma_n_c=np.exp(np.tile(psi(alpha_n).reshape(N,1),(1,C))-mu_n_c/2)/2/C/np.cosh(f_n_c/2)
    w_n_c=(gamma_n_c+y)/2/f_n_c*np.tanh(f_n_c/2)

    mu_n_c_init = None #torch.zeros(model_state.N, model_state.C, device=model_state.Y.device) # N*C
    Sigma_n_c_init = None #model_state.K.detach().clone().expand(model_state.C,model_state.N,model_state.N) # C*N*N
    f_n_c_init = torch.diag(model_state.K).sqrt().view(-1, 1).expand(model_state.N,model_state.C) # N*C
    alpha_n_init = torch.ones(model_state.N)
    gnc = alpha_n_init.digamma().view(-1, 1).expand(model_state.N,model_state.C)
    gamma_n_c_init = torch.exp(gnc)/2/model_state.C/torch.cosh(f_n_c_init/2)
    w_n_c_init = (gamma_n_c_init + model_state.Y)/2/f_n_c_init*torch.tanh(f_n_c_init/2)
    pg_state = MDKTPGState(mu_n_c_init, Sigma_n_c_init, f_n_c_init, alpha_n_init, gamma_n_c_init, w_n_c_init)



    w_n_c_list=[]
    alpha_n_list=[]
    gamma_n_c_list=[]
    f_n_c_list=[]
    mu_n_c_list=[]
    Sigma_n_c_list=[]
    logl_train_list=[]
    logl_test_list=[]
    neg_ELBO_list=[]
    accuracy_train_list=[]
    accuracy_test_list=[]
    theta0_list=[]
    theta1_list=[]
    noise_var_list=[]
    a_list=[]

    for iteration in range(num_iter):
        ## update Sigma_c
        for c in range(C):
            Sigma_n_c[c]=np.linalg.inv(np.diag(w_n_c[:,c])+K_inv)

        Sigma_n_c_new = (pg_state.w_n_c.T.diag_embed() + model_state.K_inv).inverse()


        ## update mu_c
        for c in range(C):
            mu_n_c[:,c]=Sigma_n_c[c].dot(y[:,c]-gamma_n_c[:,c])/2+Sigma_n_c[c].dot(K_inv).dot(np.ones(N) * 0.1)
        # mu_n_c_new = (Sigma_n_c_new @ (model_state.Y - pg_state.gamma_n_c).T.unsqueeze(-1)).squeeze().T / 2.
        print(Sigma_n_c_new.shape, model_state.K_inv.shape)
        mu_n_c_new = (Sigma_n_c_new @ (model_state.Y - pg_state.gamma_n_c).T.unsqueeze(-1)).squeeze().T / 2. \
                    + (Sigma_n_c_new @ model_state.K_inv).sum(-1).T*0.1
        print(torch.dist(torch.from_numpy(mu_n_c), mu_n_c_new))
        exit()



        ## update f_n_c
        f_n_c=np.sqrt(mu_n_c**2+np.array([np.diag(Sigma_n_c[c]) for c in range(C)]).T)

        f_n_c_new = (mu_n_c_new**2 + torch.diagonal(Sigma_n_c_new, offset=0, dim1=1, dim2=2).T).sqrt()



        ## update gamma_n_c
        gamma_n_c=np.exp(np.tile(psi(alpha_n).reshape(N,1),(1,C))-mu_n_c/2)/2/C/np.cosh(f_n_c/2)

        gnc = pg_state.alpha_n.digamma().view(-1, 1).expand(model_state.N,model_state.C)

        gamma_n_c_new = torch.exp(gnc-mu_n_c_new/2)/2/model_state.C/torch.cosh(f_n_c_new/2)  # element-wise div



        ## update alpha_n
        alpha_n=np.sum(gamma_n_c,axis=1)+1

        alpha_n_new = torch.sum(gamma_n_c_new, axis=1) + 1




        print(
        torch.dist(torch.from_numpy(alpha_n), alpha_n_new))


        ## update w_n_c
        w_n_c=(gamma_n_c+y)/2/f_n_c*np.tanh(f_n_c/2)

        w_n_c_new = (gamma_n_c_new + model_state.Y)/2/f_n_c_new * torch.tanh(f_n_c_new/2)

        print(
        torch.dist(torch.from_numpy(w_n_c), w_n_c_new))

        pg_state = MDKTPGState(mu_n_c_new, Sigma_n_c_new, f_n_c_new, alpha_n_new, gamma_n_c_new, w_n_c_new)


        elbo0=np.sum(-(y+gamma_n_c)*np.log(2)+(y-gamma_n_c)/2*mu_n_c-w_n_c/2*(f_n_c)**2)\
        +np.sum(alpha_n-np.log(C)+np.log(gamma(alpha_n))+(1-alpha_n)*psi(alpha_n))\
        -np.sum(gamma_n_c*(np.log(gamma_n_c)-1)-gamma_n_c*((psi(alpha_n)-np.log(C)).reshape(N,1)))-np.sum(alpha_n)\
        -np.sum(-f_n_c**2/2*w_n_c+(gamma_n_c+y)*np.log(np.cosh(f_n_c/2)))
        for c in range(C):
            elbo0-=0.5*(np.linalg.slogdet(K)[1]-np.linalg.slogdet(Sigma_n_c[c])[1]-N+np.trace(np.dot(K_inv,Sigma_n_c[c]))+np.dot(np.dot(a-mu_n_c[:,c],K_inv),a-mu_n_c[:,c]))




        prior = MultivariateNormal(torch.zeros(model_state.K.shape[0], device=model_state.K.device), model_state.K)
        KL_all = 0
        for c in range(C):
            approx_posterior = MultivariateNormal(pg_state.mu_n_c[:,c], pg_state.Sigma_n_c[c])
            KL_all += torch.distributions.kl.kl_divergence(approx_posterior, prior)

        others1 = (-(model_state.Y+pg_state.gamma_n_c)*math.log(2.)
                    + (model_state.Y-pg_state.gamma_n_c)/2.*pg_state.mu_n_c - pg_state.f_n_c**2 / 2. * pg_state.w_n_c).sum()

        others2 = (pg_state.alpha_n - math.log(C) + pg_state.alpha_n.lgamma()
                    + (1 - pg_state.alpha_n) * pg_state.alpha_n.digamma()).sum()

        others3 = -(pg_state.gamma_n_c * (pg_state.gamma_n_c.log() - 1
                        - pg_state.alpha_n.digamma().view(-1, 1) + math.log(C))).sum() \
                   - pg_state.alpha_n.sum()

        others4 = (pg_state.f_n_c**2 / 2. * pg_state.w_n_c
                   - (pg_state.gamma_n_c + model_state.Y) * torch.log(torch.cosh(pg_state.f_n_c/2))).sum()

        elbo1 = -(others1 + others2 + others3 + others4 - KL_all)

        print(elbo0, elbo1)
        exit()

        ## update kernel
        initial_guss=np.array([theta0,theta1,noise_var]+list(a))
        res=minimize(neg_ELBO,initial_guss, args=(x,y,mu_n_c,Sigma_n_c,f_n_c,gamma_n_c,alpha_n,w_n_c),jac=gradient_hp,\
            method='SLSQP', bounds=[(1e-2,None),(1e-2,None),(1e-5,None)]+[(None,None)]*N, options={'maxiter': 5})
        theta0=res.x[0]
        theta1=res.x[1]
        noise_var=res.x[2]
        a=res.x[3:]
        K=expo_quad_kernel(theta0,theta1,noise_var,x)
        K_inv=np.linalg.inv(K)

        ## log-likelihood
        logl_train=multi_class_logl(y,mu_n_c)
        logl_test=multi_class_logl(y_test,mu_n_c)

        ## -ELBO
        neg_ELBO_list.append(neg_ELBO(np.array([theta0,theta1,noise_var]+list(a)),x,y,mu_n_c,Sigma_n_c,f_n_c,gamma_n_c,alpha_n,w_n_c))

        ## accuracy
        accuracy_train=accuracy_evaluate(y,mu_n_c)
        accuracy_test=accuracy_evaluate(y_test,mu_n_c)

        ## record all
        Sigma_n_c_list.append(Sigma_n_c)
        mu_n_c_list.append(mu_n_c)
        f_n_c_list.append(f_n_c)
        gamma_n_c_list.append(gamma_n_c)
        alpha_n_list.append(alpha_n)
        w_n_c_list.append(w_n_c)
        logl_train_list.append(logl_train)
        logl_test_list.append(logl_test)
        accuracy_train_list.append(accuracy_train)
        accuracy_test_list.append(accuracy_test)
        theta0_list.append(theta0)
        theta1_list.append(theta1)
        noise_var_list.append(noise_var)
        a_list.append(a)
    return mu_n_c_list,Sigma_n_c_list,logl_train_list,logl_test_list,accuracy_train_list,accuracy_test_list,\
           theta0_list,theta1_list,noise_var_list,a_list,neg_ELBO_list

polya_gamma_conjugate_vb(x,y,1,1,0.001,np.zeros(len(x)),100)
