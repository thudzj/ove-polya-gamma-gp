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

from methods.meta_template import MetaTemplate
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
            self.register_parameter("output_scale_raw", nn.Parameter(torch.tensor([math.log(math.e**1-1)])))
            self.register_parameter("mean_value", nn.Parameter(torch.zeros(1)))###################
        else:
            self.register_buffer("output_scale_raw", torch.tensor([math.log(math.e**1-1)]))
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
        return F.softplus(self.output_scale_raw) * (x1 @ x2.T)  # Linear Kernel Eq.48


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
            self.register_parameter("output_scale_raw", nn.Parameter(torch.Tensor([math.log(math.e**1-1)])))
        else:
            self.register_buffer("output_scale_raw", torch.Tensor([math.log(math.e**1-1)]))

    def mean_function(self, X):
        # return torch.zeros(X.size(0) * self.num_classes, dtype=X.dtype, device=X.device)
        return torch.ones(X.size(0) * self.num_classes, dtype=X.dtype, device=X.device)

    def normalize(self, X):
        D = X.size(-1)
        return X / math.sqrt(D)

    def cov_block(self, x1, x2=None):
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)
        return F.softplus(self.output_scale_raw) * (x1.mm(x2.t()) ** 2)


class L2QuadraticKernel(QuadraticKernel):
    def normalize(self, X):
        return F.normalize(X)    # 3rd kernel: L2 Quadratic Kernel


class RBFKernel(Kernel):  # RBF Kernel: output_scale_raw, lengthscale_raw, nn.param.
    def __init__(self, *args, **kwargs):
        super(RBFKernel, self).__init__(*args, **kwargs)
        if self.learn_params:
            self.register_parameter("output_scale_raw", nn.Parameter(torch.Tensor([math.log(math.e**1-1)])))
            self.register_parameter("lengthscale_raw", nn.Parameter(torch.zeros(1)))
        else:
            self.register_buffer("output_scale_raw", torch.Tensor([math.log(math.e**1-1)]))
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
        # print()
        return F.softplus(self.output_scale_raw) * torch.exp(
            -0.5 * dists / torch.exp(self.lengthscale_raw)
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

MDKTpgModelState = namedtuple(
    "MDKTpgModelState",
    ["N", "C", "K", "K_inv", "X", "Y", "kernel"],
)
MDKTPGState = namedtuple(
    "MDKTPGState", ["mu_n_c", "Sigma_n_c", "f_n_c", "alpha_n", "gamma_n_c","w_n_c"]
)

################################################################################
class MDKT(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, fast_inference=False):
        super(MDKT, self).__init__(model_func, n_way, n_support)
        self.n_way = n_way
        self.kernel = load_kernel(n_way)
        self.register_parameter("noise", nn.Parameter(torch.Tensor([math.log(math.e**0.001 - 1)])))
        # self.register_buffer("noise", torch.Tensor([math.log(math.e**0.001 - 1)]))
        # self.noise = nn.Parameter(torch.Tensor([math.log(math.e**0.001 - 1)]), require_grad=False)
        # self.register_parameter("a", nn.Parameter(torch.Tensor([0.])))
        self.register_buffer("a", torch.Tensor([0.]))
        self.num_steps = 100
        self.num_draws = 50
        self.fast_inference = fast_inference
        self.bpti = 0

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
    # def multi_class_logl(self, y, f): # y and f are N*C matrixes
    #     p_unnormed = f.sigmoid()
    #     p = p_unnormed / p_unnormed.sum(-1, keepdim=True)
    #     return (y * p).sum(-1).log().sum()

    def ELBO(self, model_state, pg_state):
        K = model_state.K
        C = model_state.C
        X = model_state.X
        Y = model_state.Y
        mu_n_c = pg_state.mu_n_c
        Sigma_n_c = pg_state.Sigma_n_c
        if not self.bpti: # todo
            mu_n_c = mu_n_c.detach()
            Sigma_n_c = Sigma_n_c.detach()
            prior = MultivariateNormal(self.a.expand(K.shape[0]), K)
            KL_all = 0
            for c in range(C):
                approx_posterior = MultivariateNormal(mu_n_c[:,c],
                    Sigma_n_c[c] + 1e-4 * torch.eye(Sigma_n_c[c].shape[0], device=Sigma_n_c[c].device))
                KL_all += torch.distributions.kl.kl_divergence(approx_posterior, prior)
            return KL_all
        else:
            #todo add the a
            prior = MultivariateNormal(torch.zeros(K.shape[0], device=K.device), K)
            KL_all = 0
            for c in range(C):
                approx_posterior = MultivariateNormal(mu_n_c[:,c], Sigma_n_c[c])
                KL_all += torch.distributions.kl.kl_divergence(approx_posterior, prior)

            others1 = (-(Y+pg_state.gamma_n_c)*math.log(2.)
                        + (Y-pg_state.gamma_n_c)/2.*mu_n_c - pg_state.f_n_c**2 / 2. * pg_state.w_n_c).sum()

            others2 = (pg_state.alpha_n - math.log(C) + pg_state.alpha_n.lgamma()
                        + (1 - pg_state.alpha_n) * pg_state.alpha_n.digamma()).sum()

            others3 = -(pg_state.gamma_n_c * (pg_state.gamma_n_c.log() - 1
                            - pg_state.alpha_n.digamma().view(-1, 1) + math.log(C))).sum() \
                       - pg_state.alpha_n.sum()

            others4 = (pg_state.f_n_c**2 / 2. * pg_state.w_n_c
                       - (pg_state.gamma_n_c + Y) * torch.log(torch.cosh(pg_state.f_n_c/2))).sum()

            others = -others1 - others2 - others3 - others4

            return KL_all + others


    def fit(self, X, Y):
        K = self.kernel.cov_block_wrapper(X)
        if self.noise is not None:
            K = K + F.softplus(self.noise).add(0.00003) * torch.eye(X.size(0), dtype=X.dtype, device=X.device)
        return MDKTpgModelState(
            N=Y.shape[0],
            C=self.n_way,
            K=K,
            K_inv=K.inverse(),
            X=X,
            Y=Y,
            kernel=self.kernel,
        )

    def pg_update(self, model_state, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps
        pg_state = self.initial_pg_state(model_state)
        for _ in range(num_steps):
            # print("\n inner", self.multi_class_logl(model_state.Y, pg_state.mu_n_c).item(), (pg_state.mu_n_c.argmax(axis=1)==model_state.Y.argmax(axis=1)).float().mean(), self.ELBO(model_state, pg_state).item())

            pg_state = self.next_pg_state(model_state, pg_state)
            # print("************NEGELBO IS:************", self.ELBO(model_state, pg_state))
        return pg_state

    def initial_pg_state(self, model_state):
        # todo random init
        mu_n_c_init = None #torch.zeros(model_state.N, model_state.C, device=model_state.Y.device) # N*C
        Sigma_n_c_init = None #model_state.K.detach().clone().expand(model_state.C,model_state.N,model_state.N) # C*N*N
        f_n_c_init = torch.diag(model_state.K).sqrt().view(-1, 1).expand(model_state.N,model_state.C) # N*C
        alpha_n_init = torch.ones(model_state.N, device=model_state.Y.device)
        gnc = alpha_n_init.digamma().view(-1, 1).expand(model_state.N,model_state.C)
        gamma_n_c_init = torch.exp(gnc)/2/model_state.C/torch.cosh(f_n_c_init/2)
        w_n_c_init = (gamma_n_c_init + model_state.Y)/2/f_n_c_init*torch.tanh(f_n_c_init/2)
        return MDKTPGState(mu_n_c_init, Sigma_n_c_init, f_n_c_init, alpha_n_init, gamma_n_c_init, w_n_c_init)


    def next_pg_state(self, model_state, pg_state):
        # todo check
        Sigma_n_c_new = (pg_state.w_n_c.T.diag_embed() + model_state.K_inv).inverse()

        mu_n_c_new = (Sigma_n_c_new @ (model_state.Y - pg_state.gamma_n_c).T.unsqueeze(-1)).squeeze().T / 2. \
                    + (Sigma_n_c_new @ model_state.K_inv).sum(-1).T * self.a

        f_n_c_new = (mu_n_c_new**2 + torch.diagonal(Sigma_n_c_new, offset=0, dim1=1, dim2=2).T).sqrt()

        gnc = pg_state.alpha_n.digamma().view(-1, 1).expand(model_state.N,model_state.C)

        gamma_n_c_new = torch.exp(gnc-mu_n_c_new/2)/2/model_state.C/torch.cosh(f_n_c_new/2)  # element-wise div

        alpha_n_new = torch.sum(gamma_n_c_new, axis=1) + 1

        w_n_c_new = (gamma_n_c_new + model_state.Y)/2/f_n_c_new * torch.tanh(f_n_c_new/2)

        return MDKTPGState(mu_n_c_new, Sigma_n_c_new, f_n_c_new, alpha_n_new, gamma_n_c_new, w_n_c_new)

    def predict_mu_sigma(self, z_query, model_state, pg_state):
        z_train = model_state.X
        k_qq = model_state.kernel.cov_block_wrapper(z_query)
        k_qs = model_state.kernel.cov_block_wrapper(z_query, z_train)
        k_inv_ss = model_state.K_inv

        k_sq = k_qs.T
        sigma_s = pg_state.Sigma_n_c  # 5* 25*25
        mu_s = pg_state.mu_n_c  # 25*5
        if not self.bpti:
            sigma_s = sigma_s.detach()
            mu_s = mu_s.detach()

        mu_pre = k_qs @ k_inv_ss @ mu_s #80*5 #len(z), self.n_way
        Sigma_pre = k_qq + k_qs @ k_inv_ss @ (sigma_s @ k_inv_ss - torch.eye(len(z_train), device=z_query.device)) @ k_sq
        sigma_pre = torch.diagonal(Sigma_pre, offset=0, dim1=1, dim2=2).T.sqrt() # 80 * 5

        # print(torch.stack([mu_s[0], sigma_s[:, 0, 0], mu_pre[0], sigma_pre[0]]).data.cpu().numpy())
        return mu_pre, sigma_pre  #80*5,  80*5

    def set_forward(self, X, is_feature=False, verbose=False, return_all_samples=False, mean_prob=False):
        X_support, X_query = self.encode(X, is_feature=is_feature)

        X_support, Y_support = self.extract_dataset(X_support)
        X_query, Y_query = self.extract_dataset(X_query)

        model_state = self.fit(X_support, Y_support)
        pg_state = self.pg_update(model_state)
        mu_pre, sigma_pre = self.predict_mu_sigma(X_query, model_state, pg_state)
        f_samples = mu_pre + sigma_pre * torch.randn(self.num_draws, *mu_pre.shape, device=mu_pre.device)

        # print((pg_state.mu_n_c.argmax(axis=1)==Y_support.argmax(axis=1)).float().mean().item(), (mu_pre.argmax(axis=1)==Y_query.argmax(axis=1)).float().mean().item())

        if return_all_samples:
            return F.logsigmoid(f_samples).log_softmax(-1)
        elif mean_prob:
            return F.logsigmoid(f_samples).softmax(-1).mean(0).log()
        else:
            return F.logsigmoid(mu_pre).log_softmax(-1), pg_state.mu_n_c[0], torch.diagonal(pg_state.Sigma_n_c, offset=0, dim1=1, dim2=2).T.sqrt()[0], mu_pre[0], sigma_pre[0] #  #

    def set_forward_loss(self, X): #x:tensor, y:tensor
        X, Y = self.extract_dataset(self.merged_encode(X))

        model_state = self.fit(X, Y)
        pg_state = self.pg_update(model_state)
        #####
        # max prediction
        # y_pred = pg_state.mu_n_c.argmax(axis=1) # 1*85   array([0, 1, 2, 3, 4])
        # y_support = Y.argmax(axis=1)
        # accuracy = (torch.sum(y_pred==y_support) / float(len(y_support))) * 100.0
        # print(accuracy.item(), F.softplus(self.noise).item(), F.softplus(self.kernel.output_scale_raw).item())
        return self.ELBO(model_state, pg_state)

class PredictiveMDKT(MDKT):
    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()
        # scores = self.set_forward(x, return_all_samples=True)
        # return F.nll_loss(scores.flatten(0, 1), y_query.repeat(scores.shape[0]))
        return F.nll_loss(self.set_forward(x, mean_prob=True), y_query)
