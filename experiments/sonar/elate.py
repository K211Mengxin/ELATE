#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax.numpy as jnp
import sys
sys.path.insert(0, '/Users/k21163353/anaconda3/lib/python3.11/site-packages')
import particles
from particles import smc_samplers as ssp
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
from monte import monte_est
from jax.scipy.special import logsumexp
import numpy as np
from jax import vmap
from it import it_ls,it_dev,it_var
from elate import run_gp
import matplotlib.pyplot as plt
from particles.distributions import ProbDist


# In[2]:


#get estimators from smc, it, elate+smc,elate+it

#read smc samples
data=np.load('sonar.npy')
smc_data=jnp.load('smc_samples.npz',allow_pickle=True)
tem=np.array(smc_data['tem'])
samp=np.array(smc_data['samp'])
wei=np.array(smc_data['W'])
resample_N=np.array(smc_data['re_N'])
phi=lambda x: np.mean(x,axis=1)


# In[3]:


samp[0].theta[:,0].shape


# In[4]:


class GaussianPrior(ProbDist):
    def __init__(self, scale):
        self.scale = np.array(scale)
        self.var = self.scale**2
        self.cst = -1/2 * np.sum(np.log(2 * np.pi * self.var))

    def logpdf(self, x):
        #tested
        return self.cst - np.sum(x**2/(2*self.var), axis=1)

    def ppf(self, u):
        raise NotImplementedError

    @property
    def dim(self):
        return len(self.scale)

    def rvs(self, size=None):
        #tested
        if size is None:
            size = 1
        return np.random.normal(size=(size, len(self.scale))) * self.scale

def log_logit(arr):
    # tested for correctness
    # a little bit faster than old versions.
    """
    :return: -log(1+exp(-arr)), calculated safely
    """
    delta = jnp.abs(arr)
    return 1/2 * (arr - delta) - jnp.log1p(jnp.exp(-delta))
    
class ToyModel(ssp.StaticModel):
    def __init__(self, data):
        self.d = data.shape[1] - 1
        self.cpu = 0
        scale = [20] + [5] * (self.d - 1)
        prior = GaussianPrior(scale=scale)
        self.predictors_T = data[:, 1:].T
        self.response = data[:, 0]
        ssp.StaticModel.__init__(self, data=np.array([0]), prior=prior)
        
    def log_pyt(self,theta, t):
        self.cpu += len(theta)/50000
        gram_matrix = theta @ self.predictors_T
        return jnp.sum(log_logit(gram_matrix * self.response), axis=1)
    
    def loglik(self,theta, t=None):
        if t is None:
            t =self.T - 1
        l = jnp.zeros(shape=theta.shape[0])
        for s in range(t + 1):
            l += self.log_pyt(theta, s)
        #np.nan_to_num(l, copy=False, nan=-np.inf)
        return l

model = ToyModel(data=data)     


# In[5]:


#smc estimator
#temperature ladder would need to exclude t=0
smc_est=monte_est(tem[1:],wei,samp,resample_N,phi,model.loglik)

# importance tempering estimator
h_i=it_ls(samp,tem,phi,model.loglik)
dev_h=it_dev(samp,tem,phi,model.loglik)
B=10
var_it,var_dev_it=it_var(samp,tem,phi,model.loglik,B)

#elate+smc
esmc=run_gp(tem[1:],smc_est['sample_mean'],smc_est['dev_est'],smc_est['var_mean'],smc_est['dev_var'])
#elate+it
eit=run_gp(tem[1:],h_i,dev_h,var_it,var_dev_it)

eit_new=run_gp(tem[1:],h_i,dev_h,smc_est['var_mean'],smc_est['dev_var'])


# In[6]:


#save estimators
data={}
data['smc_est']=smc_est
data['it_est']=h_i
data['it_dev']=dev_h
data['it_var']=var_it
data['it_dev_var']=var_dev_it
data['esmc']=esmc
data['eit']=eit
data['eit_new']=eit_new
data['tem_list']=tem[1:]
np.savez('estimators',**data)


# In[ ]:





# In[ ]:





# In[ ]:




