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


# In[16]:


#get estimators from smc, it, elate+smc,elate+it

#read smc samples
smc_data=jnp.load('smc_samples.npz',allow_pickle=True)
tem=np.array(smc_data['tem'])
samp=np.array(smc_data['samp'])
wei=np.array(smc_data['W'])
resample_N=np.array(smc_data['re_N'])

#define test function
phi=lambda x:x[:,0]


# In[17]:


#Toymodel used for sampling
from jax.numpy.linalg import inv
from jax.numpy.linalg import det


class ToyModel(ssp.StaticModel):
    def loglik(self,x):  
        a = 4.0
        mu_list = jnp.array([
            [-a, a], [0, a], [a, a],
            [-a, 0], [0, 0], [a, 0],
            [-a, -a], [0, -a], [a, -a]
        ])  # shape: (9, 2)
        var_data = 0.5
        cov = var_data * jnp.eye(2)   # shape: (2, 2)
        inv_cov = inv(cov)            # constant inverse
        
        def gauss(mu):
            diff = x - mu  # shape (..., 2)
            temp = jnp.einsum('...i,ij->...j', diff, inv_cov)
            result = jnp.einsum('...i,...i->...', temp, diff)
            return -0.5 * result
    
        # map gauss over all 9 mu (shape: (9, 2))
        rv_list = vmap(gauss)(mu_list)  # shape: (9, N) or (9,) depending on x
        return logsumexp(rv_list, axis=0) 
        
model = ToyModel()       


# In[4]:


def phi(x):
    a = 4.0
    var_data = 0.5
    cov = jnp.array([[var_data, 0.0], [0.0, var_data]])
    inv_cov = jnp.linalg.inv(cov)

    mus = jnp.array([
        [-a,  a], [0, a], [ a, a],
        [-a,  0], [0, 0], [ a, 0],
        [-a, -a], [0,-a], [ a,-a]
    ])

    def gauss_log(x, mu):
        diff = x - mu
        temp = diff @ inv_cov
        return -0.5 * jnp.sum(temp * diff, axis=-1)

    rv_list = jnp.array([gauss_log(x, mu) for mu in mus])  # (9,)
    return jnp.exp(rv_list[0] - logsumexp(rv_list))


# In[18]:


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


# In[19]:


var_dev_it


# In[20]:


#save estimators
data={}
data['smc_est']=smc_est
data['it_est']=h_i
data['it_dev']=dev_h
data['it_var']=var_it
data['it_dev_var']=var_dev_it
data['esmc']=esmc
data['eit']=eit
np.savez('estimators',**data)


# In[ ]:




