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
from jax import vmap,random
import jax


# In[2]:


#get estimators from smc, it, elate+smc,elate+it

#read smc samples
smc_data=jnp.load('smc_samples.npz',allow_pickle=True)
tem=np.array(smc_data['tem'])
old_samp=np.array(smc_data['samp'])
wei=np.array(smc_data['W'])
resample_N=np.array(smc_data['re_N'])

class Sample:
    def __init__(self, theta):
        self.theta = theta

samp = []

for i in range(len(old_samp)):
    km = old_samp[i].theta['km']
    delta = old_samp[i].theta['delta']
    beta = old_samp[i].theta['beta']
    t0 = old_samp[i].theta['t0']
    theta_array = jnp.stack([km, delta, beta, t0], axis=1)
    samp.append(Sample(theta=theta_array))
    
#define test function
phi=lambda x: x[:,1]


# In[ ]:





# In[3]:


def mu(t, params):
    km,delta, beta,t0 = params
    denom = delta - beta
    dt = t - t0
    raw_mu = km / denom * (1 - jnp.exp(-denom * dt)) * jnp.exp(-beta * dt)

    safe_mu = jnp.where(
        jnp.isfinite(raw_mu),
        raw_mu,
        10000.0
    )

    return safe_mu

def warn_if_nan(arr):
    nan_indices = np.where(np.isnan(arr))[0]
    if len(nan_indices) != 0:
        warnings.warn("Warning: jax.scipy.stats.multivariate_normal.logpdf returned NaN. Check the input values, especially the covariance matrix.")


def sim_data(mu, sigma, key):
    return mu + sigma * random.normal(key)

mrna_t=np.linspace(0,10,50)
km=5
delta=0.1
beta=0.8
t0=2
params=km,delta,beta,t0
mu_t=mu(mrna_t,params)

key = random.PRNGKey(0)
keys = random.split(key, len(mrna_t))
sigma =1
mrna_data = vmap(sim_data, in_axes=(0, None, 0))(mu_t, sigma, keys)
    
class ToyModel(ssp.StaticModel):
    def loglik(self,theta):
        km=theta[:,0]
        delta=theta[:,1]
        beta=theta[:,2]
        t0=theta[:,3]
        params=km,delta,beta,t0
        mu_t= vmap(mu, in_axes=(None,0), out_axes=0)(mrna_t,params)
        # assert jnp.all(jnp.isfinite(mu_t)), "mu_t has NaNs"
        # assert jnp.all(jnp.isfinite(mrna_data - mu_t)), "ys - mu_t has NaNs"
        def compute_log_likelihood(mu_t, ys, sigma):
            log_lik_value = jnp.sum(jax.scipy.stats.norm.logpdf(ys-mu_t, loc=0.0, scale=sigma))
            return log_lik_value

        vectorized_log_likelihood = vmap(compute_log_likelihood, in_axes=(0, None,None))
        log_likelihood_values = vectorized_log_likelihood(mu_t, mrna_data , sigma)
        mask_ind = jnp.logical_not(jnp.isfinite(log_likelihood_values))
        log_likelihood_values = jnp.where(mask_ind, -1e10, log_likelihood_values)
        return jnp.array(log_likelihood_values)
        
model = ToyModel()      


# In[4]:


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


# In[5]:


#save estimators
data={}
data['smc_est']=smc_est
data['it_est']=h_i
data['it_dev']=dev_h
data['it_var']=var_it
data['it_dev_var']=var_dev_it
data['esmc']=esmc
data['eit']=eit
data['tem_list']=tem[1:]
np.savez('estimators',**data)


# 

# In[ ]:




