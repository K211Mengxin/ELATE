#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import multivariate_normal
import sys
sys.path.insert(0, '/Users/k21163353/anaconda3/lib/python3.11/site-packages')
import particles
from particles import smc_samplers as ssp
from scipy.special import logsumexp
import jax.numpy as jnp
from particles import distributions as dists
from jax import vmap,random
import jax
from particles import collectors as col


# In[2]:


#length of chain
len_chain=1000
#resample size
N=10
#sample size
N_num=len_chain*N


# In[3]:


#generate mrna data
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


# In[4]:


#define bayesian model/log likelihood function
class ToyModel(ssp.StaticModel):
    def loglik(self,theta):
        km=theta['km']
        delta=theta['delta']
        beta=theta['beta']
        t0=theta['t0']
        params=km,delta,beta,t0
        mu_t= vmap(mu, in_axes=(None,0), out_axes=0)(mrna_t,params)
        assert jnp.all(jnp.isfinite(mu_t)), "mu_t has NaNs"
        assert jnp.all(jnp.isfinite(mrna_data - mu_t)), "ys - mu_t has NaNs"
        def compute_log_likelihood(mu_t, ys, sigma):
            log_lik_value = jnp.sum(jax.scipy.stats.norm.logpdf(ys-mu_t, loc=0.0, scale=sigma))
            return log_lik_value

        vectorized_log_likelihood = vmap(compute_log_likelihood, in_axes=(0, None,None))
        log_likelihood_values = vectorized_log_likelihood(mu_t, mrna_data , sigma)
        warn_if_nan(log_likelihood_values)
        return np.array(log_likelihood_values)

my_prior=dists.StructDist({'km':dists.Uniform(a=0.0,b=6),
                          'delta': dists.Uniform(a=0.0,b=1),
                           'beta': dists.Uniform(a=0.0,b=1),
                           't0': dists.Uniform(a=0.0,b=3),
                          })
collectors_list = [ssp.Var_logLt(),col.LogLts()]
my_static_model = ToyModel(prior=my_prior)
fk_tempering = ssp.AdaptiveTempering(my_static_model,len_chain=len_chain,ESSrmin=0.65) #initialise from prio
my_temp_alg = particles.SMC(fk=fk_tempering, N=N,store_history=True, ESSrmin=1,
                           verbose=True,collect=collectors_list)
my_temp_alg.run()


# In[5]:


#save estimation for normalising constant for tempered posteriors
#save the associated variance for normalising constant estimator
var_logzt=np.array(my_temp_alg.summaries.var_logLt)/N_num
log_zt=my_temp_alg.summaries.logLts


# In[6]:


# save: temperature ladder,samples and associated weightes, resample size
data={}
#this would include temperature 0
data['tem']=my_temp_alg.X.shared["exponents"]
data['samp']=my_temp_alg.hist.X
data['W']=my_temp_alg.hist.wgts
data['re_N']=my_temp_alg.N
data['log_zt']=log_zt
data['var_logzt']=var_logzt
np.savez('smc_samples',**data)


# In[ ]:




