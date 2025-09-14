#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
sys.path.insert(0, '/Users/k21163353/anaconda3/lib/python3.11/site-packages')
from particles.distributions import ProbDist
from particles import smc_samplers as ssp
import particles
import jax.numpy as jnp
from particles import collectors as col


# In[2]:


data=np.load('sonar.npy')
#length of chain
len_chain=200
#resample size
N=100
#sample size
N_num=len_chain*N

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
    delta = np.abs(arr)
    return 1/2 * (arr - delta) - np.log1p(np.exp(-delta))
    
class BinaryRegression(ssp.StaticModel):
    #tested, much faster than previous version based on logpyt
    def __init__(self, data):
        self.d = data.shape[1] - 1
        self.cpu = 0
        scale = [20] + [5] * (self.d - 1)
        prior = GaussianPrior(scale=scale)
        self.predictors_T = data[:, 1:].T
        self.response = data[:, 0]
        ssp.StaticModel.__init__(self, data=np.array([0]), prior=prior)

    def logpyt(self, theta, t):
        self.cpu += len(theta)/50000
        gram_matrix = theta @ self.predictors_T
        return np.sum(log_logit(gram_matrix * self.response), axis=1)

    def loglik(self, theta, t=None):
        if t is None:
            t = self.T - 1
        l = np.zeros(shape=theta.shape[0])
        for s in range(t + 1):
            l += self.logpyt(theta, s)
        #np.nan_to_num(l, copy=False, nan=-np.inf)
        return l

static_model = BinaryRegression(data=data)


# In[3]:


collectors_list = [ssp.Var_logLt(),col.LogLts()]
fk_tempering = ssp.AdaptiveTempering(static_model, len_chain=len_chain, ESSrmin=0.50)

my_temp_alg = particles.SMC(fk=fk_tempering, N=N,store_history=True, ESSrmin=1, verbose=True,collect=collectors_list)     
my_temp_alg.run()


# In[4]:


#save estimation for normalising constant for tempered posteriors
#save the associated variance for normalising constant estimator
var_logzt=np.array(my_temp_alg.summaries.var_logLt)/N_num
log_zt=my_temp_alg.summaries.logLts


# In[5]:


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





# In[ ]:




