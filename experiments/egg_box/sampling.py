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
from particles import collectors as col


# In[2]:


#length of chain
len_chain=100
#resample size
N=15

#sample size
N_num=len_chain*N

#variance for prior
var_prior=10


# In[ ]:





# In[3]:


#define bayesian model/log likelihood function
class ToyModel(ssp.StaticModel):
    def loglik(self,x): 
        def compute_mu(x):
            a = 4
            x1 = [-a, 0, a, -a,0, a, -a, 0, a]
            x2 = [a, a, a, 0,0, 0, -a, -a, -a]
            return np.array([x1[x] , x2[x] ])
            
        def gauss(x, mu, cov):
            diff = x - mu  # Shape: (..., 2)
            inv_cov = np.linalg.inv(cov)  # Shape: (2, 2)
            temp = np.einsum('...i,ij->...j', diff, inv_cov)
            result = np.einsum('...i,...i->...', temp, diff)
            return -0.5 * result
        var_data = 0.5
        mu_list = [compute_mu(i) for i in range(9)]
        cov = [[var_data, 0], [0, var_data]]
        rv_list = np.array([gauss(x, mu, cov) for mu in mu_list])  # Shape: (9, N1, N2)
        return logsumexp(rv_list, axis=0)

my_prior = multivariate_normal([0,0],[[var_prior,0],[0,var_prior]])
#estimation for log-normalising constant, and associated variance
collectors_list = [ssp.Var_logLt(),col.LogLts()]
my_static_model = ToyModel(prior=my_prior)
my_static_model = ToyModel(prior=my_prior)
fk_tempering = ssp.AdaptiveTempering(my_static_model,len_chain=len_chain,ESSrmin=0.998) #initialise from prio
my_temp_alg = particles.SMC(fk=fk_tempering, N=N,store_history=True, ESSrmin=1,
                            verbose=True,collect=collectors_list)
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




