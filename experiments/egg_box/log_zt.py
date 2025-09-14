#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
sys.path.insert(0, '/Users/k21163353/anaconda3/lib/python3.11/site-packages')
import particles
from particles import smc_samplers as ssp
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
from monte import monte_est
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp
from scipy import integrate
from elate import run_gp,run_gp_2
import math
import jax


# In[2]:


smc_data=np.load('smc_samples.npz',allow_pickle=True)
#read temperature ladder, samples, weights
tem=np.array(smc_data['tem'][1:])
samp=np.array(smc_data['samp'])
wei=np.array(smc_data['W'])
resample_N=np.array(smc_data['re_N'])

#read smc estimation for log normalising constant
log_zt=np.array(smc_data['log_zt'])
var_logzt=np.array(smc_data['var_logzt'])


# In[3]:


#Toymodel used for sampling
from jax.numpy.linalg import inv
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


# In[4]:


#smc estimator, with test function log likelihood
#temperature ladder would need to exclude t=0
model = ToyModel() 
smc_data=monte_est(tem,wei,samp,resample_N,model.loglik,model.loglik)
g_t=smc_data['sample_mean']


# ## Simpson's rule and Trapezoid rule

# In[5]:


from collections.abc import Sequence

def simpson_nonuniform(x: Sequence[float], f: Sequence[float]) -> float:
    N = len(x) - 1
    h = [x[i + 1] - x[i] for i in range(0, N)]
    assert N > 0

    result = 0.0
    for i in range(1, N, 2):
        h0, h1 = h[i - 1], h[i]
        hph, hdh, hmh = h1 + h0, h1 / h0, h1 * h0
        result += (hph / 6) * (
            (2 - hdh) * f[i - 1] + (hph**2 / hmh) * f[i] + (2 - 1 / hdh) * f[i + 1]
        )

    if N % 2 == 1:
        h0, h1 = h[N - 2], h[N - 1]
        result += f[N]     * (2 * h1 ** 2 + 3 * h0 * h1) / (6 * (h0 + h1))
        result += f[N - 1] * (h1 ** 2 + 3 * h1 * h0)     / (6 * h0)
        result -= f[N - 2] * h1 ** 3                     / (6 * h0 * (h0 + h1))
    return result


# In[6]:


simp_log_z=simpson_nonuniform(tem,g_t)
trap_log_z=integrate.trapezoid(g_t,x=tem)


# ## ELATE 

# In[7]:


elate=run_gp(jnp.array(tem),jnp.array(smc_data['sample_mean']),jnp.array(smc_data['dev_est']),jnp.array(smc_data['var_mean']),jnp.array(smc_data['dev_var']))


# In[8]:


ts=jnp.array(tem)
p=np.ravel(elate['p'])
q=np.ravel(elate['q'])
v=float(elate['v'])
ell=float(elate['ell'])

g_dev=smc_data['dev_est']
sigma2_y = smc_data['var_mean']
sigma2_yd =smc_data['dev_var']
ys = jnp.stack([g_t, g_dev], -1)
kxy = lambda t, u, ell: jnp.exp(-0.5 * jnp.square((t-u)/ell))
dkxy = lambda t, u, ell: -(t-u) / jnp.square(ell) * kxy(t, u, ell)
ddkxy = lambda t, u, ell: (1 - jnp.square((t-u)/ell)) / jnp.square(ell) * kxy(t, u, ell)

mapped = lambda func: jax.vmap(jax.vmap(func, (None, 0, None)), (0, None, None))
mapped1 = lambda func: jax.vmap(func, (0, None, None))


# In[9]:


def mean_func( x):
     def rational_function(_x, p, q):
        p_poly = jnp.polyval(p, _x)
        q_poly = jnp.polyval(q, _x)
        return p_poly / q_poly
     return jnp.array(jax.value_and_grad(rational_function, 0)(x, p, q))

def gram_matrix(ell, _xs):
    A = mapped(kxy)(_xs, _xs, ell)
    B = mapped(dkxy)(_xs, _xs, ell)
    C = mapped(ddkxy)(_xs, _xs, ell)
    M1 = jnp.concatenate([A, -B], -1)
    M2 = jnp.concatenate([B, C], -1)
    return jnp.concatenate([M1, M2], 0)

def kxu(ell, xnew):

    A = mapped1(kxy)(ts, xnew, ell)
    B = mapped1(dkxy)(ts, xnew, ell)
    D = mapped1(ddkxy)(ts, xnew, ell)
    M1 = jnp.concatenate([A, B], -1)
    M2 = jnp.concatenate([-B, D], -1)
    xn = jnp.array(xnew).reshape((1,))
    kuu = gram_matrix(ell, xn)
    return jnp.stack([M1, M2], 1), kuu

def pmean(xnew):
    pmean = mean_func(xnew)
    return pmean[0]#, jnp.diagonal(vs)

def integral_kx(ti):
    return ell*np.sqrt(math.pi/2)*(math.erf((1-ti)/(np.sqrt(2)*ell))+math.erf((ti)/(np.sqrt(2)*ell)))

def integral_dkx(ti):
    return kxy(1,ti,ell)-kxy(0,ti,ell)


# In[10]:


w_func = jnp.array([integral_kx(ti) for ti in ts])
w_deriv = jnp.array([integral_dkx(ti) for ti in ts])
w = jnp.concatenate([w_func, w_deriv])

sigma2 = jnp.concatenate([sigma2_y, sigma2_yd], -1)
K = gram_matrix(ell, ts) + jnp.diag(sigma2/v)
chol = jnp.linalg.cholesky(K)
chol_inv = jax.scipy.linalg.solve_triangular(chol, jnp.eye(chol.shape[0]), lower=True)

var_term1=2*ell**2*(np.exp(-1/(2*ell**2))-1)+np.sqrt(2*math.pi)*math.erf(1/np.sqrt(2))
var_logz=var_term1*v-v*w.T @ chol_inv.T @ chol_inv @ w
rat_inte,error=integrate.quad(pmean, 0, 1)
res = jnp.transpose(ys-jax.vmap(mean_func, (0))( ts)).reshape((-1,))
elate_log_z=rat_inte+w.T @ chol_inv.T @ chol_inv @ res
m2_logz=np.random.normal(loc=elate_log_z, scale=np.sqrt(var_logz))



# ## ELATE-v2

# In[11]:


elate_v2=run_gp_2(ts,log_zt,var_logzt)
elate_v2_log_z=elate_v2['post_mean'][-1]


# In[12]:


data={}
data['simpson']=simp_log_z
data['trapezoidal']=trap_log_z
data['smc']=log_zt[-1]
data['elate']=elate_log_z
data['elate_v2']=elate_v2_log_z
np.savez('norm_const',**data)


# In[ ]:




