#define weight function for t_i
import jax.numpy as jnp
from jax import vmap

def is_wei(theta,tem_i,tem,log_lik):
    log_likelihood_values=log_lik(theta)
    wei=log_likelihood_values*(tem_i-tem)
    ti=wei-jnp.max(wei)
    return jnp.exp(ti)

#dfine hij, j is the index for temperatures previous to t_i
def is_est(samp_j,t_i,t_j,phi,log_lik):
    #wei list corresponding to each sample
    #acutually t_{j-1}
    wei=is_wei(samp_j,t_i,t_j,log_lik)
    l_j=jnp.sum(wei)**2/jnp.sum(wei**2)
    val=jnp.sum(wei*phi(samp_j))
    #normalise weight
    norm_wei=jnp.sum(wei)
    return val/norm_wei,l_j



def it_ls(samp,tem,phi,log_lik):
    h_i=jnp.empty(0)
    for i in range(len(samp)):
        hi=0
        l_j=0
        for j in range(i+1):
            val,wei=is_est(samp[j].theta,tem[i+1],tem[j],phi,log_lik)
            hi+=val*wei
            l_j+=wei
        hi=hi/l_j
        h_i=jnp.append(h_i,hi)
    return h_i

import numpy as np

def boots(values, B, random_state=None):
    values = jnp.asarray(values)
    
    n = len(values)
    samples=[]
    rng = np.random.default_rng(random_state)  
    for _ in range(B):
        sample = rng.choice(values, size=n, replace=True)
        samples.append(jnp.array(sample))
    return samples

def it_var(samp,tem,phi,log_lik,b):
    fl=lambda x:phi(x)*log_lik(x)
    var=jnp.empty(0)
    var_fl=jnp.empty(0)
    var_l=jnp.empty(0)
    for i in range(len(samp)):
        hi=jnp.empty(b)
        l_j=jnp.empty(b)
        l_t=jnp.empty(0)

        hi_fl=jnp.empty(b)
        l_j_fl=jnp.empty(b)
        l_t_fl=jnp.empty(0)

        hi_l=jnp.empty(b)
        l_j_l=jnp.empty(b)
        l_t_l=jnp.empty(0)
        for j in range(i+1):
            #list of boots hi
            samples=boots(samp[j].theta,B=b)
            v_is_est = vmap(is_est, in_axes=(0, None, None, None, None))
            val,l = v_is_est(jnp.array(samples), tem[i+1],tem[j], phi, log_lik)
            #val,wei=vmap(is_est,in_axes=(0,None,None,None,None))(jnp.array(samples),tem[i+1],tem[j],phi,log_lik)
            val = jnp.array(val)  # shape (B,)
            l = jnp.array(l)  
            hi+=val*l
            #print(l)
            l_j+=l
            l_t=jnp.append(l_t,l[0])

            val_fl,l_fl = v_is_est(jnp.array(samples), tem[i+1],tem[j], fl, log_lik)
            #val,wei=vmap(is_est,in_axes=(0,None,None,None,None))(jnp.array(samples),tem[i+1],tem[j],phi,log_lik)
            val_fl = jnp.array(val_fl)  # shape (B,)
            l_fl = jnp.array(l_fl)  
            hi_fl+=val_fl*l_fl
            #print(l)
            l_j_fl+=l_fl

            val_l,l_l = v_is_est(jnp.array(samples), tem[i+1],tem[j], log_lik, log_lik)
            #val,wei=vmap(is_est,in_axes=(0,None,None,None,None))(jnp.array(samples),tem[i+1],tem[j],phi,log_lik)
            val_l = jnp.array(val_l)  # shape (B,)
            l_l = jnp.array(l_l)  
            hi_l+=val_l*l_l
            l_j_l+=l_l

        
        # print(l_t/l_j[0])
        # print(len(l_t))
        hi=hi/l_j
        #print(len(hi))
        var_h_lam=np.var(hi)
        var=jnp.append(var,var_h_lam)

        hi_fl=hi_fl/l_j_fl
        #print(len(hi))
        var_fl_lam=np.var(hi_fl)
        var_fl=jnp.append(var_fl,var_fl_lam)

        hi_l=hi_l/l_j_l
        #print(len(hi))
        var_l_lam=np.var(hi_l)
        var_l=jnp.append(var_l,var_l_lam)
        
    term_2 = it_ls(samp,tem,phi,log_lik)
    term_3 = it_ls(samp,tem,log_lik,log_lik)
    var_dev = var_fl + term_2**2 * var_l + term_3**2 * var 
    return var,var_dev

def it_dev_var(samp,tem,phi,log_lik):
    fl=lambda x:phi(x)*log_lik(x)
    
    var_fl,_ = it_var(samp,tem,fl,log_lik)
    var_l,_ = it_var(samp,tem,log_lik,log_lik)
    var_f,_ =it_var(samp,tem,phi,log_lik)
    
    
    term_2 = it_ls(samp,tem,phi,log_lik)
    term_3 = it_ls(samp,tem,log_lik,log_lik)
    var_dev = var_fl + term_2**2 * var_l + term_3**2 * var_f 
    return var_dev

def it_dev(samp,tem,phi,log_lik):
    fl=lambda x:phi(x)*log_lik(x)
    e_fl=it_ls(samp,tem,fl,log_lik)
    e_f=it_ls(samp,tem,phi,log_lik)
    e_l=it_ls(samp,tem,log_lik,log_lik)
    return e_fl-e_f*e_l
