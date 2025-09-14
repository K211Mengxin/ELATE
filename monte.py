import numpy as np
import sys
import particles
from particles.variance_mcmc import MCMC_variance

#input weights for each sample
#samples
#resample size
#test function
#index for t_i
def var_wf(wei,samp,N,phi,i):
    N0 = wei[i].W.shape[0]
    fx = phi(samp[i].theta)
    mask_ind=np.logical_not(np.isfinite(fx))
    fx=np.where(mask_ind,1e10,fx)
    fmean = np.average(fx, weights=wei[i].W)
    wphi = wei[i].W * (fx - fmean)
    wphi_reshaped = np.reshape(wphi, (-1, N), "C")
    return MCMC_variance(wphi_reshaped, method="init_seq") * N0 ** 2


def monte_est(tem,wei,samp,N,phi,loglik):

    sample_mean = []
    var_list = []

    derivative = []
    dev_var = []
    for i in range(len(tem)):
        N_num=wei[i].W.shape[0]

        mean_value = np.sum([a * b for a, b in zip(wei[i].W, phi(samp[i].theta))])
        var_est = var_wf(wei,samp,N,phi,i)
        
        var_list.append(var_est / N_num)
        sample_mean.append(mean_value)

        
        log_lik = loglik(samp[i].theta)
        term_1 = np.sum([a * b for a, b in zip(wei[i].W, phi(samp[i].theta)* log_lik)])
        mask_ind=np.logical_not(np.isfinite(term_1))
        term_1=np.where(mask_ind,1e10,term_1)
        
        term_2 = mean_value   
        term_3 = np.sum([a * b for a, b in zip(wei[i].W, log_lik)])
        dev = term_1 - term_2 * term_3
        derivative.append(dev)

        fl=lambda x:phi(x)*loglik(x)
        var_fl = var_wf(wei,samp,N,fl , i) / N_num
        var_l = var_wf(wei,samp,N, loglik, i) / N_num
        var_f = var_wf(wei,samp,N, phi, i) / N_num
        var_dev = var_fl + term_2**2 * var_l + term_3**2 * var_f
        dev_var.append(var_dev)


    sample_mean = np.array(sample_mean)
    var_list = np.array(var_list)
    derivative = np.array(derivative)
    dev_var = np.array(dev_var)
    data = {
        'tem_list':np.array(tem),
        'sample_mean': np.array(sample_mean),
        'var_mean': np.array(var_list),
        'dev_est': np.array(derivative),
        'dev_var': np.array(dev_var)       
    }
    return data
