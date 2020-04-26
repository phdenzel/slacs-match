#!/usr/bin/env python
# Imports
import sys
import numpy as np
import scipy as sp
import cPickle as pickle
import phdmcmc
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
gleam_root = "/Users/phdenzel/gleam"
sys.path.append(gleam_root)
from gleam.reconsrc import ReconSrc, run_model
from mcmc_matching import load_lo, load_lm


# Parameter settings
ids = ['SDSSJ0029-0055', 'SDSSJ0737+3216', 'SDSSJ0753+3416', 'SDSSJ0956+5100',
       'SDSSJ1051+4439', 'SDSSJ1430+6104', 'SDSSJ1627-0053']
idx = 0
lens = ids[idx]     # lens name   
pixrad = 11         # pixrad of resampled kappa map
sigf = 5e-1*0.125   # multiplier of the Poisson noise


# Functions
def load_mcmcpkl(lens, mdlidx=0, mcmcdir='mcmc/', discard=0.1):
    pklfile = "{mcmcdir}{lens}/mcmc_{lens}_mdl{mdlidx}.pkl".format(mcmcdir=mcmcdir, lens=lens, mdlidx=mdlidx)
    with open(pklfile, 'rb') as f:
        full_acc, full_rej, full_probs, full_priors, full_n_acc = pickle.load(f)
    acc, rej, probs, priors = phdmcmc.flat_chain(full_acc, full_rej, full_probs, full_priors, full_n_acc, discard=discard)
    return (full_acc, full_rej, full_probs, full_priors, full_n_acc),  (acc, rej, probs, priors)

def get_mcmcpeaks(mcmc_chains, nclusters=2, verbose=True):
    acc, rej, probs, priors = mcmc_chains
    imax = np.argmax(probs)
    if verbose:
        mlkstr = "Maximum likelihood @ {:6.4f}: {:6.4f}".format(float(acc[imax]), probs[imax])
        print(mlkstr)
    peaks, _ = sp.signal.find_peaks(probs, distance=len(probs)//15)
    kmeans = KMeans(n_clusters=nclusters).fit(acc[peaks])
    clusters = kmeans.cluster_centers_
    return imax, (kmeans, clusters)

def get_mcmcstats(mcmc_chains, rawmcmc_chains=None):
    acc, rej, probs, priors = mcmc_chains
    dimstr = "Ndims:     \t{}".format(acc.shape[-1])
    if rawmcmc_chains is not None:
        full_acc, full_rej, full_probs, full_priors, full_n_acc = rawmcmc_chains
        nwkstr = "Nwalkers:  \t{}".format(full_acc.shape[0])
    else:
        nwkstr = ""
    accstr = "Accepted:  \t{}".format(acc.shape[0])
    rejstr = "Rejected:  \t{}".format(rej.shape[0])
    prcstr = "Acceptance:\t{:5.2f}%".format((100.*acc.shape[0])/(acc.shape[0]+rej.shape[0]))
    print("\n".join([dimstr, nwkstr, accstr, rejstr, prcstr]))

def read_mcmctxt(filename):
    dta = np.loadtxt(filename)
    mdl_idcs = dta[:, 0].astype(int)
    logprobs = dta[:, 1]
    theta = dta[:, 2:]
    return mdl_idcs, logprobs, theta



if __name__ == "__main__":

    lo = load_lo(lens, verbose=True)
    lm = load_lm(lens, update_pixrad=pixrad, verbose=True)
    print("# <ReconSrc>")
    reconsrc = ReconSrc(lo, lm.resampled['obj'], M=60, M_fullres=256, mask_keys=['circle'])
    print(reconsrc.__v__ + "\n")
    sig2 = sigf*np.sqrt(reconsrc.lensobject.data)
    
    kw = dict(method='lsqr', dzsrc=0, reduced=False, sigma2=sig2.copy())

    # Loop through MCMC chains
    peaks = []
    clusters = []
    maxP = []
    for mdlidx in range(0, reconsrc.model.N):
        rawmcmc_chains, mcmc_chains = load_mcmcpkl(lens, mdlidx)
        imax, (kmeans, centers) = get_mcmcpeaks(mcmc_chains, nclusters=2, verbose=False)
        acc, rej, probs, priors = mcmc_chains
        peaks.append(acc[imax])
        clusters.append(centers)
        maxP.append(probs[imax])
    peaks = np.array(peaks)
    clusters = np.array(clusters)
    maxP = np.array(maxP)

    # Evaluate MCMC data
    best_to_worst = maxP.argsort()[::-1]
    mcmcdata = np.zeros(best_to_worst.shape + (3*peaks.shape[-1],))
    mcmcdata[:, 0] = best_to_worst
    mcmcdata[:, 1] = maxP[best_to_worst]
    for i in range(peaks.shape[-1]):
        mcmcdata[:, 2+i] = peaks[:, i][best_to_worst]

    hdrstr = "MCMC eval of {}\nModel idx | max log(P) | theta".format(lens)
    savename = "mcmc/mcmceval_{}.txt".format(lens)
    np.savetxt(savename, mcmcdata, fmt=["%3d", "%7.4f", "%6.2f"],
               delimiter='   \t',
               header=hdrstr)
    print mcmcdata
