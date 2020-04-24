#!/usr/bin/env python
# Imports
import sys
import glob
import phdmcmc
import cPickle as pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
gleam_root = "/Users/phdenzel/gleam"
sys.path.append(gleam_root)
from gleam.utils.lensing import LensModel
from gleam.lensobject import LensObject
from gleam.reconsrc import ReconSrc, run_model
from gleam.utils.encode import an_sort
from gleam.utils.plotting import plot_scalebar, plot_labelbox
import gleam.utils.colors as gcl
gcl.GLEAMcmaps.register_all()


# Parameter settings
ids = ['SDSSJ0029-0055', 'SDSSJ0737+3216', 'SDSSJ0753+3416', 'SDSSJ0956+5100',
       'SDSSJ1051+4439', 'SDSSJ1430+6104', 'SDSSJ1627-0053']
idx = 2
lens = ids[idx] # lens name
pixrad = 11     # pixrad of resampled kappa map
sigf = 2e+1     # multiplier of the Poisson noise
npars = 1       # dimensions of the MCMC parameter space
nwalkers = 50   # number of MCMC walkers
iters = 100     # number of MCMC iterations
step = 25.0     # step width


# Functions
def load_lo(lens, loaddir='data/', extension='json', verbose=False):
    """
    Load the SLACS LensObjects in the project
    """
    loadfile = '{}{}.{}'.format(loaddir, lens, extension)
    if verbose:
        print("# <LensObject>")
    if 'json' in extension:
        with open(loadfile) as f:
            lo = LensObject.from_json(f)
    elif 'fits' in extension:
        lo = LensObject(loadfile)
    if verbose:
        print(lo.__v__ + "\n")
    return lo

def load_lm(lens, loaddir='models/', extension='pkl',
            update_redshift=True, slacs_table='slacs_params.csv',
            update_pixrad=11,
            verbose=False):
    """
    Load the LensModels for the SLACS matching from the pickle files
    """
    loadfile = glob.glob('{}{}*.{}'.format(loaddir, lens, extension))[0]
    if 'pkl' not in extension:
        return NotImplemented
    if verbose:
        print("# <LensModel>")
    with open(loadfile) as f:
        mdl_pars = pickle.load(f)
        mdl_pars[1].pop('filename')
        lm = LensModel(mdl_pars[0], filename=loadfile, **mdl_pars[1])
    if update_redshift:
        csvf = glob.glob('slacs_params.csv')
        slacs_info = pd.read_csv(csvf[0])
        zl = float(slacs_info[slacs_info['Name'] == lens]['z_{l}'])
        zs = float(slacs_info[slacs_info['Name'] == lens]['z_{s}'])
        lm.zl = zl
        lm.zs = zs
    else:
        zl = None
        zs = None
    if update_pixrad:
        lm.resample(pixrad, data_attr='data', create_instance=True, zl=zl, zs=zs)
        if verbose:
            print(lm.resampled['obj'].__v__ + "\n")
    elif verbose:
        print(lm.__v__ + "\n")
    return lm



# Prepare functions for MCMC
def log_prior(theta, **kw):
    angle = theta
    angle = angle % 360
    if 0.0 < angle < 360.:
        return 0., angle
    return -np.inf, angle

def log_likelihood(theta, args=(), **kw):
    angle = theta
    reconsrc = args
    chi2 = run_model(reconsrc, angle=angle, **kw)
    prob = -0.5 * chi2
    return prob



if __name__ == "__main__":
    
    print("Loading data files...")
    lo = load_lo(lens, verbose=True)
    lm = load_lm(lens, update_pixrad=pixrad, verbose=True)
    print("# <ReconSrc>")
    reconsrc = ReconSrc(lo, lm.resampled['obj'], M=60, M_fullres=256, mask_keys=['circle'])
    reconsrc.inv_proj_matrix(use_mask=True)
    print(reconsrc.__v__ + "\n")
    sig2 = sigf*np.sqrt(reconsrc.lensobject.data)
    
    kw = dict(method='lsqr', dzsrc=0, reduced=False, sigma2=sig2.copy())

    # Start MCMC
    pars = np.zeros((nwalkers, npars))
    pars[:, 0] = np.random.rand(nwalkers) * 360
    print("N walkers: {}".format(nwalkers))
    print("N dims:    {}".format(npars))
    print("Starting MCMC run...")

    for imdl in range(257, reconsrc.model.N):
        print("Model {}".format(imdl))
        
        acc, rej, probs, priors, n_acc = phdmcmc.mcmc_mh(
            log_likelihood, log_prior, pars, args=(reconsrc), mdl_index=imdl,
            stepsize=step, nwalkers=nwalkers, iterations=iters, verbose=1, **kw)

        with open("mcmc/mcmc_{}_mdl{}.pkl".format(ids[idx], imdl), 'wb') as f:
            pickle.dump((acc, rej, probs, priors, n_acc), f)