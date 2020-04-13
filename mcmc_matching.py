#!/usr/bin/env python
import sys
import os
import pprint
import glob
import phdmcmc
import time
import cPickle as pickle
from functools import partial
import numpy as np
from scipy.stats.distributions import chi2 as fchi2
import pandas as pd
from scipy import ndimage
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


idx = 0         # index of lens
angle = 0       # rotation angle
pixrad = 11     # pixrad of resampled kappa map
mdl_index = -1  # model index (-1 is ensemble average)


# Load lensing data and models
ids = ['SDSSJ0029-0055', 'SDSSJ0737+3216', 'SDSSJ0753+3416', 'SDSSJ0956+5100',
       'SDSSJ1051+4439', 'SDSSJ1430+6104', 'SDSSJ1627-0053']
jsons = an_sort(glob.glob('data/*.json'))
obs = {k:v for k, v in zip(ids, jsons)}
pkls = an_sort(glob.glob('models/*.pkl'))
mdls = {k:v for k, v in zip(ids, pkls)}

objects = []
with open(obs[ids[idx]]) as f:
    l = LensObject.from_json(f)
    objects.append(l)

models = []
print("Loading...")
print(ids[idx])
with open(mdls[ids[idx]]) as f:
    mdl_pars = pickle.load(f)
    mdl_pars[1].pop('filename')
    lm = LensModel(mdl_pars[0], filename=mdls[ids[idx]], **mdl_pars[1])
    models.append(lm)

csvf = glob.glob('slacs_params.csv')
slacs_info = pd.read_csv(csvf[0])
zl = {}
zs = {}
zl[ids[idx]] = float(slacs_info[slacs_info['Name'] == ids[idx]]['z_{l}'])
zs[ids[idx]] = float(slacs_info[slacs_info['Name'] == ids[idx]]['z_{s}'])


# Resample maps
lens_id = ids[idx]
lo = objects[idx]
lm = models[idx]
lm.resample(pixrad, data_attr='data', create_instance=True, zl=zl[lens_id], zs=zs[lens_id])
reconsrc = ReconSrc(lo, lm.resampled['obj'], M=60, M_fullres=256, mask_keys=['circle'])
reconsrc.inv_proj_matrix(use_mask=True)


# Prepare functions for MCMC
#
# def log_prior2(theta):
#     angle, dzsrc = theta
#     if 0.0 < angle < 360. and -0.2 < dzsrc < 0.2:
#         return 1.0
#     return -np.inf
#
# def log_likelihood2(theta, reconsrc, **kw):
#     angle, dzsrc = theta
#     mdl_index = kw.get('mdl_index', 0)
#     prob = run_model(reconsrc, angle=angle, dzsrc=dzsrc,
#                      mdl_index=mdl_index,
#                      reduced=False, sigma2=0.0625*np.sqrt(reconsrc.lensobject.data))
#     return np.log(prob)

def log_prior(theta):
    angle = theta
    angle = angle % 360
    if 0.0 < angle < 360.:
        return 0., angle
    return -np.inf, angle

def log_likelihood(theta, args=(), **kw):
    angle = theta
    mdl_index = kw.get('mdl_index', 0)
    reconsrc = args
    chi2 = run_model(reconsrc, angle=angle, dzsrc=0, mdl_index=mdl_index,
                     reduced=False, sigma2=0.0625*np.sqrt(reconsrc.lensobject.data),
                     from_cache=True, cached=True, save_to_cache=True, flush_cache=True)
    prob = -0.5* chi2
    # prob = fchi2.logsf(chi2, len(theta))
    return prob


# Start MCMC
npars = 1
nwalkers = 360
pars = np.zeros((nwalkers, npars))
pars[:, 0] = np.random.rand(nwalkers) * 360
print(pars[0].shape)

print("Starting MCMC run...")
kw = dict(method='lsqr', dzsrc=0, mdl_index=0, reduced=False, sigma2=0.0625*np.sqrt(reconsrc.lensobject.data))
ti = time.time()
acc, rej, probs, priors = phdmcmc.mcmc_mh(
    log_likelihood, log_prior, pars, args=(reconsrc),
    stepsize=0.5, nwalkers=nwalkers, iterations=1, verbose=True, **kw)
tf = time.time()
print("Execution time: {}".format(tf-ti))

with open("mcmc.pkl", 'wb') as f:
    pickle.dump((acc, rej, probs, priors), f)
