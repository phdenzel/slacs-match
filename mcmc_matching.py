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
imdl = 0        # model index
npars = 1       # dimensions of the MCMC parameter space
nwalkers = 50   # number of MCMC walkers
iters = 100     # number of MCMC iterations
step = 25.0     # step width


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
print("# <LensModel>")
print(lm.__v__)
print("# <Reconsrc>")
print(reconsrc.__v__)

# Prepare sigma2
sig2 = 5e-1*0.125 * np.sqrt(reconsrc.lensobject.data)
kw = dict(method='lsqr', dzsrc=0, reduced=False, sigma2=sig2.copy())

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


# for angle in [0, 10, 70, 160, 220, 280, 340]:
#     chi2, (srcplane, synth, resids, s2) = \
#         run_model(reconsrc, angle=angle, mdl_index=imdl, output_maps=True, **kw)
#     print(chi2)
#     print(np.sum(resids))
#     plt.imshow(resids, cmap='vilux')
#     plt.show()

#     plt.imshow(s2)
#     plt.show()
# exit(1)


###
# Start MCMC
pars = np.zeros((nwalkers, npars))
pars[:, 0] = np.random.rand(nwalkers) * 360
print("N walkers: {}".format(nwalkers))
print("N dims:    {}".format(npars))

print("Starting MCMC run...")

for imdl in range(100, reconsrc.model.N):
    
    print("Model {}".format(imdl))
    
    acc, rej, probs, priors, n_acc = phdmcmc.mcmc_mh(
        log_likelihood, log_prior, pars, args=(reconsrc), mdl_index=imdl,
        stepsize=step, nwalkers=nwalkers, iterations=iters, verbose=1, **kw)

    with open("mcmc/mcmc_{}_mdl{}.pkl".format(ids[idx], imdl), 'wb') as f:
        pickle.dump((acc, rej, probs, priors, n_acc), f)

