#!/usr/bin/env python
import sys
import os
import pprint
import glob
import cPickle as pickle
from functools import partial
import numpy as np
import pandas as pd
from scipy import ndimage
from matplotlib import pyplot as plt
gleam_root = "/Users/phdenzel/gleam"
sys.path.append(gleam_root)
from gleam.reconsrc import ReconSrc, run_model
from gleam.utils.plotting import plot_scalebar, plot_labelbox
import gleam.utils.colors as gcl
gcl.GLEAMcmaps.register_all()
from mcmc_matching import load_lo, load_lm


# Parameter settings
ids = ['SDSSJ0029-0055', 'SDSSJ0737+3216', 'SDSSJ0753+3416', 'SDSSJ0956+5100',
       'SDSSJ1051+4439', 'SDSSJ1430+6104', 'SDSSJ1627-0053']
idx = 0
lens = ids[idx]                      # lens name
pixrad = 11                          # pixrad of resampled kappa map
sigf = 5e-1*0.125                    # multiplier of the Poisson noise
mdl_range = range(2, 3)              # range of models to plot
angles = np.random.rand(360) * 360   # range of angles to plot
# angles = [116.5, 335.2]


def plot_models(reconsrc, lens_id, model_index, angles, pixrad,
                sigma2=None,
                savedir='match_plots/', extension='.pdf'):
    """
    Plot models [source planes, synthetics, residuals, lens data]
    """
    do_not_save = savedir is None
    if not do_not_save:
        savedir = savedir + '{}/'.format(lens_id)
    # calculate projections
    reconsrc.chmdl(model_index)
    reconsrc.inv_proj_matrix(cy_opt=False, use_mask=True)
    kw = dict(method='lsqr', use_psf=False, sigma2=sigma2.copy(),
              cached=False, from_cache=False, save_to_cache=False)

    for angle in angles:

        chi2, (srcplane, synth, resids, _) = \
            run_model(reconsrc, angle=angle, mdl_index=model_index, output_maps=True, **kw)
    
        # Source plane plot
        fig, ax = plt.subplots()
        plt.imshow(srcplane, origin='lower', cmap='vilux', vmin=0, vmax=0.8)
        plt.axis('off')
        plt.colorbar()
        savename = "{}_mdl{}_rot{:08.4f}_srcplane{}".format(
            lens_id, reconsrc.model_index, angle, extension)
        plt.savefig(savedir+savename, bbox_inches='tight')
        print(savename)
        plt.close()
        
        # Synthetic plot
        fig, ax = plt.subplots()
        plt.imshow(synth, origin='lower', cmap='vilux',
                   vmin=0, vmax=10, extent=reconsrc.lensobject.extent)
        plot_scalebar(R=reconsrc.lensobject.maprad, length=1)
        plt.axis('off')
        plt.colorbar()
        savename = "{}_mdl{}_rot{:08.4f}_synth{}".format(
            lens_id, reconsrc.model_index, angle, extension)
        plt.savefig(savedir+savename, bbox_inches='tight')
        print(savename)
        plt.close()

        # Residual plot
        fig, ax = plt.subplots()
        vmax = 0.025 if chi2 < 10 else 0.25
        plt.imshow(resids, origin='lower', cmap='vilux', vmin=0, vmax=vmax)
        plot_scalebar(R=reconsrc.lensobject.maprad, length=1)
        plt.axis('off')
        plt.text(0.95, 0.95, 'chi2: {:2.4f}'.format(chi2), color='#DEDEDE',
                 horizontalalignment='right', transform=ax.transAxes)
        plt.colorbar()
        savename = "{}_mdl{}_rot{:08.4f}_resid{}".format(
            lens_id, reconsrc.model_index, angle, extension)
        plt.savefig(savedir+savename, bbox_inches='tight')
        print(savename)
        plt.close()

        # Lens plot
        fig, ax = plt.subplots()
        plt.imshow(reconsrc.lens_map(), origin='lower', cmap='vilux',
                   vmin=0, vmax=10, extent=reconsrc.lensobject.extent)
        plot_scalebar(R=reconsrc.lensobject.maprad, length=1)
        plt.axis('off')
        plt.colorbar()
        savename = "{}_rot{:08.4f}{}".format(lens_id, angle, extension)
        plt.savefig(savedir+savename, bbox_inches='tight')
        plt.close()



if __name__ == "__main__":

    lo = load_lo(lens, verbose=True)
    lm = load_lm(lens, update_pixrad=pixrad, verbose=True)
    print("# <ReconSrc>")
    reconsrc = ReconSrc(lo, lm.resampled['obj'], M=60, M_fullres=256, mask_keys=['circle'])
    print(reconsrc.__v__ + "\n")
    sig2 = sigf*np.sqrt(reconsrc.lensobject.data)

    for model_idx in mdl_range:
        plot_models(reconsrc, lens, model_idx, angles, pixrad, sigma2=sig2, extension='.png')

    if 0:
        pklname = 'synths/{}/{}_pixrad{}_reconsrc.pkl'.format(lens, lens, pixrad)
        with open(pklname, 'wb') as f:
            reconsrc.model = reconsrc.model.save()
            pickle.dump(reconsrc, f)
