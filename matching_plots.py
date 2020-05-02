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
from gleam.utils.plotting import plot_scalebar, plot_labelbox, kappa_map_plot
import gleam.utils.colors as gcl
gcl.GLEAMcmaps.register_all()
from mcmc_matching import load_lo, load_lm
from mcmc_eval import read_mcmctxt


# Parameter settings
ids = ['SDSSJ0029-0055', 'SDSSJ0737+3216', 'SDSSJ0753+3416', 'SDSSJ0956+5100',
       'SDSSJ1051+4439', 'SDSSJ1430+6104', 'SDSSJ1627-0053']
idx = 0
lens = ids[idx]                                                      # lens name
pixrad = 11                                      # pixrad of resampled kappa map
sigf = [1,][idx]                               # multiplier of the Poisson noise
# mdl_range = range(  0,   1)                            # range of models to plot
# angles = [np.random.rand(11) * 360] * len(mdl_range)   # range of angles to plot
mdl_range, _, angles = read_mcmctxt('mcmc/mcmceval_{}.txt'.format(lens))



def plot_models(reconsrc, lens_id, model_index, angles, figsize=None,
                savedir='match_plots/', name_add='', extension='.pdf', **kw):
    """
    Plot models [source planes, synthetics, residuals, lens data]
    """
    do_not_save = savedir is None
    if not do_not_save:
        savedir = savedir + '{}/'.format(lens_id)
    # calculate projections
    reconsrc.chmdl(model_index)
    reconsrc.inv_proj_matrix(cy_opt=False, use_mask=False)
    
    for angle in angles:
        chi2, (srcplane, synth, resids, _) = \
            run_model(reconsrc, angle=angle, mdl_index=model_index, output_maps=True, **kw)

        name = "{}{}_mdl{:03d}_rot{:08.4f}".format(
            lens_id, name_add, reconsrc.model_index, angle)
        print("# {}".format(name))
        print("Chi2: {:9.4f}".format(chi2))
        # Source plane plot
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize, forward=True)
        plt.imshow(srcplane, origin='lower', cmap='vilux', vmin=0, vmax=0.8,
                   extent=reconsrc.src_extent)
        plot_scalebar(R=reconsrc.r_max)
        plt.axis('off')
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        # plt.colorbar()
        savename = "{}_srcplane{}".format(name, extension)
        plt.savefig(savedir+savename, bbox_inches='tight')
        # print(savename)
        plt.close()
        # Synthetic plot
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize, forward=True)
        plt.imshow(synth, origin='lower', cmap='vilux',
                   vmin=0, extent=reconsrc.lensobject.extent)
        plot_scalebar(R=reconsrc.lensobject.maprad, length=1)
        plt.axis('off')
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        # plt.colorbar()
        savename = "{}_synth{}".format(name, extension)
        plt.savefig(savedir+savename, bbox_inches='tight')
        # print(savename)
        plt.close()
        # Residual plot
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize, forward=True)
        plt.imshow(resids, origin='lower', cmap='vilux', vmin=0, vmax=0.25,
                   extent=reconsrc.lensobject.extent)
        plot_scalebar(R=reconsrc.lensobject.maprad, length=1)
        plt.axis('off')
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        plt.text(0.95, 0.95, 'chi2: {:2.4f}'.format(chi2), color='#DEDEDE',
                 horizontalalignment='right', transform=ax.transAxes)
        # plt.colorbar()
        savename = "{}_resid{}".format(name, extension)
        plt.savefig(savedir+savename, bbox_inches='tight')
        # print(savename)
        plt.close()
        # Kappa plot
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize, forward=True)
        kappa_map_plot(reconsrc.model, mdl_index=model_index, extent=reconsrc.model.extent,
                       levels=8, log=True, contours=False, scalebar=True)
        savename = "{}_kappa{}".format(name, extension)
        plt.savefig(savedir+savename, bbox_inches='tight')
        # print(savename)
        plt.close()
        # Lens plot
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize, forward=True)
        plt.imshow(reconsrc.lens_map(mask=True), origin='lower', cmap='vilux',
                   vmin=0, extent=reconsrc.lensobject.extent)
        plot_scalebar(R=reconsrc.lensobject.maprad, length=1)
        plt.axis('off')
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        # plt.colorbar()
        savename = "{}_data{}".format(name, extension)
        plt.savefig(savedir+savename, bbox_inches='tight')
        # print(savename)
        plt.close()



if __name__ == "__main__":

    lo = load_lo(lens, verbose=True)
    lm = load_lm(lens, update_pixrad=pixrad, verbose=True)
    print("# <ReconSrc>")
    reconsrc = ReconSrc(lo, lm.resampled['obj'], M=80, M_fullres=256, mask_keys=['circle'])
    print(reconsrc.__v__ + "\n")
    sig2 = sigf*reconsrc.lensobject.data

    kw = dict(method='minres', use_psf=False, sigma2=sig2.copy(),
              nonzero_only=True, within_radius=0.8,
              cached=True, from_cache=False, save_to_cache=False)

    for i, model_idx in enumerate(mdl_range):
        plot_models(reconsrc, lens, model_idx, angles[i], figsize=(7.83, 7.83),
                    name_add="_mcmc{:03d}".format(i), extension='.png', **kw)

    if 0:
        pklname = 'synths/{}/{}_pixrad{}_reconsrc.pkl'.format(lens, lens, pixrad)
        with open(pklname, 'wb') as f:
            reconsrc.model = reconsrc.model.save()
            pickle.dump(reconsrc, f)
