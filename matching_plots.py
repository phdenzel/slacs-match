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
import matplotlib as mpl
mpl.rc('font', **{'size': 16, 'family':'serif','serif':['Computer Modern']})
mpl.rc('font',**{'size': 16, 'family':'sans-serif','sans-serif':['Computer Modern']})
mpl.rc('text', usetex=True)
from matplotlib import pyplot as plt
gleam_root = "/Users/phdenzel/gleam"
sys.path.append(gleam_root)
from gleam.reconsrc import ReconSrc, run_model
from gleam.utils.plotting import plot_scalebar, plot_labelbox, kappa_map_plot
import gleam.utils.colors as gcl
gcl.GLEAMcmaps.register_all()
from mcmc_matching import load_lo, load_lm
from mcmc_eval import read_mcmctxt
from match_eval import read_matchlog


# Parameter settings
ids = ['SDSSJ0029-0055', 'SDSSJ0737+3216', 'SDSSJ0753+3416', 'SDSSJ0956+5100',
       'SDSSJ1051+4439', 'SDSSJ1430+6104', 'SDSSJ1627-0053']
idx = 0 if len(sys.argv) < 2 else int(sys.argv[1])
lens = ids[idx]                                  # lens name
pixrad = 11                                      # pixrad of resampled kappa map
bestof = 11
sigf = [1., 1., 4., 1., 4., 1., 1.][idx]         # gain multiplier of the Poisson noise
dmaxf = [1., 0.25, 0.25, 0.6, .069, .065, 1.]    # data flux limit factors
smaxf = [1., 1., 1., 1., 1.45, 1., 1.]            # source plane (0.25 conserves surface brightness)
wrad = [.8, .5, .85, .8, .85, .8, .8][idx]       # chi2 radius
# savedir = 'match_plots/'
savedir = 'match_plots/'

# load model selection
mcmcmdl_range, chi2, mcmcangles = read_mcmctxt('mcmc/mcmceval_{}.txt'.format(lens))
chi2 = np.log(-2*chi2)
mdir = 'match_plots/{lens}/'.format(lens=lens)
matchlog = '{}{}_matching.log'.format(mdir, lens)
matchmdl_range, chi2_psf, matchangles = read_matchlog(matchlog)
sortidcs = np.argsort(chi2_psf)
matchmdl_range = matchmdl_range[sortidcs]
chi2_psf = chi2_psf[sortidcs]
matchangles = matchangles[sortidcs]
# mdl_range = mcmcmdl_range
# angles = mcmcangles

# print(mcmcmdl_range.shape, matchmdl_range.shape)
# print(mcmcangles.shape, matchangles.shape)
# mdl_range = np.concatenate((mcmcmdl_range[:bestof], matchmdl_range[:bestof]))
# angles = np.concatenate((mcmcangles[:bestof, 0], matchangles[:bestof]))
mdl_range = mcmcmdl_range[:bestof]
angles = mcmcangles[:bestof, 0]


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
        print(chi2)

        name = "{}{}_mdl{:03d}_rot{:08.4f}".format(
            lens_id, name_add, reconsrc.model_index, angle)
        print("# {}".format(name))
        print("Chi2: {:9.4f}".format(chi2))

        dmax = dmaxf[idx] * np.max(reconsrc.lensobject.data)
        sbf = smaxf[idx] * (0.25*reconsrc.lensobject.px2arcsec[0]/reconsrc.src_pxscale)**2
        cmap = 'gravic'
        # cmap = gcl.ReNormColormapAdaptor(gcl.GLEAMcmaps.vilux, mpl.colors.LogNorm(0, dmax))
        plt_kw = dict(vmin=0, origin='lower')
        
        # Source plane plot
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize, forward=True)
        plt.imshow(srcplane, extent=reconsrc.src_extent,
                   vmax=sbf*dmax, cmap=cmap, **plt_kw)
        plot_scalebar(R=reconsrc.r_max, length=1, fontsize=24)
        plt.axis('off')
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        # plt.colorbar()
        savename = "{}_srcplane{}".format(name, extension)
        plt.savefig(savedir+savename, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Synthetic plot
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize, forward=True)
        plt.imshow(synth, extent=reconsrc.lensobject.extent,
                   vmax=dmax, cmap=cmap, **plt_kw)
        plot_scalebar(R=reconsrc.lensobject.maprad, length=1, fontsize=24)
        plt.axis('off')
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        # plt.colorbar()
        savename = "{}_synth{}".format(name, extension)
        plt.savefig(savedir+savename, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Residual plot
        fig, ax = plt.subplots()
        fig.set_size_inches((figsize[0]*1.4, figsize[1]), forward=True)
        plt.imshow(resids, extent=reconsrc.lensobject.extent,
                   # vmax=chi2/100, 
                   vmax=chi2*1.5, 
                   cmap='vilux',
                   # cmap='gravic',
                   interpolation='bicubic',
                   **plt_kw)
        plot_scalebar(R=reconsrc.lensobject.maprad, length=1, fontsize=24)
        plt.text(0.95, 0.92,
                 r'$\mathrm{\chi}_{\nu}^{2}$: '+'{:2.4f}'.format(chi2),
                 horizontalalignment='right', transform=ax.transAxes,
                 color='#DEDEDE', fontsize=22)
        plt.colorbar()
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        plt.axis('off')
        savename = "{}_resid{}".format(name, extension)
        plt.savefig(savedir+savename, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # # Kappa plot
        # fig, ax = plt.subplots()
        # fig.set_size_inches(figsize, forward=True)
        # kappa_map_plot(reconsrc.model, mdl_index=model_index, extent=reconsrc.model.extent,
        #                levels=17, log=True, contours=False, scalebar=True)
        # savename = "{}_kappa{}".format(name, extension)
        # plt.savefig(savedir+savename, bbox_inches='tight')
        # plt.close()
        
        # Lens plot
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize, forward=True)
        plt.imshow(reconsrc.lens_map(mask=False),
                   extent=reconsrc.lensobject.extent,
                   vmax=dmax, cmap=cmap, **plt_kw)
        plot_scalebar(R=reconsrc.lensobject.maprad, length=1, fontsize=24)
        plot_labelbox(lens_id, position='top left', fontsize=22)
        plt.axis('off')
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        # plt.colorbar()
        savename = "{}_data{}".format(name, extension)
        plt.savefig(savedir+savename, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

        return chi2



if __name__ == "__main__":

    lo = load_lo(lens, verbose=True)
    lm = load_lm(lens, update_pixrad=pixrad, verbose=True)
    print("# <ReconSrc>")
    reconsrc = ReconSrc(lo, lm.resampled['obj'], M=120, M_fullres=400, mask_keys=['circle'])
    print(reconsrc.__v__ + "\n")
    reconsrc.calc_psf("psf/tinytim_ACS.fits", normalize=True, window_size=8, verbose=True)
    sig2 = sigf*np.abs(reconsrc.lensobject.data)
    sig2[sig2 <= 0] = sig2[sig2 > 0].min()

    kw = dict(method='minres', reduced=False, use_psf=True, use_filter=False,
              sigma2=sig2.copy(), nonzero_only=True, within_radius=wrad,
              cached=True, from_cache=False, save_to_cache=False)

    output = []

    for i, model_idx in enumerate(mdl_range[:1]):
        chi2 = plot_models(reconsrc, lens, model_idx, [angles[i]], figsize=(7.83, 7.83),
                           name_add="_mcmc{:03d}".format(i), extension='.pdf', **kw)

        output.append((chi2, model_idx))

    if 0:
        pklname = 'synths/{}/{}_pixrad{}_reconsrc.pkl'.format(lens, lens, pixrad)
        with open(pklname, 'wb') as f:
            reconsrc.model = reconsrc.model.save()
            pickle.dump(reconsrc, f)

    os.system('say "Matches have been processed"')
