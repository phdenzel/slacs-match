#!/usr/bin/env python
import sys
import os
import pprint
import cPickle as pickle
import numpy as np
from matplotlib import pyplot as plt
gleam_root = "/Users/phdenzel/gleam"
sys.path.append(gleam_root)
from gleam.utils.lensing import LensModel, DLDSDLS, DLSDS, DL, DS, find_einstein_radius
from gleam.utils.plotting import kappa_map_plot, arrival_time_surface_plot, kappa_profiles_plot
from gleam.utils.plotting import plot_scalebar, plot_labelbox
import gleam.utils.colors as gcl
gcl.GLEAMcmaps.register_all()
from mcmc_matching import load_lo, load_lm
from mcmc_eval import read_mcmctxt
from match_eval import read_matchlog


# Lens parameters
ids = ['SDSSJ0029-0055', 'SDSSJ0737+3216', 'SDSSJ0753+3416', 'SDSSJ0956+5100',
       'SDSSJ1051+4439', 'SDSSJ1430+6104', 'SDSSJ1627-0053']
idx = 0 if len(sys.argv) < 2 else int(sys.argv[1])
lens = ids[idx]
pixrad = 11
zl_mdl = 0.23
zs_mdl = 0.8
mdir = 'match_plots/{lens}/'.format(lens=lens)
sdir = os.path.join(mdir, 'sorted')

bestof = 11
print(lens)

KAPPA = 1
ARRIV = 1
KPROF = 1
savedir = 'models/matches/'
extension = 'pdf'


# load list of best models (modelled w/o & w/ PSF)
mcmcmdl_range, chi2, mcmcangles = read_mcmctxt('mcmc/mcmceval_{}.txt'.format(lens))
chi2 = np.log(-2*chi2)
matchlog = '{}{}_matching.log'.format(mdir, lens)
matchmdl_range, chi2_psf, matchangles = read_matchlog(matchlog)
sortidcs = np.argsort(chi2_psf)
matchmdl_range = matchmdl_range[sortidcs]
chi2_psf = chi2_psf[sortidcs]
matchangles = matchangles[sortidcs]

mdlidcs = np.concatenate((mcmcmdl_range[:bestof], matchmdl_range[:bestof]))
# mdlidcs = matchmdl_range[:bestof]



def read_modelmatches(lens, filedir='models/matches/', bestof=51,
                      field_types=[str, float, float]):
    filename = '{}_best{bestof}_{bestof}.txt'.format(lens, bestof=bestof)
    filename = os.path.join(savedir, filename)
    with open(filename, 'rb') as f:
        text = f.readlines()
    data = []
    for line in text:
        if line.startswith('#'):
            continue
        fields = [field_types[i](l.strip()) for i, l in enumerate(line.split('\t'))]
        data.append(fields)
    return [np.array([d[i] for d in data]) for i in range(len(data[0]))]


if __name__ == "__main__":
    lm = load_lm(lens, update_pixrad=pixrad, verbose=False)
    # lm.rescale(zl_new=zl_mdl, zs_new=zs_mdl)
    dl = DL(lm.zl, lm.zs)
    dlsds = DLSDS(lm.zl, lm.zs)
    dldsdls = DLDSDLS(lm.zl, lm.zs)
    lfactor = (1.+lm.zl)
    sfactor = (1.+lm.zs)
    print("DL: {:6.4f}, DLSDS: {:6.4f}, DLDSDLS: {:6.4f}".format(dl, dlsds, dldsdls))
    print("DL factor: {}".format(lfactor))
    print("DS factor: {}".format(sfactor))
    print("Kappa factor: {}".format( np.pi * lfactor ))
    # print("DLSDS factor: {}".format( (1.+lm.zs)/(1.+lm.zl) ))
    # lsfactor = (1.+lm.zs)/(1.+lm.zl)
    print(lm.__v__)

    model_names, _, _ = read_modelmatches(lens, filedir=savedir, bestof=51)

    for i, mdl_idx in enumerate(mdlidcs):
        # i = i + bestof
        mdlname = model_names[i]
        mdlname = mdlname.split('.')[0].replace('/', '.')

        # Kappa map plot
        if KAPPA:
            kappa_map_plot(lm, mdl_index=mdl_idx, extent=lm.extent, oversample=True,
                           log=True, contours=True, levels=17, delta=0.1, factor=np.pi*lfactor)
            plot_scalebar(lm.maprad, length=max(int(lm.maprad/4), 1))
            plt.axis('off')
            plt.gcf().axes[0].get_xaxis().set_visible(False)
            plt.gcf().axes[0].get_yaxis().set_visible(False)
            savename = "{}_{}_{}_kappa.{}".format(lens, i, mdlname, extension)
            print(savename)
            plt.savefig(os.path.join(savedir, lens, savename),
                        transparent=True, bbox_inches='tight', pad_inches=0)
            # plt.show()
            plt.close()

            kappa_map_plot(lm.resampled['obj'], mdl_index=mdl_idx, extent=lm.extent,
                           log=True, contours=True, levels=17, delta=0.1, factor=np.pi*lfactor)
            plot_scalebar(lm.maprad, length=max(int(lm.maprad/4), 1))
            plt.axis('off')
            plt.gcf().axes[0].get_xaxis().set_visible(False)
            plt.gcf().axes[0].get_yaxis().set_visible(False)
            savename = "{}_{}_{}_kappa_pixrad{}.{}".format(
                lens, i, mdlname, pixrad, extension)
            print(savename)
            plt.savefig(os.path.join(savedir, lens, savename),
                        transparent=True, bbox_inches='tight', pad_inches=0)
            # plt.show()
            plt.close()

        # Arrival-time surface
        if ARRIV and i == 0:
            gfactor = 1./dlsds
            pfactor = np.pi * lfactor
            arrival_time_surface_plot(lm,  # .resampled['obj'],
                                      mdl_index=mdl_idx,
                                      geofactor=gfactor, psifactor=pfactor,
                                      draw_images=False, contours=True,
                                      levels=60,
                                      min_contour_shift=None,
                                      sad_contour_shift=None,
                                      colorbar=False, origin='lower',
                                      color='black')
            plot_scalebar(lm.maprad, length=max(int(lm.maprad/4), 1),
                          length_scale=sfactor, color='black')
            plt.axis('off')
            plt.gcf().axes[0].get_xaxis().set_visible(False)
            plt.gcf().axes[0].get_yaxis().set_visible(False)
            savename = "{}_{}_{}_arriv.{}".format(lens, i, mdlname, extension)
            print(savename)
            plt.savefig(os.path.join(savedir, lens, savename),
                        transparent=True, bbox_inches='tight', pad_inches=0)
            # plt.show()
            plt.close()

        # os.system('say "Model {} completed"'.format(i))

    if KPROF:
        # # MCMC
        kscale = lfactor**2
        rscale = 1.
        qtype = "mcmc"
        # lm_best = lm.subset(mdlidcs)
        lm_best = lm.subset(mdlidcs[:bestof])
        _, profiles, radii = kappa_profiles_plot(
            lm_best, ensemble_average=False, kfactor=kscale, rfactor=rscale,
            refined=True, interpolate=300,
            levels=14, as_range=True, maprad=lm.maprad, pixrad=lm.pixrad, adjust_limits=True,
            annotation_color='black', kappa1_line=True, einstein_radius_indicator=True,
            label_axes=True, fontsize=20)
        plt.xlim(right=0.6*np.max(radii))
        plt.tight_layout()
        r_E = find_einstein_radius(radii[0], profiles[0])
        savename = "{}_{}_best{}_rE@{:6.4f}_kprofile.{}".format(
            lens, qtype, bestof-1, r_E, extension)
        print(savename)
        print("Eintein radius:\t{:8.4f}".format(r_E))
        plt.savefig(os.path.join(savedir, lens, savename),
                    transparent=True, bbox_inches='tight', pad_inches=0)
        # plt.show()
        plt.close()

        # # Match
        kscale = lfactor**2
        rscale = 1. 
        qtype = "match"
        # lm_best = lm.subset(mdlidcs)
        lm_best = lm.subset(mdlidcs[bestof:])
        _, profiles, radii = kappa_profiles_plot(
            lm_best, ensemble_average=False, kfactor=kscale, rfactor=rscale,
            refined=True, interpolate=300,
            levels=14, as_range=True, maprad=lm.maprad, pixrad=lm.pixrad, adjust_limits=True,
            annotation_color='black', kappa1_line=True, einstein_radius_indicator=True,
            label_axes=True, fontsize=20)
        plt.xlim(right=0.6*np.max(radii))
        plt.tight_layout()
        r_E = find_einstein_radius(radii[0], profiles[0])
        savename = "{}_{}_best{}_rE@{:6.4f}_kprofile.{}".format(
            lens, qtype, bestof-1, r_E, extension)
        print(savename)
        print("Eintein radius:\t{:8.4f}".format(r_E))
        plt.savefig(os.path.join(savedir, lens, savename),
                    transparent=True, bbox_inches='tight', pad_inches=0)
        # plt.show()
        plt.close()


    os.system('say "All models have been processed"')

    message = \
              "\nSDSSJ0029-0055 \t r_E: {:4.2f}\n".format(0.9657) \
              + "SDSSJ0737+3216 \t r_E: {:4.2f}\n".format(1.0252) \
              + "SDSSJ0753+3416 \t r_E: {:4.2f}\n".format(1.3137) \
              + "SDSSJ0956+5100 \t r_E: {:4.2f}\n".format(1.3956) \
              + "SDSSJ1051+4439 \t r_E: {:4.2f}\n".format(1.4956) \
              + "SDSSJ1430+6104 \t r_E: {:4.2f}\n".format(1.1528) \
              + "SDSSJ1627-0053 \t r_E: {:4.2f}\n".format(1.2667)
    print(message)

    redshifts = [(0.2270, 0.9313), ]
