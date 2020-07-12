#!/usr/bin/env python
import sys
import os
import pprint
import parse
import shutil
import numpy as np
gleam_root = "/Users/phdenzel/gleam"
sys.path.append(gleam_root)
from gleam.utils.makedir import mkdir_p
from gleam.utils.encode import an_sort
from mcmc_eval import read_mcmctxt
from match_eval import read_matchlog



# Parameter settings
ids = ['SDSSJ0029-0055', 'SDSSJ0737+3216', 'SDSSJ0753+3416', 'SDSSJ0956+5100',
       'SDSSJ1051+4439', 'SDSSJ1430+6104', 'SDSSJ1627-0053']
idx = 0
lens = ids[idx]              # lens name

mdir = 'match_plots/{lens}/'.format(lens=lens)
sdir = os.path.join(mdir, 'sorted')
mcmcmdl_range, _, mcmcangles = read_mcmctxt('mcmc/mcmceval_{}.txt'.format(lens))
matchlog = '{}{}_matching.log'.format(mdir, lens)
matchmdl_range, chi2_psf, matchangles = read_matchlog(matchlog)


if __name__ == "__main__":

    sortidcs = np.argsort(chi2_psf)
    matchmdl_range = matchmdl_range[sortidcs]
    chi2_psf = chi2_psf[sortidcs]
    matchangles = matchangles[sortidcs]
    print(matchmdl_range)

    mkdir_p(sdir)

    match_files = an_sort(next(os.walk(mdir))[2])
    if '.DS_Store' in match_files:
        match_files.remove('.DS_Store')
    match_files.remove(os.path.basename(matchlog))
    for f in match_files:
        prsd = parse.parse('{lens}_mcmc{mcmc_idx:03d}_mdl{mdl_idx:03d}_rot{angle:9.4f}_{type}.png', f)
        if prsd is None:
            continue
        keys = prsd.named
        keys['match_idx'] = list(matchmdl_range).index(keys['mdl_idx'])
        # keys['chi2'] = chi2_psf[keys['match_idx']]
        savename = '{lens}_match{match_idx:03d}_mdl{mdl_idx:03d}_{type}.png'.format(**keys)
        savename = os.path.join(sdir, savename)
        fname = os.path.join(mdir, f)
        # print(fname, savename)
        shutil.copy2(fname, savename)
        
        
