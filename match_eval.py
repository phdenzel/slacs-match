#!/usr/bin/env python
# Imports
import sys
import numpy as np
import scipy as sp
import parse
import pprint


def read_matchlog(filename,
                  strformat=('# {lens}_mcmc{mcmc_idx:03d}_mdl{mdl_idx:03d}_rot{angle:9.4f}',
                             'Chi2: {chi2:9.4f}'),
                  unpack_order=[[0, 2], [1, 0], [0, 3]],
                  verbose=False):
    with open(filename, 'rb') as f:
        txt = f.readlines()
    if isinstance(strformat, str):
        strformat = [strformat]
    match_log = [[] for i in range(len(strformat))]
    for line in txt:
        for pidx, sfrmt in enumerate(strformat):
            prsd = parse.parse(sfrmt, line)
            if prsd is not None:
                break
        if prsd is None:
            continue
        match_log[pidx].append(prsd.named.values())
    if unpack_order is None:
        return match_log
    else:
        outputs = []
        for iout in unpack_order:
            out = [elem[iout[1]] for elem in match_log[iout[0]]]
            outputs.append(out)
        return (np.array(o) for o in outputs)
