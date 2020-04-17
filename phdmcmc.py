import sys
import numpy as np

def flat_chain(acc, rej, probs, priors, n_acc, discard=0):
    """
    Utility function to convert MCMC run results into flattened arrays

    Note:
        - Simply takes the entire output of <mcmc_mh>
    """
    # dimensions
    nwalkers, iters, ndims = acc.shape
    N_acc = sum(n_acc)
    # create masks
    amsk = np.zeros_like(acc, dtype=bool)
    rmsk = np.zeros_like(rej, dtype=bool)
    pmsk = np.zeros_like(probs, dtype=bool)
    for w in range(nwalkers):
        amsk[w, int(n_acc[w]*discard):n_acc[w], :] = True
        rmsk[w, :(iters-n_acc[w]), :] = True
        pmsk[w, int(n_acc[w]*discard):n_acc[w]] = True
    # flatten
    acc_arr = acc[amsk].reshape(N_acc-sum((n_acc*discard).astype(int)), ndims)
    rej_arr = rej[rmsk].reshape(iters*nwalkers - N_acc, ndims)
    pbs_arr = probs[pmsk]
    prs_arr = priors[pmsk]
    return acc_arr, rej_arr, pbs_arr, prs_arr

def proposal_lorenzian(theta, accepted=[], stepsize=1.0):
    """
    Propose new set of parameters for Metropolis-Hastings algorithm from a Lorentzian distribution
    """
    return np.random.standard_cauchy(theta.shape)

def proposal_norm(theta, accepted=[], stepsize=1.0):
    """
    Propose new set of parameters for Metropolis-Hastings algorithm from a Gaussian distribution
    """
    return np.random.uniform(low=theta-stepsize, high=theta+stepsize, size=theta.shape)

def proposal(theta, accepted=[], stepsize=1.0):
    """
    Propose new set of parameters for Metropolis-Hastings algorithm from a Uniform distribution
    """
    return np.random.normal(theta, stepsize, theta.shape)


def acceptance(p, p_new):
    """
    Accept or reject 

    Args:
        l <float> - log-likelihood from a set of parameters
        l_new <float> - log-likelihood from a new set of parameters
    """
    if not np.isfinite(p_new):
        return False
    if p_new > p:
        return True
    else:
        uniform = np.random.uniform(0, 1)
        alpha = np.exp(p_new - p)
        return uniform < alpha


def metropolis_hastings(log_lh, log_p, theta, lp=None, llh=None, accepted=[],
                        args=(), proposal=proposal, acceptance=acceptance,
                        stepsize=None, nwalkers=1, iterations=100,
                        verbose=False, **kwargs):
    """
    Args:
        log_lh <callable> - return log-likelihood that the parameters generated the data
        log_p <callable> - return log-likelihood regardless of evidence, i.e. the prior
        theta <np.ndarray> - set of parameters from the previous MCMC step

    Kwargs:
        lp <float> - log-prior from the previous MCMC step
        llh <float> - log-likelihood from the previous MCMC step
        args <tuple/list> - additional arguments for log_p
        proposal <callable> - return a proposal for new parameters
        acceptance <callable> - return acceptance of new sample given an old one
        stepsize <float> - stepsize for the proposal function
        iterations <int> - number of steps
        verobse <bool/int> - print status to command-line, if ==1 stdout is flushed
    """
    # sample new parameter proposal
    theta_new = proposal(theta, accepted=accepted, stepsize=stepsize)
    # priors
    if not np.isfinite(lp):
        lp, theta = log_p(theta, **kwargs)
    lp_new, theta_new = log_p(theta_new, **kwargs)
    # likelihoods
    if not np.isfinite(llh):
        llh = log_lh(theta, args=args, **kwargs)
    llh_new = log_lh(theta_new, args=args, **kwargs)
    # accept or reject?
    accepted = acceptance(lp+llh, lp_new+llh_new)
    return theta, theta_new, lp_new, llh_new, accepted


def mcmc_mh(log_lh, log_p, init,
            args=(), proposal=proposal, acceptance=acceptance,
            stepsize=1.0, nwalkers=None, iterations=1000,
            verbose=False, **kwargs):
    """
    Args:
        log_lh <callable> - return log-likelihood that the parameters generated the data
        log_p <callable> - return log-likelihood regardless of evidence, i.e. the prior
        init <np.ndarray> - initial guess

    Kwargs:
        args <tuple/list> - additional arguments for log_p
        proposal <callable> - return a proposal for new parameters
        acceptance <callable> - return acceptance of new sample given an old one
        stepsize <float> - stepsize for the proposal function
        nwalkers <int> - number of MCMC walkers
        iterations <int> - number of steps
        verbose <bool/int> - print status to command-line, modes: [1, 2, 3, 4]

    Note:
        - verbose: the lower the verbose integer value the more info is outputted
    """
    ndims = init.shape[-1]
    if nwalkers is None:
        if len(init.shape) == 1:
            nwalkers = 1
        else:
            nwalkers = init.shape[0]
    if init.shape[0] != nwalkers and len(init.shape) > 1:
        raise ValueError("Dimensions of <init> don't match <nwalkers>!")
    pars = init.astype(np.float32)*1                    # current parameter guess
    lp = np.full(nwalkers, -np.inf, dtype=np.float32)   # current log-prior
    llh = np.full(nwalkers, -np.inf, dtype=np.float32)  # current log-likelihood
    acc_llhs = np.full((nwalkers, iterations), -np.inf, dtype=np.float32)         # accepted log-likelihoods
    acc_lprs = np.full((nwalkers, iterations), -np.inf, dtype=np.float32)         # accepted log-priors
    acc_pars = np.full((nwalkers, iterations, ndims), -np.inf, dtype=np.float32)  # accepted parameters
    rej_pars = np.full((nwalkers, iterations, ndims), -np.inf, dtype=np.float32)  # rejected parameters
    n_accepted = np.zeros(nwalkers, dtype=int)
    for i in range(iterations):
        for iwalker in range(nwalkers):
            # Metropolis-Hastings step: Proposal - Prior & Likelihood - Uniform(0,1) < alpha
            pars[iwalker, :], pars_new, lp_new, llh_new, accepted = metropolis_hastings(
                log_lh, log_p, pars[iwalker, :], lp=lp[iwalker], llh=llh[iwalker],
                args=args, proposal=proposal, acceptance=acceptance, stepsize=stepsize,
                **kwargs)
            if accepted:
                nacc = n_accepted[iwalker]
                pars[iwalker] = pars_new
                lp[iwalker] = lp_new
                llh[iwalker] = llh_new
                acc_lprs[iwalker, nacc] = lp_new
                acc_llhs[iwalker, nacc] = llh_new
                acc_pars[iwalker, nacc, :] = pars_new[:]
                n_accepted[iwalker] = nacc + 1
            else:
                nrej = i - n_accepted[iwalker]
                rej_pars[iwalker, nrej, :] = pars_new[:]
            # some verbosity
            if 0 < verbose < 3:
                np.set_printoptions(formatter={'float': '{: 7.3f}'.format})
                acc_perc = 100. * np.sum(n_accepted) / ((i+1) * nwalkers)
                message = "{:4d} / {:4d}: [{:4d}] \t theta {} \t logP {:+9.4f} \t acceptance {:6.2f}%\r".format(
                    i, iterations, iwalker, pars_new, llh_new, acc_perc)
                if verbose == 1:
                    sys.stdout.write(message)
                    sys.stdout.flush()
                elif verbose == 2:
                    print(message)
        if verbose == 3:
            np.set_printoptions(formatter={'float': '{: 7.3f}'.format})
            acc_perc = 100. * np.sum(n_accepted) / ((i+1) * nwalkers)
            message = "{:4d} / {:4d}: \t theta {} \t logP {:+9.4f} \t acceptance {:6.2f}%\r".format(
                    i, iterations, acc_pars[0, max(0, n_accepted[0]-1), :], acc_llhs[0, max(0, n_accepted[0]-1)], acc_perc)
            print(message)
        elif (verbose == 4) and ((i % 1000) == 0):
            np.set_printoptions(formatter={'float': '{: 7.3f}'.format})
            acc_perc = 100. * np.sum(n_accepted) / ((i+1) * nwalkers)
            message = "{:4d} / {:4d}: \t theta {} \t logP {:+9.4f} \t acceptance {:6.2f}%\r".format(
                    i, iterations, acc_pars[0, max(0, n_accepted[0]-1), :], acc_llhs[0, max(0, n_accepted[0]-1)], acc_perc)
            print(message)
    if verbose:
        print("")
    return acc_pars, rej_pars, acc_llhs, acc_lprs, n_accepted
