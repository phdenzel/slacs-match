import sys
import numpy as np

def flat_chain(acc, rej, probs, priors, discard=0):
    """
    Utility function to convert MCMC run results into flattened arrays
    """
    acc_arr = np.asarray(acc)[:, int(discard*len(acc[0])):]
    rej_arr = np.asarray(rej)
    pbs_arr = np.asarray(probs)[:, int(discard*len(probs[0])):]
    prs_arr = np.asarray(priors)[:, int(discard*len(priors[0])):]
    return acc_arr.reshape(np.prod(acc_arr.shape[:-1]), acc_arr.shape[-1]), \
        rej_arr.reshape(np.prod(rej_arr.shape[:-1]), rej_arr.shape[-1]), \
        pbs_arr.reshape(np.prod(pbs_arr.shape[:-1]), pbs_arr.shape[-1]), \
        prs_arr.reshape(np.prod(prs_arr.shape[:-1]), prs_arr.shape[-1])


def proposal_normal(theta, accepted=[]):
    """
    Propose new set of parameters for Metropolis-Hastings algorithm
    """
    sample = np.random.normal
    scale = [0.5]*len(theta)
    return np.array([sample(t, scale[i]) for i, t in enumerate(theta)])


def proposal_normal2(theta, accepted=[]):
    """
    Propose new set of parameters for Metropolis-Hastings algorithm
    """
    sample = np.random.normal
    accepted = np.asarray(accepted)
    if len(accepted) < 1:
        scale = [0.5]*len(theta)
    elif len(accepted) < 10:
        scale = 0.5*accepted
    else:
        scale = np.std(accepted, axis=0)
    return np.array([sample(t, scale[i]) for i, t in enumerate(theta)])


def proposal_rvs(theta, **kwargs):
    """
    Propose new set of parameters for Metropolis-Hastings algorithm
    """
    pass


def proposal(theta, accepted=[], stepsize=1.0):
    """
    Propose new set of parameters for Metropolis-Hastings algorithm
    """
    return np.random.uniform(low=theta-stepsize, high=theta+stepsize, size=theta.shape)


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


def metropolis_hastings(log_lh, log_p, pars, lp=None, llh=None, accepted=[],
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
    theta_new = proposal(pars, accepted=accepted, stepsize=stepsize)
    # priors
    if lp is None:
        lp, pars = log_p(pars, **kwargs)
    lp_new, pars = log_p(theta_new, **kwargs)
    # likelihoods
    if llh is None:
        llh = log_lh(pars, args=args, **kwargs)
    llh_new = log_lh(theta_new, args=args, **kwargs)
    # accept or reject?
    accepted = acceptance(lp+llh, lp_new+llh_new)
    return pars, theta_new, lp_new, llh_new, accepted


def mcmc_mh(log_lh, log_p, init,
            args=(), proposal=proposal, acceptance=acceptance,
            stepsize=1.0, nwalkers=None, iterations=100,
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
        verobse <bool/int> - print status to command-line, if ==1 stdout is flushed
    """
    # ndims = init.shape[-1]
    if nwalkers is None:
        if len(init.shape) == 1:
            nwalkers = 1
        else:
            nwalkers = init.shape[0]
    if init.shape[0] != nwalkers and len(init.shape) > 1:
        raise ValueError("Dimensions of <init> don't match <nwalkers>!")
    theta = init*1             # current parameter guess
    lp = [None]*nwalkers       # current log-prior
    llh = [None]*nwalkers      # current log-likelihood
    llh_acc = [[]]*nwalkers    # accepted log-likelihoods
    lp_acc = [[]]*nwalkers     # accepted log-priors
    theta_acc = [[]]*nwalkers  # accepted parameters
    theta_rej = [[]]*nwalkers  # rejected parameters
    i = 0
    while i < iterations:
        for iwalker in range(nwalkers):
            # Metropolis-Hastings step: Proposal - Prior & Likelihood - Uniform(0,1) < alpha
            theta[iwalker], theta_new, lp_new, llh_new, accepted = metropolis_hastings(
                log_lh, log_p, theta[iwalker], lp=lp[iwalker], llh=llh[iwalker],
                args=args, proposal=proposal, acceptance=acceptance, stepsize=stepsize,
                **kwargs)
            if accepted:
                theta[iwalker] = theta_new
                lp[iwalker] = lp_new
                llh[iwalker] = llh_new
                lp_acc[iwalker].append(lp_new)
                llh_acc[iwalker].append(llh_new)
                theta_acc[iwalker].append(theta_new)
            else:
                theta_rej[iwalker].append(theta_new)
            # some verbosity
            if verbose:
                np.set_printoptions(formatter={'float': '{: 7.3f}'.format})
                N = sum([len(k) for k in theta_acc] + [len(k) for k in theta_rej])
                acc_p = (100. * sum([len(k) for k in theta_acc])) / N
                message = "{:4d} / {:4d}: [{:4d}] \t theta {} \t logP {:+9.4f} \t acceptance {:6.2f}%\r".format(
                    i, iterations, iwalker, theta_new, llh_new, acc_p)
                if verbose > 1:
                    print(message)
                else:
                    sys.stdout.write(message)
                    sys.stdout.flush()
        i += 1
    if verbose:
        print("")
    return theta_acc, theta_rej, llh_acc, lp_acc


def explore(log_lh, log_p,
            init, data, args=(),
            proposal=proposal, acceptance=acceptance, stepsize=None, iterations=100,
            verbose=False, **kwargs):
    """
    Args:
        log_lh <callable> - return log-likelihood that the parameters generated the data
        log_p <callable> - return log-likelihood regardless of evidence, i.e. the prior
        init <np.ndarray> - initial guess
        data <np.ndarray> - data

    Kwargs:
        args <tuple/list> - additional arguments for log_p
        proposal <callable> - return a proposal for new parameters
        acceptance <callable> - return acceptance of new sample given an old one
        stepsize <float> - stepsize for the proposal function
        iterations <int> - number of steps
        verobse <bool/int> - print status to command-line, if ==1 stdout is flushed

    """
    pass
