"""============================================================================
Author: Gregory Gundersen

Python implementation of Bayesian online changepoint detection for a normal
model with unknown mean parameter. For algorithm details, see

    Adams & MacKay 2007
    "Bayesian Online Changepoint Detection"
    https://arxiv.org/abs/0710.3742

For Bayesian inference details about the Gaussian, see:

    Murphy 2007
    "Conjugate Bayesian analysis of the Gaussian distribution"
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

This code is associated with the following blog posts:

    http://gregorygundersen.com/blog/2019/08/13/bocd/
    http://gregorygundersen.com/blog/2020/10/20/implementing-bocd/
============================================================================"""

import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import numpy as np
from   scipy.stats import norm
from   scipy.special import logsumexp


# -----------------------------------------------------------------------------

def bocd(data, model, hazard):
    """Return run length posterior using Algorithm 1 in Adams & MacKay 2007.
    """
    # 1. Initialize lower triangular matrix representing the posterior as
    #    function of time. Model parameters are initialized in the model class.
    #    
    #    When we exponentiate R at the end, exp(-inf) --> 0, which is nice for
    #    visualization.
    #
    log_R       = -np.inf * np.ones((T+1, T+1))
    log_R[0, 0] = 0              # log 0 == 1
    pmean       = np.empty(T)    # Model's predictive mean.
    pvar        = np.empty(T)    # Model's predictive variance. 
    log_message = np.array([0])  # log 0 == 1
    log_H       = np.log(hazard)
    log_1mH     = np.log(1 - hazard)

    for t in range(1, T+1):
        # 2. Observe new datum.
        x = data[t-1]

        # Make model predictions.
        pmean[t-1] = np.sum(np.exp(log_R[t-1, :t]) * model.mean_params[:t])
        pvar[t-1]  = np.sum(np.exp(log_R[t-1, :t]) * 1./model.prec_params[:t])
        
        # 3. Evaluate predictive probabilities.
        log_pis = model.log_pred_prob(t, x)

        # 4. Calculate growth probabilities.
        log_growth_probs = log_pis + log_message + log_1mH

        # 5. Calculate changepoint probabilities.
        log_cp_prob = logsumexp(log_pis + log_message + log_H)

        # 6. Calculate evidence
        new_log_joint = np.append(log_cp_prob, log_growth_probs)

        # 7. Determine run length distribution.
        log_R[t, :t+1]  = new_log_joint
        log_R[t, :t+1] -= logsumexp(new_log_joint)

        # 8. Update sufficient statistics.
        model.update_params(t, x)

        # Pass message.
        log_message = new_log_joint

    R = np.exp(log_R)
    return R, pmean, pvar


# -----------------------------------------------------------------------------


class GaussianUnknownMean:
    
    def __init__(self, mean0, var0, varx):
        """Initialize model.
        
        meanx is unknown; varx is known
        p(meanx) = N(mean0, var0)
        p(x) = N(meanx, varx)
        """
        self.mean0 = mean0
        self.var0  = var0
        self.varx  = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])
    
    def log_pred_prob(self, t, x):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.mean_params[:t]
        post_vars  = 1. / self.prec_params[:t] + self.varx
        return norm.logpdf(x, post_means, post_vars)
    
    def update_params(self, t, x):
        """Upon observing a new datum x at time t, update all run length 
        hypotheses.
        """
        # See eq. 19 in (Murphy 2007).
        new_prec_params  = self.prec_params + (1/self.varx)
        self.prec_params = np.append([1/self.var0], new_prec_params)
        # See eq. 24 in (Murphy 2007).
        new_mean_params  = (self.mean_params * self.prec_params[:-1] + \
                            (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)


# -----------------------------------------------------------------------------

def generate_data(varx, mean0, var0, T, cp_prob):
    """Generate partitioned data of T observations according to constant
    changepoint probability `cp_prob` with hyperpriors `mean0` and `prec0`.
    """
    data  = []
    cps   = []
    meanx = mean0
    for t in range(0, T):
        if np.random.random() < cp_prob:
            meanx = np.random.normal(mean0, var0)
            cps.append(t)
        data.append(np.random.normal(meanx, varx))
    return data, cps


# -----------------------------------------------------------------------------

def plot_posterior(T, data, cps, R, pmean, pvar):
    fig, axes = plt.subplots(2, 1, figsize=(20,10))

    ax1, ax2 = axes

    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)
    
    # Plot predictions.
    ax1.plot(range(0, T), pmean, c='k')
    _2std = 2 * np.sqrt(pvar)
    ax1.plot(range(0, T), pmean - _2std, c='k', ls='--')
    ax1.plot(range(0, T), pmean + _2std, c='k', ls='--')

    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', 
               norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, T])
    ax2.margins(0)

    for cp in cps:
        ax1.axvline(cp, c='red', ls='dotted')
        ax2.axvline(cp, c='red', ls='dotted')

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    T      = 500    # Number of observations.
    hazard = 1/100  # Constant prior on changepoint probability.
    mean0  = 0      # The prior mean on the mean parameter.
    var0   = 3      # The prior variance for mean parameter.
    varx   = 1      # The known variance of the data.

    data, cps      = generate_data(varx, mean0, var0, T, hazard)
    model          = GaussianUnknownMean(mean0, var0, varx)
    R, pmean, pvar = bocd(data, model, hazard)

    plot_posterior(T, data, cps, R, pmean, pvar)
