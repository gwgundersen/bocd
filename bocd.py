"""============================================================================
Author: Gregory Gundersen

Python implementation of Bayesian online changepoint detection for a normal
model with unknown mean parameter. For details, see Adams & MacKay 2007:

    "Bayesian Online Changepoint Detection"
    https://arxiv.org/abs/0710.3742

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


# -----------------------------------------------------------------------------

def bocd(data, model, hazard):
    """Return run length posterior using Algorithm 1 in Adams & MacKay 2007.
    """
    # 1. Initialize lower triangular matrix representing the posterior as
    # function of time. Model parameters are initialized in the model class.
    R       = np.zeros((T+1, T+1))
    R[0, 0] = 1
    message = np.array([1])

    for t in range(1, T+1):
        # 2. Observe new datum.
        x = data[t-1]

        # 3. Evaluate predictive probabilities.
        pis = model.pred_prob(t, x)

        # 4. Calculate growth probabilities.
        growth_probs = pis * message * (1 - hazard)

        # 5. Calculate changepoint probabilities.
        cp_prob = sum(pis * message * hazard)

        # 6. Calculate evidence
        new_joint = np.append(cp_prob, growth_probs)
        evidence  = np.sum(new_joint)

        # 7. Determine run length distribution.
        R[t, :t+1] = new_joint
        if evidence > 0:
            R[t, :t+1] /= evidence

        # 8. Update sufficient statistics.
        model.update_params(t, x)

        # Setup message passing.
        message = new_joint

    return R


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

    def pred_prob(self, t, x):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        preds = np.empty(t)
        for n in range(t):
            # Posterior predictive: see eq. 40 in (Murphy 2007).
            mean_n   = self.mean_params[n]
            var_n    = 1. / self.prec_params[n]
            preds[n] = norm.pdf(x, mean_n, var_n + self.varx)
        return preds
    
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

def plot_posterior(T, data, R, cps):
    """Plot data, run length posterior, and groundtruth changepoints.
    """
    fig, axes = plt.subplots(2, 1, figsize=(20,10))

    ax1, ax2 = axes

    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)

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
    T      = 300   # Number of observations.
    hazard = 1/50  # Constant prior on changepoint probability.
    mean0  = 0     # The prior mean on the mean parameter.
    var0   = 3     # The prior variance for mean parameter.
    varx   = 1     # The known variance of the data.

    data, cps = generate_data(varx, mean0, var0, T, hazard)
    model     = GaussianUnknownMean(mean0, var0, varx)
    R         = bocd(data, model, hazard)

    plot_posterior(T, data, R, cps)
