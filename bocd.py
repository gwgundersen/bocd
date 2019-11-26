"""============================================================================
Python implementation of Bayesian online changepoint detection for a normal
model with unknown mean parameter. For details, see Adams & MacKay 2007:

    "Bayesian Online Changepoint Detection"
    https://arxiv.org/abs/0710.3742

This code implements the figure in the following blog post:

    http://gregorygundersen.com/blog/2019/08/13/bocd/

Author: Gregory Gundersen
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
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1
    message = np.array([1])

    for t in range(1, T + 1):

        # 2. Observe new datum.
        x = data[t - 1]

        # 3. Evaluate predictive probabilities.
        pis = model.pred_prob(t, x)

        # 4. Calculate growth probabilities.
        growth_probs = pis * message * (1 - hazard)

        # 5. Calculate changepoint probabilities.
        cp_prob = sum(pis * message * hazard)

        # 6. Calculate evidence
        new_joint = np.append(cp_prob, growth_probs)

        # 7. Determine run length distribution.
        R[t, :t + 1] = new_joint
        evidence = sum(new_joint)
        R[t, :] /= evidence

        # 8. Update sufficient statistics.
        model.update_statistics(t, x)

        # Setup message passing.
        message = new_joint

    return R


# -----------------------------------------------------------------------------

# Implementation of a Gaussian model with known precision. See Kevin Murphy's
# "Conjugate Bayesian analysis of the Gaussian distribution" for a complete
# derivation of the model:
#
#     https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
#
class NormalKnownPrecision:

    def __init__(self, mean0, prec0):
        """Initialize model parameters.
        """
        self.mean0 = mean0
        self.prec0 = prec0
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([prec0])

    def pred_prob(self, t, x):
        """Compute predictive probabilities.
        """
        d = lambda x, mu, tau: norm.pdf(x, mu, 1 / tau + 1)
        return np.array([d(x, self.mean_params[i], self.prec_params[i])
                         for i in range(t)])

    def update_statistics(self, t, x):
        """Update sufficient statistics.
        """
        # `offsets` is just a clever way to +1 all the sufficient statistics.
        offsets = np.arange(1, t + 1)
        new_mean_params = (self.mean_params * offsets + x) / (offsets + 1)
        new_prec_params = self.prec_params + 1
        self.mean_params = np.append([self.mean0], new_mean_params)
        self.prec_params = np.append([self.prec0], new_prec_params)


# -----------------------------------------------------------------------------

def generate_data(mean0, prec0, T, cp_prob):
    """Generate partitioned data of T observations according to constant
    changepoint probability `cp_prob` with hyperpriors `mean0` and `prec0`.
    """
    means = [0]
    data = []
    cpts = []
    for t in range(0, T):
        if np.random.random() < cp_prob:
            mean = np.random.normal(mean0, 1 / prec0)
            means.append(mean)
            cpts.append(t)
        data.append(np.random.normal(means[-1], 1))
    return data, cpts


# -----------------------------------------------------------------------------

def plot_posterior(T, data, R, cpts):
    """Plot data, run length posterior, and groundtruth changepoints.
    """
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    ax1, ax2 = axes

    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)

    norm = LogNorm(vmin=0.0001, vmax=1)
    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', norm=norm)
    ax2.set_xlim([0, T])
    # This just reverses the y-tick marks.
    ticks = list(range(0, T+1, 50))
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(ticks[::-1])
    ax2.margins(0)

    for cpt in cpts:
        ax1.axvline(cpt, c='r', ls='dotted')
        ax2.axvline(cpt, c='r', ls='dotted')

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    T = 300         # Number of observations.
    cp_prob = 1/50  # Constant prior on changepoint probability.
    mean0 = 0       # Prior on Gaussian mean.
    prec0 = 0.2     # Prior on Gaussian precision.

    data, cpts = generate_data(mean0, prec0, T, cp_prob)
    model = NormalKnownPrecision(mean0, prec0)
    R = bocd(data=data, model=model, hazard=1/50)
    # The model becomes numerically unstable for large `T` because the mass is
    # distributed across a support whose size is increasing.
    for row in R:
        assert np.isclose(np.sum(row), 1)
    plot_posterior(T, data, R, cpts)
