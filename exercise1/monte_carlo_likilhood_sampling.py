import numpy as np
from numpy.random import choice
from observations import y_obs_long, y_obs_short
from parameters import pxk_xkm1, pyk_xk, px0
import colored_traceback
colored_traceback.add_hook()


class LiklihoodSamplingInference():

    def __init__(self, transition_probs, evidence_probs, initial_probs, observations, n_samples):
        self.n_states = len(initial_probs)
        self.init_probs = initial_probs
        self.ev_probs = evidence_probs
        self.trans_probs = transition_probs
        self.n_observations = len(observations)
        self.observations = observations
        self.n_samples = n_samples

    def sample_state(self, km1):
        return choice(4, 1, p=list(self.trans_probs[:, km1]))[0]

    def liklihood_sampling(self):
        w = 1
        sample = []
        xkm1 = self.sample_state(0)
        sample.append(xkm1)
        for k in range(1, self.n_observations):
            xk = self.sample_state(xkm1)
            sample.append(xk)
            xkm1 = xk

# http://www.cse.psu.edu/~rtc12/CSE598C/samplingSlides.pdf

if __name__ == "__main__":
    lli = LiklihoodSamplingInference(pxk_xkm1, pyk_xk, px0, y_obs_short, 10)
    lli.liklihood_sampling()