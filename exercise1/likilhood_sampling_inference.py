import numpy as np
from numpy.random import choice
from observations import y_obs_long, y_obs_short
from parameters import pxk_xkm1, pyk_xk, px0
from exponential_scaling import eexp, eln, elnsum, elnproduce
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

    def run_inference(self):
        samples = []
        weights = []
        for n in range(self.n_samples):
            sample, weight = self.liklihood_sampling()
            samples.append(sample)
            weights.append(weight)
        for idx, sample in enumerate(samples):
            print(sample, weights[idx])
        probs = []
        for k in range(self.n_observations):
            prob = []
            xk_stack = list(zip(*samples))[k]
            for s in range(self.n_states):
                idxs = [idx for idx, value in enumerate(xk_stack) if value == s]
                prob.append(sum([weight for idx, weight in enumerate(weights) if idx in idxs]) / self.n_samples)
            probs.append(prob)

    def liklihood_sampling(self):
        w = 1
        X = []
        xkm1 = self.sample_state(0, self.init_probs)
        X.append(xkm1)
        for k in range(0, self.n_observations):
            xk = self.sample_state(xkm1, list(self.trans_probs[:, xkm1]))
            ob_idx = self.observations[k]
            py_xk = self.ev_probs[ob_idx, xk]
            print(w)
            w = w * py_xk
            X.append(xk)
            xkm1 = xk
        return(X, w)

    def sample_state(self, km1, probs):
        return choice(4, 1, p=probs)[0]

    


# http://www.cse.psu.edu/~rtc12/CSE598C/samplingSlides.pdf

if __name__ == "__main__":
    lli = LiklihoodSamplingInference(pxk_xkm1, pyk_xk, px0, y_obs_short, 10)
    lli.run_inference()