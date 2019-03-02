import numpy as np
from numpy.random import choice
from observations import y_obs_long, y_obs_short
from parameters import pxk_xkm1, pyk_xk, px0
from exponential_scaling import eexp, eln, elnsum, elnproduct
import colored_traceback
colored_traceback.add_hook()


class LikelihoodSamplingInference():

    def __init__(self, transition_probs, evidence_probs, initial_probs, observations):
        self.n_states = len(initial_probs)
        self.init_probs = initial_probs
        self.ev_probs = evidence_probs
        self.trans_probs = transition_probs
        self.n_observations = len(observations)
        self.observations = observations

    def max_likelihood_state_estimate(self, probs):
        probs = probs
        states = [np.argmax(np.array(prob_vec)) + 1 for prob_vec in probs]
        return states

    def run_inference(self, n_samples=100):
        samples, weights = self.run_sampling(n_samples)
        probs = []
        for k in range(self.n_observations + 1):
            prob = []
            xk_stack = list(list(zip(*samples))[k])
            for s in range(self.n_states):
                idxs = [idx for idx, value in enumerate(xk_stack) if value == s]
                prob.append(sum([w for idx, w in enumerate(weights) if idx in idxs]) / sum(weights))
            probs.append(prob)
        return probs

    def run_sampling(self, n_samples):
        samples = []
        weights = []
        for n in range(n_samples):
            sample, weight = self.liklihood_sampling()
            samples.append(sample)
            weights.append(weight)
        return samples, weights

    def liklihood_sampling(self):
        w = 1
        X = []
        xkm1 = self.sample_state(self.init_probs)
        X.append(xkm1)
        for k in range(0, self.n_observations):
            xk = self.sample_state(list(self.trans_probs[:, xkm1]))
            ob_idx = self.observations[k]
            py_xk = self.ev_probs[ob_idx, xk]
            w = w * py_xk
            X.append(xk)
            xkm1 = xk
        return(X, w)

    def sample_state(self, probs):
        return choice(4, 1, p=probs)[0]