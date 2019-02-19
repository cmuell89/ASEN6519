import numpy as np
import os
from observations import y_obs_long, y_obs_short
from parameters import pxk_xkm1, pyk_xk, px0


class ForwardBackwardsHMM():

    def __init__(self, transition_probs, evidence_probs, initial_probs, observations):
        self.n_states = len(initial_probs)
        self.init_probs = initial_probs
        self.ev_probs = evidence_probs
        self.trans_probs = transition_probs
        self.n_observations = len(observations)
        self.observations = observations

    def foward_backwards(self):
        pass

    def _forwards(self):
        alphas = np.zeros((self.n_states, self.n_observations))

        # Initialization of f
        obs_idx = self.observations[0]
        for s in range(0, self.n_states):
            alphas[s, 0] = self.init_probs[s] * self.ev_probs[obs_idx, s]

        # alpha steps step
        # we ignore time step 1
        for t in range(1, self.n_observations):
            # get the index value given observation at time t.
            obs_idx = self.observations[t]
            for s in range(0, self.n_states):
                alphas[s, t] = self.ev_probs[obs_idx, s] * sum([alphas[s_plus_1, t - 1] * self.trans_probs[s_plus_1, s] for s_plus_1 in range(0, self.n_states)])

        return alphas

    def _backwards(self):
        pass


if __name__ == "__main__":
    print(pyk_xk.shape)
    fbhmm = ForwardBackwardsHMM(pxk_xkm1, pyk_xk, px0, y_obs_short)
    print(fbhmm._forwards())