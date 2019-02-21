import numpy as np
from observations import y_obs_long, y_obs_short
from parameters import pxk_xkm1, pyk_xk, px0
import colored_traceback
colored_traceback.add_hook()


class ForwardBackwardHMM():

    def __init__(self, transition_probs, evidence_probs, initial_probs, observations):
        self.n_states = len(initial_probs)
        self.init_probs = initial_probs
        self.ev_probs = evidence_probs
        self.trans_probs = transition_probs
        self.n_observations = len(observations)
        self.observations = observations

    def forward_backward(self):
        """
        Implements the forward backward algorithm for a simple HMM example. This function calls 
        private methods _foward and _backwards to obtain the state probabilities for the forward
        and backward passes. They are then multiplied and normalized to get the full probabilities 
        for each state at each time step.
        """
        alphas = self._forward()
        print(alphas.transpose())
        betas = self._backward()
        print(betas.transpose())
        # recast as np.arrays to perform element-wise multiplication
        for i in range(self.n_observations):
            print(np.sum(alphas[:, i] * betas[:, i]))
        probs = np.array(alphas) * np.array(betas)
        probs = probs / np.sum(probs, 0)

        return probs, alphas, betas

    def _forward(self):
        """
        The forward (filtering) pass starting from the initial time step to the ending time step. 
        alphas stores the calculated alpha value at each iteration normalized across states to avoid
        vanishing probabilities. The initial time step t_0 is initialized with the initial state 
        probabilities. 
        """
        alphas = np.zeros((self.n_states, self.n_observations + 1))
        alphas[:, 0] = self.init_probs
        for k in range(0, self.n_observations):
            # alphas[:, k] = np.matrix(alphas[:, k - 1]) * np.matrix(self.trans_probs) * np.matrix(
            #     np.diag(self.ev_probs.transpose()[:, self.observations[k]]))
            alphas[:, k + 1] = alphas[:, k].dot(self.trans_probs.transpose()) * self.ev_probs[self.observations[k], :]

        return alphas

    def _forward_iter(self):
        alphas = np.zeros((self.n_states, self.n_observations + 1))

        # base case
        alphas[:, 0] = self.init_probs
        # recursive case
        for k in range(0, self.n_observations):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    alphas[i, k + 1] += alphas[j, k] * self.trans_probs[i, j] * self.ev_probs[self.observations[k], i]
        return alphas

    def _backward(self):
        """
        The backward (smoothing) pass starting from the an arbitrary future time step to 
        the beginning time step. The matrix 'betas' stores the calculated beta value at each 
        iteration normalized across states to avoid vanishing probabilities.
        """
        betas = np.zeros((self.n_states, self.n_observations + 1))
        betas[:, -1] = 1
        for k in range(self.n_observations, 1, -1):
            beta_vec = np.matrix(betas[:, k]).transpose()
            betas[:, k - 1] = (np.matrix(self.trans_probs.transpose()) *
                               np.matrix(np.diag(self.ev_probs.transpose()[:, self.observations[k - 1]])) *
                               beta_vec).transpose()
        return betas[:, 0:self.n_observations + 1]

if __name__ == "__main__":
    fbhmm = ForwardBackwardHMM(pxk_xkm1, pyk_xk, px0, y_obs_short)
    probs, alphas, betas = fbhmm.forward_backward()
