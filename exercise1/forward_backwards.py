import numpy as np
from observations import y_obs_long, y_obs_short
from parameters import pxk_xkm1, pyk_xk, px0


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
        alphas = self._forward_iter()
        betas = self._backward()

        # recast as np.arrays to perform element-wise multiplication
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
        for ob_idx in range(self.n_observations):
            alpha_vec = np.matrix(alphas[:, ob_idx])
            alphas[:, ob_idx + 1] = alpha_vec * np.matrix(self.trans_probs.transpose()) * np.matrix(
                np.diag(self.ev_probs.transpose()[:, self.observations[ob_idx]]))
            # normalize
            alphas[:, ob_idx + 1] = alphas[:, ob_idx + 1] / \
                np.sum(alphas[:, ob_idx + 1])
        return alphas[:, 1:self.n_observations + 1]

    def _forward_iter(self):
        alphas = np.zeros((self.n_observations, self.n_states))

        # base case
        alphas[0, :] = self.init_probs
        # recursive case
        for i in range(1, self.n_observations):
            for s2 in range(self.n_states):
                for s1 in range(self.n_states):
                    alphas[i, s2] += alphas[i - 1, s1] * self.trans_probs[s2,
                                                                          s1] * self.ev_probs.transpose()[s2, self.observations[i]]
            alphas[i - 1, :] = alphas[i - 1, :] / np.sum(alphas[i - 1, :])
        return alphas.transpose()

    def _backward(self):
        """
        The backward (smoothing) pass starting from the an arbitrary future time step to 
        the beginning time step. The matrix 'betas' stores the calculated beta value at each 
        iteration normalized across states to avoid vanishing probabilities.
        """
        betas = np.zeros((self.n_states, self.n_observations + 1))
        betas[:, -1] = 1
        for ob_idx in range(self.n_observations, 0, -1):
            beta_vec = np.matrix(betas[:, ob_idx]).transpose()
            betas[:, ob_idx - 1] = (np.matrix(self.trans_probs.transpose()) *
                                    np.matrix(np.diag(self.ev_probs.transpose()[:, self.observations[ob_idx - 1]])) *
                                    beta_vec).transpose()
            # normalize
            betas[:, ob_idx - 1] = betas[:, ob_idx - 1] / \
                np.sum(betas[:, ob_idx - 1])
        return betas[:, 0:self.n_observations]

if __name__ == "__main__":
    fbhmm = ForwardBackwardHMM(pxk_xkm1, pyk_xk, px0, y_obs_short)
    probs, alphas, betas = fbhmm.forward_backward()
    print(probs.transpose())
