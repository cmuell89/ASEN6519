import numpy as np
from exponential_scaling import eexp, eln, elnsum, elnproduct
import colored_traceback
colored_traceback.add_hook()


class ForwardBackwardHMM():

    def __init__(self, transition_probs, evidence_probs, initial_probs, obs):
        self.n_states = len(initial_probs)
        self.init = initial_probs
        self.emis = evidence_probs
        self.trans = transition_probs
        self.n_obs = len(obs)
        self.obs = obs

    def max_likelihood_state_estimate(self, probs):
        probs = probs.transpose()
        states = [np.argmax(prob_vec) + 1 for prob_vec in probs]
        return states

    def forward_backward(self):
        """
        Implements the forward backward algorithm for a simple HMM example. This function calls 
        private methods _foward and _backwards to obtain the state probabilities for the forward
        and backward passes. They are then multiplied and normalized to get the full probabilities 
        for each state at each time step.
        """
        alphas = self._forward_iter()
        betas = self._backward_iter()

        # recast as np.arrays to perform element-wise multiplication
        probs = np.array(alphas)[:, 0:] * np.array(betas)[:, :self.n_obs + 1]
        probs = probs / np.sum(probs, 0)

        # for i in range(self.n_obs):
        #     print(np.sum(alphas[:, i] * betas[:, i]))

        return probs, alphas, betas

    def forward_backward_eln(self):
        logalphas = self._forward_iter_eln()
        logbetas = self._backward_iter_eln()
        probs = np.zeros((self.n_states, self.n_obs + 1))
        for k in range(0, self.n_obs + 1):
            norm = -np.inf
            for i in range(self.n_states):
                probs[i, k] = elnproduct(logalphas[i, k], logbetas[i, k])
                norm = elnsum(norm, probs[i, k])
            for i in range(self.n_states):
                probs[i, k] = eexp(elnproduct(probs[i, k], -norm))
        # for i in range(self.n_obs):
        #     print(sum([p for p in list(probs[:, i]) if p != -np.inf]))
        return probs, logalphas, logbetas

    def _forward(self):
        """
        The forward (filtering) pass starting from the initial time step to the ending time step. 
        alphas stores the calculated alpha value at each iteration normalized across states to avoid
        vanishing probabilities. The initial time step t_0 is initialized with the initial state 
        probabilities.
        """
        alphas = np.zeros((self.n_states, self.n_obs + 1))
        alphas[:, 0] = self.init
        for k in range(0, self.n_obs):
            alphas[:, k + 1] = alphas[:, k].dot(self.trans.transpose()) * self.emis[
                self.obs[k], :]
        return alphas

    def _backward(self):
        """
        The backward (smoothing) pass starting from the an arbitrary future time step to 
        the beginning time step. The matrix 'betas' stores the calculated beta value at each 
        iteration normalized across states to avoid vanishing probabilities.
        """
        betas = np.zeros((self.n_states, self.n_obs))
        betas[:, -1] = 1
        for k in range(self.n_obs, 0, -1):
            beta_vec = np.matrix(betas[:, k]).transpose()
            betas[:, k - 1] = (np.matrix(self.trans.transpose()) * np.matrix(np.diag(
                self.emis.transpose()[:, self.obs[k - 1]])) * beta_vec).transpose()
        return betas

    def _forward_iter(self):
        alphas = np.zeros((self.n_states, self.n_obs + 1))

        # base case
        alphas[:, 0] = self.init
        # recursive case
        for k in range(0, self.n_obs):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    alphas[i, k + 1] += alphas[j, k] * self.trans[i,
                                                                  j] * self.emis[self.obs[k], i]
        return alphas

    def _backward_iter(self):
        betas = np.zeros((self.n_states, self.n_obs + 1))
        # base case
        betas[:, -1] = 1
        # recursive case
        for k in range(self.n_obs, -1, -1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    betas[i, k - 1] += self.trans[j, i] * \
                        self.emis[self.obs[
                            k - 1], j] * betas[j, k]
        return betas

    def _forward_iter_eln(self):
        logalphas = np.zeros((self.n_states, self.n_obs + 1))

        # base case
        logalphas[:, 0] = [eln(x) for x in self.init]
        # recursive case
        for k in range(1, self.n_obs + 1):
            for j in range(self.n_states):
                logalpha = -np.inf
                for i in range(self.n_states):
                    logalpha = elnsum(logalpha, elnproduct(
                        logalphas[i, k - 1], eln(self.trans.transpose()[i, j])))
                logalphas[j, k] = elnproduct(logalpha, eln(
                    self.emis[self.obs[k - 1], j]))
        return logalphas

    def _backward_iter_eln(self):
        logbetas = np.zeros((self.n_states, self.n_obs + 1))
        # base case
        logbetas[:, -1] = 0
        # recursive case
        for k in range(self.n_obs, -1, -1):
            for i in range(self.n_states):
                logbeta = -np.inf
                for j in range(self.n_states):
                    logbeta = elnsum(logbeta, elnproduct(eln(self.trans.transpose()[i, j]), elnproduct(eln(
                        self.emis[self.obs[k], j]), logbetas[j, k])))
                logbetas[i, k - 1] = logbeta
        return logbetas
