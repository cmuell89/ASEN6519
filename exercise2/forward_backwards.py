import numpy as np
from exponential_scaling import eexp, eln, elnsum, elnproduct
import colored_traceback
colored_traceback.add_hook()


class ForwardBackwardHMM():

    def __init__(self, initial_probs, transition_probs, evidence_probs, obs):
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

    def forward_backward_eln(self):
        elnalpha = self._forward_iter_eln()
        elnbeta = self._backward_iter_eln()
        elngamma, gamma = self._generate_gamma(elnalpha, elnbeta)
        # for i in range(self.n_obs):
        #     print(sum([p for p in list(probs[:, i]) if p != -np.inf]))
        return elngamma, gamma, elnalpha, elnbeta

    def _generate_gamma(self, elnalpha, elnbeta):
        elngamma = np.zeros((self.n_states, self.n_obs + 1))
        gamma = np.zeros((self.n_states, self.n_obs + 1))
        for k in range(0, self.n_obs + 1):
            norm = -np.inf
            for i in range(self.n_states):
                elngamma[i, k] = elnproduct(elnalpha[i, k], elnbeta[i, k])
                norm = elnsum(norm, elngamma[i, k])
            for i in range(self.n_states):
                elngamma[i, k] = elnproduct(elngamma[i, k], -norm)
                gamma[i, k] = eexp(elngamma[i, k])
        return elngamma, gamma

    def _forward_iter_eln(self):
        elnalpha = np.zeros((self.n_states, self.n_obs + 1))

        # base case
        elnalpha[:, 0] = [eln(x) for x in self.init]
        # recursive case
        for k in range(1, self.n_obs + 1):
            for j in range(self.n_states):
                logalpha = -np.inf
                for i in range(self.n_states):
                    logalpha = elnsum(logalpha, elnproduct(
                        elnalpha[i, k - 1], eln(self.trans.transpose()[i, j])))
                elnalpha[j, k] = elnproduct(logalpha, eln(
                    self.emis[self.obs[k - 1], j]))
        return elnalpha

    def _backward_iter_eln(self):
        elnbeta = np.zeros((self.n_states, self.n_obs + 1))
        # base case
        elnbeta[:, -1] = 0
        # recursive case
        for k in range(self.n_obs, 0, -1):
            for i in range(self.n_states):
                beta = -np.inf
                for j in range(self.n_states):
                    beta = elnsum(beta, elnproduct(eln(self.trans.transpose()[i, j]), elnproduct(eln(
                        self.emis[self.obs[k - 1], j]), elnbeta[j, k])))
                elnbeta[i, k - 1] = beta
        return elnbeta
