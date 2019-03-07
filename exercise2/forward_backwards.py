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
        logalphas = self._forward_iter_eln()
        logbetas = self._backward_iter_eln()
        probs = np.zeros((self.n_states, self.n_obs + 1))
        for k in range(0, self.n_obs + 1):
            norm = np.inf
            for i in range(self.n_states):
                probs[i, k] = elnproduct(logalphas[i, k], logbetas[i, k])
                norm = elnsum(norm, probs[i, k])
            for i in range(self.n_states):
                probs[i, k] = elnproduct(probs[i, k], -norm)
        # for i in range(self.n_obs):
        #     print(sum([p for p in list(probs[:, i]) if p != -np.inf]))
        return probs, logalphas, logbetas

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
        for k in range(self.n_obs, 0, -1):
            for i in range(self.n_states):
                logbeta = -np.inf
                for j in range(self.n_states):
                    logbeta = elnsum(logbeta, elnproduct(eln(self.trans.transpose()[i, j]), elnproduct(eln(
                        self.emis[self.obs[k - 1], j]), logbetas[j, k])))
                logbetas[i, k - 1] = logbeta
        return logbetas
