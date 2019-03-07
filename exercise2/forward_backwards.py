import numpy as np
from observations import y_obs_long, y_obs_short
from parameters import pxk_xkm1, pyk_xk, px0
from exponential_scaling import eexp, eln, elnsum, elnproduct
import matplotlib.pyplot as plt
from tabulate import tabulate
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

        probs = np.zeros((self.n_states, self.n_obs))
        for k in range(0, self.n_obs):
            norm = -np.inf
            for i in range(self.n_states):
                probs[i, k] = elnproduct(logalphas[i, k + 1], logbetas[i, k])
                norm = elnsum(norm, probs[i, k])
            for i in range(self.n_states):
                probs[i, k] = eexp(elnproduct(probs[i, k], -norm))
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
        for k in range(self.n_obs, -1, -1):
            for i in range(self.n_states):
                logbeta = -np.inf
                for j in range(self.n_states):
                    logbeta = elnsum(logbeta, elnproduct(eln(self.trans.transpose()[i, j]), elnproduct(eln(
                        self.emis[self.obs[k - 1], j]), logbetas[j, k])))
                logbetas[i, k - 1] = logbeta
        return logbetas


def short_state_trace():
    fbhmm = ForwardBackwardHMM(pxk_xkm1, pyk_xk, px0, y_obs_short)
    probs, alphas, betas = fbhmm.forward_backward()
    probs_eln, logalphas, logbetas = fbhmm.forward_backward_eln()
    state_trace = fbhmm.max_likelihood_state_estimate(probs)
    eln_state_trace = fbhmm.max_likelihood_state_estimate(probs_eln)
    K = [i for i in range(0, len(y_obs_short))]

    plt.plot(K, state_trace, marker='+', markersize='12', linestyle='-.', color='b', linewidth=2, drawstyle='steps-mid', label='Regular FB')

    plt.plot(K, eln_state_trace, marker='x', markersize='12', linestyle=':', color='r', linewidth=2, drawstyle='steps-mid', label='ELN FB')
    yint = range(1, 5)
    plt.yticks(yint)
    plt.legend(handlelength=6)
    plt.title("State transition chart for Short Dataset")
    plt.ylabel("States")
    plt.xlabel("Timestamp K")
    plt.show()


def posterior_chart():
    fbhmm = ForwardBackwardHMM(pxk_xkm1, pyk_xk, px0, y_obs_short)
    probs, alphas, betas = fbhmm.forward_backward()
    probs_eln, logalphas, logbetas = fbhmm.forward_backward_eln()
    state_trace = fbhmm.max_likelihood_state_estimate(probs)
    eln_state_trace = fbhmm.max_likelihood_state_estimate(probs_eln)


def data_likelihood():
    fbhmm = ForwardBackwardHMM(pxk_xkm1, pyk_xk, px0, y_obs_short)
    probs, alphas, betas = fbhmm.forward_backward()
    print(eln(sum(alphas[:, len(y_obs_short)])))

if __name__ == "__main__":
    # state_trace()
    data_likelihood()
