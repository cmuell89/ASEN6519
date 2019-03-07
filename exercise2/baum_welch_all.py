import numpy as np
from observations import y_obs_long, y_obs_short
from parameters import pxk_xkm1, pyk_xk, px0
from exponential_scaling import eexp, eln, elnsum, elnproduct
import matplotlib.pyplot as plt
# from tabulate import tabulate
import colored_traceback
colored_traceback.add_hook()
from forward_backwards import ForwardBackwardHMM
from import_hmm_data import import_data


def uniform_parameters():
    init = np.ones((4))
    init = init / init.sum(axis=0)
    trans = np.ones((4, 4))
    row_sums = trans.sum(axis=0)
    trans = trans / row_sums[:, np.newaxis]
    emis = np.ones((15, 4))
    row_sums = emis.sum(axis=0)
    emis = emis / row_sums[np.newaxis, :]
    return init, trans, emis


class BaumWelchHMM():

    def __init__(self, init, init_trans, init_emis, sequences, n_iters=1):
        self.n_states = len(init)
        self.init = init
        self.emis = init_emis
        self.trans = init_trans
        self.sequences = sequences if sequences.ndim > 1 else [sequences]
        self.n_iters = n_iters

    def run_EM(self):
        for n in range(self.n_iters):
            elnxis, elngammas = self._e_step()
            self._m_step(elnxis, elngammas)
        return self.init, self.trans, self.emis

    def _e_step(self):
        """
        For each observation sequence, we store the elngamma and elnxi values. Thus we are running forward-backward with the current
        set of parameters for every sequence and storing the resulting log probabilities and log state-state+1 probabilities for each.

        These elngamma's and elnxi's are then passed to the maximization step.
        """
        elnxis = []
        elngammas = []
        for idx, obs_seq in enumerate(self.sequences):
            elngamma, elnalpha, elnbeta = ForwardBackwardHMM(self.init, self.trans, self.emis, obs_seq).forward_backward_eln()
            elngammas.append(elngamma)
            elnxis.append(self._eln_xi(elnalpha, elnbeta, obs_seq))
        return elnxis, elngammas

    def _m_step(self, elnxis, elngammas):
        """
        Maximization step built around the Mann log-space approach as well as the notes given by:
        https://people.eecs.berkeley.edu/~stephentu/writeups/hmm-baum-welch-derivation.pdf

        Each of the parameters are updating independently.
        """
        self._update_init(elngammas)
        self._update_trans(elngammas, elnxis)
        self._update_emis(elngammas)

    def _eln_xi(self, elnalpha, elnbeta, obs_seq):
        """
        This function calculates P(S_i_t, S_j_t+1) i.e. the probability of being in state S_i at time t and state S_j at time t+1.
        """
        elnxi = np.zeros((self.n_states, self.n_states, len(obs_seq)))
        for k in range(len(obs_seq) - 1):
            normalizer = -np.inf
            for i in range(self.n_states):
                for j in range(self.n_states):
                    elnxi[i, j, k] = elnproduct(elnalpha[j, k], elnproduct(eln(self.trans.transpose()[i, j]), elnproduct(eln(self.emis[obs_seq[k + 1] - 1, j]), elnbeta[j, k + 1])))
                    normalizer = elnsum(normalizer, elnxi[i, j, k])
            for i in range(self.n_states):
                for j in range(self.n_states):
                    elnxi[i, j, k] = elnproduct(elnxi[i, j, k], -normalizer)
        return elnxi

    def _update_init(self, elngammas):
        for i in range(self.n_states):
            numerator = -np.inf
            denominator = 0
            for idx in range(len(self.sequences)):
                numerator = elnsum(numerator, elngammas[idx][i, 1])
                denominator = denominator + 1
            self.init[i] = eexp(elnproduct(numerator, -eln(denominator)))

    def _update_trans(self, elngammas, elnxis):
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = -np.inf
                denominator = -np.inf
                for idx, obs_seq in enumerate(self.sequences):
                    for k in range(0, len(obs_seq) - 1):
                        numerator = elnsum(numerator, elnxis[idx][i, j, k])
                        denominator = elnsum(denominator, elngammas[idx][i, k])
                self.trans[i, j] = eexp(elnproduct(numerator, -denominator))

    def _update_emis(self, elngammas):
        for e in range(self.emis.shape[0]):
            for i in range(self.n_states):
                numerator = -np.inf
                denominator = -np.inf
                for idx, obs_seq in enumerate(self.sequences):
                    for k in range(0, len(obs_seq) - 1):
                        if obs_seq[k] - 1 == e:
                            numerator = elnsum(numerator, elngammas[idx][i, k])
                        denominator = elnsum(denominator, elngammas[idx][i, k])
                self.emis[e, i] = eexp(elnproduct(numerator, -denominator))


if __name__ == "__main__":
    obs_sequences = import_data()["nominal_hmm_multi_logs"]
    init, trans, emis = uniform_parameters()
    bwhmm = BaumWelchHMM(init, trans, emis, obs_sequences[0:1])
    init, trans, emis = bwhmm.run_EM()
    print("INITIAL PROBS")
    print(init)
    print(px0)
    print("\n\nTRANSITION PROBS")
    print(trans)
    print(pxk_xkm1)
    print("\n\nEMISSION PROBS")
    print(emis)
    print(pyk_xk)
