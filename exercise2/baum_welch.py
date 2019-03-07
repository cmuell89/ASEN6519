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

    def __init__(self, init, init_trans, init_emis, sequences):
        self.n_states = len(init)
        self.init = init
        self.emis = init_emis
        self.trans = init_trans
        self.sequences = sequences if sequences.ndim > 1 else [sequences]

    def run_EM(self):
        elngammas = []
        elnalphas = []
        elnbetas = []
        elnxis = []
        for idx, obs_seq in enumerate(self.sequences):
            elngamma, elnalpha, elnbeta = self._e_step(obs_seq)
            self._m_step(elngamma, elnalpha, elnbeta, obs_seq)
        return self.init, self.trans, self.emis

    def _e_step(self, obs_seq):
        return ForwardBackwardHMM(self.init, self.trans, self.emis, obs_seq).forward_backward_eln()

    def _m_step(self, elngamma, elnalpha, elnbeta, obs_seq):
        elnxi = self._eln_xi(elnalpha, elnbeta, obs_seq)
        self._update_init(elngamma)
        self._update_trans(elngamma, elnxi, len(obs_seq))
        self._update_emis(elngamma, obs_seq)

    def _eln_xi(self, elnalpha, elnbeta, obs_seq):
        elnxi = np.zeros((self.n_states, self.n_states, len(obs_seq) - 1))
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

    def _update_init(self, elngamma):
        for i in range(self.n_states):
            self.init[i] = eexp(elngamma[i, 1])

    def _update_trans(self, elngamma, elnxi, n_obs):
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = -np.inf
                denominator = -np.inf
                for k in range(n_obs - 1):
                    numerator = elnsum(numerator, elnxi[i, j, k])
                    denominator = elnsum(denominator, elngamma[i, k])
                self.trans[i, j] = eexp(elnproduct(numerator, -denominator))

    def _update_emis(self, elngamma, obs_seq):
        for e in range(self.emis.shape[0]):
            for i in range(self.n_states):
                numerator = -np.inf
                denominator = -np.inf
                for k in range(len(obs_seq)):
                    if obs_seq[k] - 1 == e:
                        numerator = elnsum(numerator, elngamma[i, k])
                    denominator = elnsum(denominator, elngamma[i, k])
                self.emis[e, i] = eexp(elnproduct(numerator, -denominator))


if __name__ == "__main__":
    obs_sequences = import_data()["unknown_hmm_multi_logs"]
    init, trans, emis = uniform_parameters()
    bwhmm = BaumWelchHMM(init, trans, emis, obs_sequences[0:1])
    init, trans, emis = bwhmm.run_EM()
    print(trans)
    print(pxk_xkm1)