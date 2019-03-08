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
import copy
import pudb

np.set_printoptions(precision=6, suppress=True)

def random_parameters():
    init = np.random.rand(4)
    init = init / init.sum(axis=0)
    trans = np.random.rand(4, 4)
    row_sums = trans.sum(axis=1)
    trans = trans / row_sums[:, np.newaxis]
    emis = np.random.rand(15, 4)
    row_sums = emis.sum(axis=0)
    emis = emis / row_sums[np.newaxis, :]
    return init, trans, emis


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

    def __init__(self, init_probs, init_trans, init_emis, sequences, n_iters=100):
        self.n_states = len(init)
        self.init = init_probs
        self.emis = init_emis
        self.trans = init_trans
        self.sequences = sequences if sequences.ndim > 1 else [sequences]
        self.n_iters = n_iters

    def run_EM(self):
        init_est = copy.deepcopy(self.init)
        trans_est = copy.deepcopy(self.trans)
        emis_est = copy.deepcopy(self.emis)
        for n in range(self.n_iters):
            elnxis, xis, elngammas, gammas = self._e_step(init_est, trans_est, emis_est)
            init_est, trans_est, emis_est = self._m_step_eln(elnxis, elngammas, init_est, trans_est, emis_est)
            # init_est, trans_est, emis_est = self._m_step(xis, gammas)
            
        return init_est, trans_est, emis_est

    def _e_step(self, init_est, trans_est, emis_est):
        """
        For each observation sequence, we store the elngamma and elnxi values. Thus we are running forward-backward with the current
        set of parameters for every sequence and storing the resulting log probabilities and log state-state+1 probabilities for each.

        These elngamma's and elnxi's are then passed to the maximization step.
        """
        elnxis = []
        xis = []
        elngammas = []
        gammas = []
        for idx, obs_seq in enumerate(self.sequences):
            elngamma, gamma, elnalpha, elnbeta = ForwardBackwardHMM(init_est, trans_est, emis_est, obs_seq).forward_backward_eln()
            elngammas.append(elngamma[:, :])
            gammas.append(gamma[:, :])
            elnxi, xi = self._eln_xi(elnalpha[:, :], elnbeta[:, :], trans_est, emis_est, obs_seq)
            elnxis.append(elnxi)
            xis.append(xi)
        return elnxis, xis, elngammas, gammas

    def _m_step(self, xis, gammas):
        """
        Maximization step built around the Mann log-space approach as well as the notes given by:
        https://people.eecs.berkeley.edu/~stephentu/writeups/hmm-baum-welch-derivation.pdf

        Each of the parameters are updating independently.
        """
        self._update_init(gammas)
        self._update_trans(gammas, xis)
        self._update_emis(gammas)

    def _m_step_eln(self, elnxis, elngammas, init_est, trans_est, emis_est):
        """
        Maximization step built around the Mann log-space approach as well as the notes given by:
        https://people.eecs.berkeley.edu/~stephentu/writeups/hmm-baum-welch-derivation.pdf

        Each of the parameters are updating independently.
        """
        init_est = self._update_init_eln(elngammas, init_est)
        trans_est = self._update_trans_eln(elngammas, elnxis, trans_est)
        emis_est = self._update_emis_eln(elngammas, emis_est)
        return init_est, trans_est, emis_est

    def _eln_xi(self, elnalpha, elnbeta, trans_est, emis_est, obs_seq):
        """
        This function calculates P(S_i_t, S_j_t+1) i.e. the probability of being in state S_i at time t and state S_j at time t+1.
        """
        elnxi = np.zeros((self.n_states, self.n_states, len(obs_seq)))
        xi = np.zeros((self.n_states, self.n_states, len(obs_seq)))
        for k in range(len(obs_seq)):
            normalizer = -np.inf
            for i in range(self.n_states):
                for j in range(self.n_states):
                    elnxi[i, j, k] = elnproduct(elnalpha[i, k], elnproduct(eln(trans_est.transpose()[i, j]), elnproduct(eln(emis_est[obs_seq[k], j]), elnbeta[j, k + 1])))
                    normalizer = elnsum(normalizer, elnxi[i, j, k])
            for i in range(self.n_states):
                for j in range(self.n_states):
                    elnxi[i, j, k] = elnproduct(elnxi[i, j, k], -normalizer)
                    xi[i, j, k] = eexp(elnxi[i, j, k])
        return elnxi, xi

    # def _update_init(self, gammas):
    #     """
    #     This function calculate the update value of the initial probabilities as defined by the Berkeley notes references in the _m_step algorithm. It is the average of the sum of probabilities of x_1 over all sequences. i.e. the expected frequency of state Si at time t = 1
    #     """
    #     for i in range(self.n_states):
    #         numerator = 0
    #         denominator = 0
    #         for idx in range(len(self.sequences)):
    #             numerator = numerator + gammas[idx][i, 1]
    #             denominator = denominator + 1
    #         self.init[i] = numerator / denominator

    def _update_init_eln(self, elngammas, init_est):
        """
        This function calculate the update value of the initial probabilities as defined by the Berkeley notes references in the _m_step algorithm. It is the average of the sum of probabilities of x_1 over all sequences. i.e. the expected frequency of state Si at time t = 1
        """
        for i in range(self.n_states):
            numerator = -np.inf
            denominator = 0
            for idx in range(len(self.sequences)):
                numerator = elnsum(numerator, elngammas[idx][i, 1])
                denominator += 1
            # protect against numerical instability
            if 1 - eexp(elnproduct(numerator, -eln(denominator))) < .000001:
                init_est[i] = 1
            else:
                init_est[i] = eexp(elnproduct(numerator, -eln(denominator)))
        return init_est

    # def _update_trans(self, gammas, xis):
    #     """
    #     This function calculates the update of the transition probabilities as defined by the Berkeley notes references in the _m_step algorithm.

    #     The numerator is the xi values summed across all sequences and across all time steps.
    #     The denominator is the gamma values summed across all sequences and across all time steps.
    #     """
    #     for i in range(self.n_states):
    #         for j in range(self.n_states):
    #             numerator = 0
    #             denominator = 0
    #             for idx, obs_seq in enumerate(self.sequences):
    #                 for k in range(1, len(obs_seq)):
    #                     # print(idx, k, xis[idx][i, j, k - 2])
    #                     numerator = numerator + xis[idx][i, j, k - 2]
    #                     denominator = denominator + gammas[idx][i, k - 1]
    #             # print(numerator, denominator)        
    #             self.trans[i, j] = numerator / denominator

    def _update_trans_eln(self, elngammas, elnxis, trans_est):
        """
        This function calculates the update of the transition probabilities as defined by the Berkeley notes references in the _m_step algorithm.

        The numerator is the eln_gamma values summed across all sequences and across all time steps.
        The denominator is the eln_gamma values summed across all sequences and across all time steps.
        """
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = -np.inf
                denominator = -np.inf
                for idx, obs_seq in enumerate(self.sequences):
                    for k in range(0, len(obs_seq)):
                        numerator = elnsum(numerator, elnxis[idx][i, j, k])
                        denominator = elnsum(denominator, elngammas[idx][i, k])
                trans_est[i, j] = eexp(elnproduct(numerator, -denominator))
        return trans_est

    # def _update_emis(self, gammas):
    #     """
    #     This function calculates the update of the transition probabilities as defined by the Berkeley notes references in the _m_step algorithm.

    #     The numerator is the gamma values summed across all sequences and across all time steps.
    #     The denominator is the gamma values summed across all sequences and across all time steps.
    #     """
    #     for e in range(self.emis.shape[0]):
    #         for i in range(self.n_states):
    #             numerator = 0
    #             denominator = 0
    #             for idx, obs_seq in enumerate(self.sequences):
    #                 for k in range(0, len(obs_seq)):
    #                     if obs_seq[k] == e:
    #                         numerator = numerator + gammas[idx][i, k]
    #                     denominator = denominator + gammas[idx][i, k]
    #             self.emis[e, i] = numerator / denominator

    def _update_emis_eln(self, elngammas, emis_est):
        """
        This function calculates the update of the transition probabilities as defined by the Berkeley notes references in the _m_step algorithm.

        The numerator is the eln_xi values summed across all sequences and across all time steps.
        The denominator is the eln_gamma values summed across all sequences and across all time steps.
        """
        for e in range(self.emis.shape[0]):
            for i in range(self.n_states):
                numerator = -np.inf
                denominator = -np.inf
                for idx, obs_seq in enumerate(self.sequences):
                    # shifted forward to ignore x0 with no emission.
                    for k in range(1, len(obs_seq) + 1):
                        if obs_seq[k - 1] == e:
                            numerator = elnsum(numerator, elngammas[idx][i, k])
                        denominator = elnsum(denominator, elngammas[idx][i, k])
                emis_est[e, i] = eexp(elnproduct(numerator, -denominator))
        return emis_est

if __name__ == "__main__":
    obs_sequences = import_data()["nominal_hmm_multi_logs"]
    init, trans, emis = random_parameters()

    print(np.sum(init, axis=0))
    print(np.sum(trans, axis=1))

    print(np.sum(emis, axis=0))

    bwhmm = BaumWelchHMM(init, trans, emis, obs_sequences[0:50], n_iters=50)
    init, trans, emis = bwhmm.run_EM()
    print("INITIAL PROBS")
    print(init)
    print(np.sum(init, axis=0))
    print(px0)
    print(np.sum(px0, axis=0))
    print("\n\nTRANSITION PROBS")
    print(trans.transpose())
    print(np.sum(trans, axis=1))
    print(pxk_xkm1)
    print(np.sum(pxk_xkm1, axis=0))
    print("\n\nEMISSION PROBS")
    print(emis)
    print(np.sum(emis, axis=0))
    print(pyk_xk)
    print(np.sum(pyk_xk, axis=0))
