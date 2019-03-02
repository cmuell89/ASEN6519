from forward_backwards import ForwardBackwardHMM
from viterbi import ViterbiHMM
from observations import y_obs_long, y_obs_short
from parameters import pxk_xkm1, pyk_xk, px0
from exponential_scaling import eexp, eln, elnsum, elnproduct
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

if plt.get_backend() == 'Qt5Agg':
    from matplotlib.backends.qt_compat import QtWidgets
    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()
plt.rcParams.update({'font.size': 22})


def state_trace():
    fbhmm = ForwardBackwardHMM(pxk_xkm1, pyk_xk, px0, y_obs_short)
    probs_eln, logalphas, logbetas = fbhmm.forward_backward_eln()
    exact_state_trace = fbhmm.max_likelihood_state_estimate(probs_eln)
    vhmm = ViterbiHMM(pxk_xkm1, pyk_xk, px0, y_obs_short)
    path, log_probs = vhmm.viterbi_path()
    
    K = [i for i in range(0, len(y_obs_short) + 1)]

    plt.plot(K, exact_state_trace, marker='+', markersize='12', linestyle='-.', color='b', linewidth=2, drawstyle='steps-mid', label='Exact Inference')

    plt.plot(K, path, marker='x', markersize='12', linestyle=':', color='r', linewidth=2, drawstyle='steps-mid', label='Viterbi')
    yint = range(1, 5)
    plt.yticks(yint)
    plt.legend(handlelength=6)
    plt.title(
        "State Transition Comparing Exact vs. Viterbi MAP Inference | Short HMM")
    plt.ylabel("State")
    plt.xlabel("Xk")
    plt.show()

    fbhmm = ForwardBackwardHMM(pxk_xkm1, pyk_xk, px0, y_obs_long)
    probs_eln, logalphas, logbetas = fbhmm.forward_backward_eln()
    exact_state_trace = fbhmm.max_likelihood_state_estimate(probs_eln)
    vhmm = ViterbiHMM(pxk_xkm1, pyk_xk, px0, y_obs_long)
    path, log_probs = vhmm.viterbi_path()

    K = [i for i in range(0, len(y_obs_long) + 1)]

    plt.plot(K, exact_state_trace, marker='+', markersize='12', linestyle='-.', color='b', linewidth=2, drawstyle='steps-mid', label='Exact Inference')

    plt.plot(K, path, marker='x', markersize='12', linestyle=':', color='r', linewidth=2, drawstyle='steps-mid', label='Viterbi')
    yint = range(1, 5)
    plt.yticks(yint)
    plt.legend(handlelength=6)
    plt.title(
        "State Transition Comparing Exact vs. Viterbi MAP Inference | Long HMM")
    plt.ylabel("State")
    plt.xlabel("Xk")
    plt.show()


if __name__ == "__main__":
    state_trace()

