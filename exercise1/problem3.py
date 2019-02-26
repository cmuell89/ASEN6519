from forward_backwards import ForwardBackwardHMM
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


def short_state_trace():
    fbhmm = ForwardBackwardHMM(pxk_xkm1, pyk_xk, px0, y_obs_long)
    probs, alphas, betas = fbhmm.forward_backward()
    probs_eln, logalphas, logbetas = fbhmm.forward_backward_eln()
    state_trace = fbhmm.max_likelihood_state_estimate(probs)
    eln_state_trace = fbhmm.max_likelihood_state_estimate(probs_eln)
    K = [i for i in range(0, len(y_obs_long) + 1)]

    plt.plot(K, state_trace, marker='+', markersize='12', linestyle='-.', color='b', linewidth=2, drawstyle='steps-mid', label='Regular FB')

    plt.plot(K, eln_state_trace, marker='x', markersize='12', linestyle=':', color='r', linewidth=2, drawstyle='steps-mid', label='ELN FB')
    yint = range(1, 5)
    plt.yticks(yint)
    plt.legend(handlelength=4, loc='upper right')
    plt.title("State transition chart for Long Dataset")
    plt.ylabel("State")
    plt.xlabel("Xk")
    plt.show()


def posterior_chart():
    fbhmm = ForwardBackwardHMM(pxk_xkm1, pyk_xk, px0, y_obs_long)
    probs, alphas, betas = fbhmm.forward_backward()
    probs = np.concatenate((probs.transpose()[:5], probs.transpose()[-5:]), axis=0).transpose()
    print(tabulate(probs, headers=["X1", "X2", "X3", "X4", "X5", "X42", "X43", "X44", "X45", "X46"], tablefmt="latex"))


def data_likelihood():
    fbhmm = ForwardBackwardHMM(pxk_xkm1, pyk_xk, px0, y_obs_long)
    probs, alphas, betas = fbhmm.forward_backward()
    print(eln(sum(alphas[:, len(y_obs_long)])))


if __name__ == "__main__":
    # short_state_trace()
    posterior_chart()
    data_likelihood()