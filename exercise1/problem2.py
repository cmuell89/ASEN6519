from forward_backwards import ForwardBackwardHMM
from likeilhood_sampling_inference import LikelihoodSamplingInference
from observations import y_obs_long, y_obs_short
from parameters import pxk_xkm1, pyk_xk, px0
from exponential_scaling import eexp, eln, elnsum, elnproduct
import matplotlib.pyplot as plt
from tabulate import tabulate

if plt.get_backend() == 'Qt5Agg':
    from matplotlib.backends.qt_compat import QtWidgets
    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()
plt.rcParams.update({'font.size': 22})


def state_trace():
    lli = LikelihoodSamplingInference(pxk_xkm1, pyk_xk, px0, y_obs_short)

    print("Running approximate inference with 100 samples...")
    state_trace_100 = lli.max_likelihood_state_estimate(lli.run_inference(100))

    print("Running approximate inference with 1000 samples...")
    state_trace_1000 = lli.max_likelihood_state_estimate(
        lli.run_inference(1000))

    print("Running approximate inference with 10000 samples...")
    state_trace_10000 = lli.max_likelihood_state_estimate(
        lli.run_inference(10000))

    print("Running exact inference...")
    fbhmm = ForwardBackwardHMM(pxk_xkm1, pyk_xk, px0, y_obs_short)
    probs_eln, logalphas, logbetas = fbhmm.forward_backward_eln()
    exact_state_trace = fbhmm.max_likelihood_state_estimate(probs_eln)

    K = [i for i in range(0, len(y_obs_short) + 1)]

    print("Plotting 'State Transition Comparing Exact vs. 100 Sample Approximate Inference'...")
    plt.plot(K, exact_state_trace, marker='+', markersize='12', linestyle='-.', color='b', linewidth=2, drawstyle='steps-mid', label='Exact Inference')

    plt.plot(K, state_trace_100, marker='x', markersize='12', linestyle=':', color='r', linewidth=2, drawstyle='steps-mid', label='LW - 100 Samples')
    yint = range(1, 5)
    plt.yticks(yint)
    plt.legend(handlelength=6)
    plt.title(
        "State Transition Comparing Exact vs. 100 Sample Approximate Inference")
    plt.ylabel("State")
    plt.xlabel("Xk")
    plt.show()

    print("Plotting 'State Transition Comparing Exact vs. 1000 Sample Approximate Inference'...")
    plt.plot(K, exact_state_trace, marker='+', markersize='12', linestyle='-.', color='b', linewidth=2, drawstyle='steps-mid', label='Exact Inference')

    plt.plot(K, state_trace_1000, marker='x', markersize='12', linestyle=':', color='r', linewidth=2, drawstyle='steps-mid', label='LW - 1000 Samples')
    yint = range(1, 5)
    plt.yticks(yint)
    plt.legend(handlelength=6)
    plt.title(
        "State Transition Comparing Exact vs. 1000 Sample Approximate Inference")
    plt.ylabel("State")
    plt.xlabel("Xk")
    plt.show()

    print("Plotting 'State Transition Comparing Exact vs. 10000 Sample Approximate Inference'...")
    plt.plot(K, exact_state_trace, marker='+', markersize='12', linestyle='-.', color='b', linewidth=2, drawstyle='steps-mid', label='Exact Inference')

    plt.plot(K, state_trace_10000, marker='x', markersize='12', linestyle=':', color='r', linewidth=2, drawstyle='steps-mid', label='LW - 10000 Samples')
    yint = range(1, 5)
    plt.yticks(yint)
    plt.legend(handlelength=6)
    plt.title(
        "State Transition Comparing Exact vs. 10000 Sample Approximate Inference")
    plt.ylabel("State")
    plt.xlabel("Xk")
    plt.show()


if __name__ == "__main__":
    state_trace()
