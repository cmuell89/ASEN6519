import numpy as np
# the state transition probability going
# from state x_k-1 to x_k i.e. p(x_k | x_k-1)
pxk_xkm1 = np.array([np.array([0.020, 0.019, 0.000, 0.666]),
                     np.array([0.000, 0.025, 0.517, 0.000]),
                     np.array([0.163, 0.769, 0.466, 0.000]),
                     np.array([0.817, 0.187, 0.017, 0.334])])


# returns the probability of evidence yk given
# xk i.e. p(yk | xk)
pyk_xk = np.array([np.array([0.0338, 0.0000, 0.0000, 0.3273]),
                   np.array([0.0934, 0.0000, 0.0000, 0.0949]),
                   np.array([0.1356, 0.0000, 0.0000, 0.0311]),
                   np.array([0.1031, 0.0000, 0.0000, 0.0125]),
                   np.array([0.1350, 0.0000, 0.0000, 0.0113]),
                   np.array([0.0289, 0.0000, 0.0000, 0.3354]),
                   np.array([0.0968, 0.0000, 0.0000, 0.1094]),
                   np.array([0.1409, 0.0000, 0.0000, 0.0488]),
                   np.array([0.1117, 0.0000, 0.0000, 0.0149]),
                   np.array([0.1208, 0.0000, 0.0000, 0.0144]),
                   np.array([0.0000, 0.0842, 0.7353, 0.0000]),
                   np.array([0.0000, 0.2048, 0.1869, 0.0000]),
                   np.array([0.0000, 0.3774, 0.0195, 0.0000]),
                   np.array([0.0000, 0.3336, 0.0267, 0.0000]),
                   np.array([0.0000, 0.0000, 0.0316, 0.0000])])

# initial states
px0 = np.array([0, 0, 1, 0])
