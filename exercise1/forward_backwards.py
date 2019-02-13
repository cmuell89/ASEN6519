import numpy as np
import os


def transtion_p(xk, x_km1):
    pxk_xkm1 = [[0.020, 0.019, 0.000, 0.666],
                [0.000, 0.025, 0.517, 0.000],
                [0.163, 0.769, 0.466, 0.000],
                [0.817, 0.187, 0.017, 0.334]]

    return pxk_xkm1[xk, x_km1]


def evidence_p(yk, xk):
    pyk_xk = [[0.0338, 0.0000, 0.0000, 0.3273]
              [0.0934, 0.0000, 0.0000, 0.0949]
              [0.1356, 0.0000, 0.0000, 0.0311]
              [0.1031, 0.0000, 0.0000, 0.0125]
              [0.1350, 0.0000, 0.0000, 0.0113]
              [0.0289, 0.0000, 0.0000, 0.3354]
              [0.0968, 0.0000, 0.0000, 0.1094]
              [0.1409, 0.0000, 0.0000, 0.0488]
              [0.1117, 0.0000, 0.0000, 0.0149]
              [0.1208, 0.0000, 0.0000, 0.0144]
              [0.0000, 0.0842, 0.7353, 0.0000]
              [0.0000, 0.2048, 0.1869, 0.0000]
              [0.0000, 0.3774, 0.0195, 0.0000]
              [0.0000, 0.3336, 0.0267, 0.0000]
              [0.0000, 0.0000, 0.0316, 0.0000]]

    return pyk_xk[yk, xk]


def initial_p(xk):
    px0 = [0, 0, 1, 0]
    return px0[xk]


def import_nominal_parameters(filename):
    curr_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(curr_path, filename)
    with open(path) as f:
        data = f.readlines()

    for line in data:
        print(line)

if __name__ == "__main__":
