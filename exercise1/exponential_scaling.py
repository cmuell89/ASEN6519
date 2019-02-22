import numpy as np


def eexp(x):
    if np.isinf(x):
        return 0
    else:
        return np.exp(x)


def eln(x):
    if x == 0:
        return -np.inf
    if x > 0:
        return np.log(x)
    else:
        raise("x must be non-negative for eln")


def elnsum(elnx, elny):
    if np.isinf(elnx) or (np.isinf(elny)):
        if (np.isinf(elnx)):
            return elny
        else:
            return elnx
    else:
        if (elnx > elny):
            return elnx + eln(1 + np.exp(elny - elnx))
        else:
            return elny + eln(1 + np.exp(elnx - elny))


def elnproduct(elnx, elny):
    if np.isinf(elnx) or np.isinf(elny):
        return -np.inf
    else:
        return elnx + elny