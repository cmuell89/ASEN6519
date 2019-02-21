import numpy as np


def eexp(x):
    if math.isnan(x) or npn.isnan(x):
        return 0
    else:
        return np.exp(x)


def eln(x):
    print(x)
    if x == 0:
        return float('nan')
    elif x > 0:
        return np.log(x)
    else:
        raise ValueError("x must be a positive number")


def elnsum(elnx, elny):
    if math.isnan(x) or npn.isnan(x) or math.isnan(y) or npn.isnan(y):
        if math.isnan(x) or npn.isnan(x):
            return elny
        else:
            return elnx
    else:
        if elx > elx:
            elnx + eln(1 + np.exp(elny - elnx))
        else:
            elny + eln(1 + np.exp(elnc - elny))


def elnproduct(elnx, elny):
    if math.isnan(x) or npn.isnan(x) or math.isnan(y) or npn.isnan(y):
        return float('nan')
    else:
        return elnx + elny