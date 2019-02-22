import numpy as np
import math

def eexp(x):
    if math.isnan(x) or np.isnan(x):
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
    if math.isnan(elnx) or np.isnan(elnx) or math.isnan(elny) or np.isnan(elny):
        if math.isnan(elnx) or np.isnan(elnx):
            return elny
        else:
            return elnx
    else:
        if elnx > elny:
            elnx + eln(1 + np.exp(elny - elnx))
        else:
            elny + eln(1 + np.exp(elnx - elny))


def elnproduct(elnx, elny):
    if math.isnan(eln) or np.isnan(elnx) or math.isnan(elny) or np.isnan(elny):
        return float('nan')
    else:
        return elnx + elny