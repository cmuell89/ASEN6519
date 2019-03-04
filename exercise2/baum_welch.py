import numpy as np
from observations import y_obs_long, y_obs_short
from parameters import pxk_xkm1, pyk_xk, px0
from exponential_scaling import eexp, eln, elnsum, elnproduct
import matplotlib.pyplot as plt
from tabulate import tabulate
import colored_traceback
colored_traceback.add_hook()


class BaumWelchHMM():

    def __init__(self, init_trans, init_emis, init, obs):
        self.n_states = len(init)
        self.init = init
        self.emis = init_emis
        self.trans = init_trans
        self.n_obs = len(obs)
        self.obs = obs

    
if __name__ == "__main__":
    # state_trace()
