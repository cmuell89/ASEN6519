import numpy as np
from observations import y_obs_long, y_obs_short
from parameters import pxk_xkm1, pyk_xk, px0
from exponential_scaling import eexp, eln, elnsum, elnproduct
import colored_traceback
colored_traceback.add_hook()


class ViterbiHMM():

    def __init__(self, transition_probs, evidence_probs, initial_probs, observations):
        self.n_states = len(initial_probs)
        self.init = initial_probs
        self.emis = evidence_probs
        self.trans = transition_probs
        self.n_obs = len(observations)
        self.obs = observations

    def likeliest_path(self):

        init = np.array([eln(val) for val in np.nditer(self.init)])
        trans = np.array([[eln(v) for v in np.nditer(axis)] for axis in np.nditer(self.trans)]).reshape(self.n_states, self.n_states)
        emis = np.array([[eln(v) for v in np.nditer(axis)] for axis in np.nditer(self.emis, order='F')]).reshape(self.n_states, self.n_obs)

        best_states = []

        logprob = init + emis[:, self.obs[0]]
        for k in range(1, self.n_obs):
            trans_p = np.zeros([self.n_states, self.n_states])
            for i in range(self.n_states):
                for j in range(self.n_states):
                    trans_p[i, j] = elnsum(logprob[i], elnproduct(trans[i, j], emis[i, self.obs[k]]))
            # Get the indices of the max probs, which give the best prior states.
            best_states.append(np.argmax(trans_p, axis=0))
            logprob = np.max(trans_p, axis=0)

        # Most likely final state.
        final_state = np.argmax(logprob)
        print(final_state)
        # Reconstruct path by following links and then reversing.
        prior_state = final_state
        best_path = [prior_state + 1]
        for best in reversed(best_states):
            prior_state = best[prior_state]
            best_path.append(prior_state + 1)
        return list(reversed(best_path)), logprob[final_state]

if __name__ == "__main__":
    vhmm = ViterbiHMM(pxk_xkm1, pyk_xk, px0, y_obs_short)
    path, log_probs = vhmm.likeliest_path()
    print(path)