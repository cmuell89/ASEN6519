import numpy as np

y_obs_long = np.array([11,
                       11,
                       13,
                       6,
                       10,
                       3,
                       1,
                       7,
                       1,
                       6,
                       6,
                       2,
                       7,
                       8,
                       7,
                       6,
                       3,
                       12,
                       14,
                       11,
                       14,
                       11,
                       14,
                       12,
                       14,
                       11,
                       13,
                       1,
                       4,
                       1,
                       3,
                       1,
                       7,
                       3,
                       6,
                       5,
                       12,
                       14,
                       11,
                       11,
                       11,
                       14,
                       11,
                       12,
                       14])

y_obs_short = np.array([14,
                        11,
                        11,
                        12,
                        14,
                        11,
                        12,
                        2,
                        6,
                        5,
                        6,
                        8,
                        1,
                        5,
                        2])

y_obs_short = y_obs_short - 1
y_obs_long = y_obs_long - 1
