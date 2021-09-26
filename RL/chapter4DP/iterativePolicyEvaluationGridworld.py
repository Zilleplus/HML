import numpy as np


# All transitions have -1 result
# The 2 final states(0, 0) and (3, 3) don't transition.
# And moves off the grid have no effect, leaving
# the state unchanged. But still taking the -1 result.

# Default policy takes random move.
# -> pi(a | s) = 0.5

vals = np.zeros((4, 4))


def InRange(row, col):
    (num_rows, num_cols) = vals.shape
    return row < num_rows and col < num_cols and row > -1 and col > -1


def GenerateLocs(row, col):
    for i in [-1, 1]:
        yield (row+i, col)
    for j in [-1, 1]:
        yield (row, col+j)


for i in range(0, 10):
    old_vals = np.copy(vals)
    vals
    for row in range(0, 4):
        for col in range(0, 4):
            if(col == 0 and row == 0):
                continue  # final
            if(col == 3 and row == 3):
                continue  # final

            locs = GenerateLocs(row, col)
            s = 0
            for (l_r, l_c) in locs:
                # We take gamma=1
                # basically a one step look ahead
                # take reward(-1) + previous value of new state
                # change old_vals to vals to do the more efficient variation
                if InRange(l_r, l_c):
                    s = s + 0.25*(-1+old_vals[l_r][l_c])
                else:
                    # If you go outside, then stay so use current row/col
                    s = s + 0.25*(-1+old_vals[row][col])

            vals[row][col] = s

vals
