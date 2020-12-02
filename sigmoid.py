import numpy as np

def sigmoid(z):
    """computes the sigmoid of z."""

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar).
#    g = 1 / (1 + np.exp(-z))
# =============================================================
    return  1 / (1 + np.exp(-z))
    
#print sigmoid(0)
#print sigmoid(10)
#print sigmoid(-11)
