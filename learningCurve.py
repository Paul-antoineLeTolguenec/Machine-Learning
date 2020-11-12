import numpy as np

from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction


def h(theta, X):
    return X@theta

def cost(X, y, theta):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
    # Initialize some useful values

    m,n = X.shape # number of training examples
    theta = theta.reshape((n,1)) # in case where theta is a vector (n,) 

    J = 1/(2*m) * (h(theta, X) - y).T@(h(theta, X)- y) 
    return J

def learningCurve(X, y, Xval, yval, Lambda):
    """returns the train and
    cross validation set errors for a learning curve. In particular,
    it returns two vectors of the same length - error_train and
    error_val. Then, error_train(i) contains the training error for
    i examples (and similarly for error_val(i)).

    In this function, you will compute the train and test errors for
    dataset sizes from 1 up to m. In practice, when working with larger
    datasets, you might want to do this in larger intervals.
    """

# Number of training examples
    m, _ = X.shape

# You need to return these values correctly
    error_train = np.zeros(m)
    error_val   = np.zeros(m)

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return training errors in 
#               error_train and the cross validation errors in error_val. 
#               i.e., error_train(i) and 
#               error_val(i) should give you the errors
#               obtained after training on i examples.
#
# Note: You should evaluate the training error on the first i training
#       examples (i.e., X[:i+1,:] and y[:i+1]).
#
#       For the cross-validation error, you should instead evaluate on
#       the _entire_ cross validation set (Xval and yval).
#
# Note: If you are using your cost function (linearRegCostFunction)
#       to compute the training and cross validation error, you should 
#       call the function with the lambda argument set to 0. 
#       Do note that you will still need to use lambda when running
#       the training to obtain the theta parameters.
#
# Hint: You can loop over the examples with the following:
#
#       for i in np.arange(m):
#           # Compute train/cross validation errors using training examples 
#           # X[:i+1,:] and y[:i+1], storing the result in 
#           # error_train and error_val
#           ....
#



# -------------------------------------------------------------------------

# =========================================================================
    for i in range(m):
        theta_trained  = trainLinearReg(X[:i+1, :], y[:i+1], Lambda)
        error_val [i] = cost(Xval, yval, theta_trained)
        error_train[i] = cost(X[:i+1, :], y[:i+1], theta_trained)
    return error_train, error_val