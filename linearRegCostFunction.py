import numpy as np

def h(theta, X):
    return X@theta

def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
    # Initialize some useful values

    m,n = X.shape # number of training examples
    theta = theta.reshape((n,1)) # in case where theta is a vector (n,) 

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost and gradient of regularized linear 
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    #
    # =========================================================================

    J = 1/(2*m) * (h(theta, X) - y).T@(h(theta, X)- y) + Lambda/(2*m) * theta.T@theta
    

    theta_grad = np.copy(theta)
    theta_grad[0] = 0.

    grad = 1/m * (X.T@(h(theta, X) - y) + Lambda/2*theta_grad)
    
    return J.flatten(), grad.flatten()