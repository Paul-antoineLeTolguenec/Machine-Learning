import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    """computes the cost and gradient of the neural network. The
    parameters for the neural network are "unrolled" into the vector
    nn_params and need to be converted back into the weight matrices.

    The returned parameter grad should be a "unrolled" vector of the
    partial derivatives of the neural network.
    """

    # Get theta1 and theta2 back from nn_params
    theta1 = nn_params[0:(hidden_layer_size*(input_layer_size+1))].reshape((input_layer_size+1),hidden_layer_size).T
    theta2 = nn_params[(hidden_layer_size*(input_layer_size+1)):].reshape((hidden_layer_size+1),num_labels).T

    # Get the shape of X
    m, _ = X.shape

    # You need to return the following variables correctly 
    J = 0
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    # One hot encoding of the waited output
    y_matrix = (np.arange(0, y.max()+1) == y.flatten()[:,None]).astype(int).T
    
    # Compute Cost
    ## Input Layer
    a1 = np.hstack((np.ones((m, 1)), X)).T
    ## Hidden Layer
    z2 = theta1@a1
    a2 = np.vstack((np.ones((1, m)), sigmoid(z2)))
    ## Output Layer
    z3 = theta2@a2
    a3 = sigmoid(z3)
    
    # Ragularized cost computing
    inner = - y_matrix.T * np.log(a3).T - (np.ones(y_matrix.shape) - y_matrix).T * np.log(np.ones((a3).shape) - a3).T
    J = np.sum(inner) / m + (Lambda/(2*m))*(np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:])))
    
    # Gradients
    d3 = a3 - y_matrix
    d2 =  (theta2[:,1:].T@d3) * sigmoidGradient(z2)
    
    delta1 = d2.dot(a1.T)
    delta2 = d3@(a2.T)
    
    # Gradient regularisation
    theta1_grad = delta1/m  
    reg = (theta1_grad[:,1:]*Lambda)/m
    theta1_grad[:,1:] = theta1_grad[:,1:] + reg 
    
    theta2_grad = delta2/m  
    reg = (theta2[:,1:]*Lambda)/m
    theta2_grad[:,1:] = theta2_grad[:,1:] + reg

    # Unroll gradient
    grad = np.hstack((theta1_grad.T.ravel(), theta2_grad.T.ravel()))

    return J, grad