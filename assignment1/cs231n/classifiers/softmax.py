from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    e_scores = np.e**scores
    idx = [i for i in range(X.shape[0])]
    correct_e_scores = e_scores[idx,y]
    dri = np.zeros((X.shape[0],W.shape[0],W.shape[1]))
    for i in range(X.shape[0]):
        loss += -np.log(correct_e_scores[i]/np.sum(e_scores[i]))
        dri[i] = (X[i]*(e_scores[i].reshape((W.shape[1],1)))).T/np.sum(e_scores[i])
        dri[i,:,y[i]] += -X[i]
        
    dW = np.sum(dri,axis=0)/X.shape[0]
    loss /= X.shape[0]
    loss += reg*np.sum(W*W)
    dW += 2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    e_scores = np.e**scores
    idx = [i for i in range(X.shape[0])]
    correct_e_scores = e_scores[idx,y]
    loss += -np.sum(scores[idx,y])
    loss += np.sum(np.log(np.sum(e_scores,axis=1)))
    loss /= X.shape[0]
    loss += reg*np.sum(W*W)
    num_mult = np.zeros_like(scores)
    num_mult = e_scores/(np.sum(e_scores,axis=1).reshape((-1,1)))
    num_mult[idx,y] += -1
    dW = np.sum(X*(num_mult.T.reshape((W.shape[1],X.shape[0],1))),axis=1).T
    dW /= X.shape[0]
    dW += 2*reg*W
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
