from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    dri_w = np.zeros((num_train,W.shape[0],W.shape[1]))
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                dri_w[i,:,j] = -X[i]*(sum((scores-correct_class_score+1)>0)-1)
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                dri_w[i,:,j] = X[i] if (scores[j]-correct_class_score+1)>0 else 0
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    dW = np.sum(dri_w,axis=0)/num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2*reg*W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #代码全在上面，混在一起了

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    idx = [i for i in range(y.shape[0])]
    corret_class_score = scores[idx,y]
    margin = scores - np.array(corret_class_score).reshape(y.shape[0],-1) + 1
    loss += (np.sum(margin[margin>0]) - margin.shape[0]) #减掉真score的margin
    loss /= X.shape[0]
    loss += reg * np.sum(W * W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_mult = np.zeros(scores.shape)
    num_mult[margin>0] = 1
    num_mult[idx,y] = -(np.sum(margin>0,axis=1)-1)
    #for i in range(W.shape[1]):
    #   dW[:,i] = np.sum(X*num_mult[:,i].reshape(X.shape[0],1),axis=0) #和下面速度差不多
    num_mult = num_mult.T.reshape((W.shape[1],-1,1))
    dW = np.sum(X*num_mult,axis=1).T
    dW /= X.shape[0]
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
