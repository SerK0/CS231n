import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= -np.max(scores)
    loss+=-np.log(np.exp(scores[y[i]])/np.sum(np.exp(scores)))
    for j in range(num_classes):
      if j==y[i]:
        dW[:, j] += -(1-np.exp(scores[j])/np.sum(np.exp(scores)))*X[i]
      else:
        dW[:, j] += np.exp(scores[j])*X[i]/np.sum(np.exp(scores))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss/=num_train
  loss+=reg*np.sum(W*W)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  scores -= -np.max(scores)
  loss = np.sum(-np.log(np.exp(scores[np.arange(scores.shape[0]),y])/np.sum(np.exp(scores),axis=1)))
  scores = np.exp(scores)
  probabilities = scores/np.sum(scores,axis=1)[:,None]
  probabilities[np.arange(probabilities.shape[0]),y] = -(1-probabilities[np.arange(probabilities.shape[0]),y])
  dW = (probabilities.T.dot(X)).T
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss/=num_train
  loss+=reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  dW /=num_train
  dW +=2*reg*W
  return loss, dW

