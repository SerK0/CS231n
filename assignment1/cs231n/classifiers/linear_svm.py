import numpy as np
from random import shuffle

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
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    cc = 0
    for j in range(num_classes):
      if j == y[i]:
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] +=X[i]
        cc+=1
    dW[:,y[i]]-= cc*X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W



  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = X.dot(W)
  num_train = X.shape[0]
  res = scores - scores[np.arange(scores.shape[0]), y].reshape(-1,1) + 1
  res[np.arange(scores.shape[0]), y] = 0
  res = np.maximum(res, 0)
  loss += np.sum(res)

  res[res > 0] = 1
  res[np.arange(scores.shape[0]), y] = -np.sum(res, axis=1)

  dW = X.T.dot(res)


  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W

  return loss,dW
