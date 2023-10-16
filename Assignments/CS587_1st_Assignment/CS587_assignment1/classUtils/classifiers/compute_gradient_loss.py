import numpy as np
from random import shuffle

def compute_gradient_and_loss(W, X, y, reg, reg_type, opt):
  """
  loss and gradient function.

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  - reg_type: (int) regularization type (1: l1, 2: l2)
  - opt: (int) 0 for computing both loss and gradient, 1 for computing loss only
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  
  #############################################################################
  # TODO:                                                                     #
  # Implement the routine to compute the loss, storing the result in loss     #
  #############################################################################

  for i in xrange(num_train):
    scores = X[i].dot(W) # predicted scores for training sample i
    ground_truth_scores = scores[y[i]] # predicted scores for the ground truth classes of each training sample
    for j in xrange(num_classes):
      if j == y[i]:
        continue #hinge is zero for the true class
      hinge = np.maximum(0, np.max(scores[j]) - ground_truth_scores +1) #Crammer-Singer hinge loss
      #hinge = np.maximum(0, scores[j] - ground_truth_scores +1) #equivalent Weston-Watkins hinge loss
      if hinge > 0:
        loss += hinge

  regularization = 0 

  if reg_type == 1:
    regularization = np.sum(abs(np.transpose(W))) #l1 norm
  else:
    regularization = np.sum(abs(np.transpose(W))**2) #l2 norm

  #final loss
  loss += reg * regularization
 
  loss /= num_train
  
  if opt == 1 :
    return loss
  
  #############################################################################
  # TODO:                                                                     #
  # Implement the gradient for the required loss, storing the result in dW.   #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  scores = X.dot(W) # predicted scores for every training sample 
  
  d_scores = np.zeros_like(scores) # init grad to zero (partials of each class)
  
  for i in range(num_train):
    #scores for this sample
    scores_i = scores[i] 
    ground_truth_score = scores_i[y[i]]
    
    for j in range(W.shape[1]):  #for each class
        if j == y[i]: #j = y_i (true class)
            #d_scores[i, j] = -np.sum(np.sign(scores_i - ground_truth_score) +1) #for the equivalent Weston-Watkins hinge loss
            d_scores[i, j] = -np.sign(scores_i[j] +1) #for Crammer-Singer hinge loss
        else:
            #d_scores[i, j] = np.sign(scores_i[j] - ground_truth_score +1) #for the equivalent Weston-Watkins hinge loss
            d_scores[i, j] = np.sign(scores_i[j] +1) #for Crammer-Singer hinge loss
  
  dW = np.dot(X.T,d_scores)
  dW -= ground_truth_score

  #for the Weston-Watkins hinge loss, comment the two lines above and uncomment the line below
  #dW = np.dot(X.T,d_scores)

  dW /= num_train #compute final gradient

  if reg_type == 1: #l1 norm
    dW += reg * np.sign(W) #
  elif reg_type == 2: #l2 norm
    dW += 2 * reg * W
  
  pass 

  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW