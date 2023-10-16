import numpy as np
from classUtils.classifiers.compute_gradient_loss import *

class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train_linear_classifier(self, X_train, y_train, X_val, y_val, learning_rate=1e-3, reg=1e-5, reg_type = 2, num_epochs = 6, batch_size = 200, num_valid_loss_evals = 100, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X_train: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y_train: A numpy array of shape (N,) containing training labels; y_train[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - X_val: A numpy array of shape (M, D) containing training data; there are M
      validation samples each of dimension D.
    - y_val: A numpy array of shape (M,) containing validation labels; y[_vali] = c
      means that X_val[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
	- reg_type: (integer) regularization type (1: l1, 2: l2)
	- num_epochs: number of epochs to go through evaluating the training samples
	- num_valid_loss_evals: number of iterations after which the validation loss will be calculated
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
	
    num_train, dim = X_train.shape
	  # num_iters is the number of steps to take when optimizing defined by the following formula
    num_iters = (num_train*num_epochs/batch_size)
    num_classes = np.max(y_train) + 1 # assume y takes values 0...K-1 where K is number of classes
	
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)

	  #array to save the computed training loss, either create an array of zeros or an empty list 
    train_loss_history =  [] #DONE 
	  #array to save the computer validation loss
    valid_loss_history = [] #DONE
    
    # Run stochastic gradient descent to optimize W
    
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None
	  
      train_loss = 0
	  
      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #																		                                    #
	    # IMPORTANT NOTE !!!													                          #
      # In your implementation you need to call the loss function of 		      #
	    # the class (self.loss(...)),										                        #
	    # Do not use the compute_gradient_and_loss directly 					          #
	    #																		                                    #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################

      ind = np.random.choice(num_train, batch_size, replace=True) #sampling with replacement
      X_batch = X_train[ind]
      y_batch = y_train[ind]
      train_loss, grad = self.loss(X_batch, y_batch, reg, reg_type, 0)

      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################		
      # evaluate loss and gradient
           
      loss, grad = self.loss(X_batch, y_batch, reg, reg_type, 0)

      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights self.W using the gradient and the learning rate.   #
      #########################################################################

      # GD learning rule is w += -(learning_rate * gradient), where gradient is dW
      self.W += -(learning_rate * grad)
      
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

	    #########################################################################
      # TODO:                                                                 #
      # Print out the computed loss values for training and validation samples#
	    # i.e the training loss can be printed as provided below   		          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration No %d / %d: loss %f' % (it, num_iters, train_loss)
        #append to array
        train_loss_history.append(train_loss)

      #validation loss
      if verbose and it % num_valid_loss_evals == 0:
        valid_loss = self.loss(X_val, y_val, reg, reg_type, 0)[0]
        print 'Iteration No: %d / %d: Validation Loss:  %f' % (it, num_iters, valid_loss)
        valid_loss_history.append(valid_loss)

      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

    return train_loss_history, valid_loss_history

  def predict_image_class(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: D x N array of training data. Each column is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[1])

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################

    #axis=1 means rows are summed and find the max of each row (each image class)
    y_pred = np.argmax(X.dot(self.W), axis=1)

    pass
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred
  
  def loss(self, X_batch, y_batch, reg, reg_type, opt):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.
	- reg_type: (int) regularization type (1: l1, 2: l2)
	- opt: (int) 0 for computing both loss and gradient, 1 for computing loss only

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
	#Implement the compute_gradient_and_loss in compute_gradient_loss.py
    return compute_gradient_and_loss(self.W, X_batch, y_batch, reg, reg_type, opt)
	
