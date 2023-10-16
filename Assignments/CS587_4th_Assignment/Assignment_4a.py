#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os
import random


# # **Utility Functions**

# In[3]:


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)] 
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)] 
   
        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1**t)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * (grads["dW" + str(l+1)] ** 2)
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * (grads["db" + str(l+1)] ** 2)
     
        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2 ** t)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / np.sqrt(s_corrected["dW" + str(l+1)] + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / np.sqrt(s_corrected["db" + str(l+1)] + epsilon)
       

    return parameters, v, s


# # **RNN cell**

# In[4]:


####################################################################################
# Task 1   TODO:                                                                   #
# 1. Define the operations that update a at each RNN step                          #
# (i.e. the value of a that exists the RNN cell),                                  #
# 2 Store the value in a_next                                                      #
# 3. Define the prediction of the cell (y_pred)                                    #
# 4. Allowed functions that you can use are (a)np.tanh, (b) np.dot,                # 
#                                          (c) our softmax (see above)             #
# 5. Store compute values in a tuple with order a_next, a_prev, xt, parameters     #
####################################################################################

def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    
    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    ### START CODE HERE ### (≈3-4 lines)
    
    # 1. compute next activation state using the formula in the RNN cell figure
  
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)

    # 2. compute output of the current cell using the formula given above   
     
    yt_pred = softmax(np.dot(Wya, a_next) + by) 
    
    # 3. store values you need for backward propagation in cache

    cache = (a_next, a_prev, xt, parameters)
   
    ### END CODE HERE ###
    
    #return a_next, yt_pred, your_tuple
    return a_next, yt_pred, cache


# # **RNN model**
# 
# A simple implementation of the forward pass

# In[5]:


#####################################################################################
# Task 2   TODO:                                                                    #
# 1. Define the tensors that store the hidden states a, and the predictions y       #
# FOR EACH TIMESTEP                                                                 #
# 2  Feed the current input x to the RNN cell (rnn_cell_forward)                    #
# 3. Use the outputs of 2.2 to define the current prediction of the cell            # 
#    and the next hidden state a_next.                                              # 
#                                                                                   #
# 4. Store the  prediction and next hidden state values in the 3D tensors a, y_pred #
# 5. Append the tuple that has the values of all the parameters of the cell into    #
#    the list named caches                                                          #
#####################################################################################


def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    
    # Initialize "caches" which will contain the list of all caches
    caches = []
    
    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    ### START CODE HERE ###
    
    # 1. initialize "a" and "y" with zeros 

    a, y_pred = np.zeros((n_a, m, T_x)), np.zeros((n_y, m, T_x))

    # 2. Initialize a_next

    a_next = a0
    
    # 3. loop over all time-steps T_x
 
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters) 

        a[:,:,t], y_pred[:,:,t] = a_next, yt_pred
            
        caches.append(cache)    

    ### END CODE HERE ###
    
    # store values needed for backward propagation in cache
    caches = (caches, x)
    
    return a, y_pred, caches


# ***Check if everything works***: 
# 
# a[4][1] should have [-0.99999375  0.77911235 -0.99861469 -0.99833267]
# 
# y_pred[1][3] should have [ 0.79560373  0.86224861  0.11118257  0.81515947]

# In[6]:


np.random.seed(1)
x_tmp = np.random.randn(3,10,4)
a0_tmp = np.random.randn(5,10)

parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(5,5)
parameters_tmp['Wax'] = np.random.randn(5,3)
parameters_tmp['Wya'] = np.random.randn(2,5)
parameters_tmp['ba'] = np.random.randn(5,1)
parameters_tmp['by'] = np.random.randn(2,1)

a_tmp, y_pred_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][1] = \n", a_tmp[4][1])

print("y_pred[1][3] =\n", y_pred_tmp[1][3])


# # **BONUS part**
# 
# Implement a GRU cell, as shown in the figure in the assignment pdf!
# 
# Good luck :)

# In[7]:


def GRU_cell_forward(xt, a_prev, parameters):
    """
    Implement a single forward step of the GRU-cell as described in Figure in your assignment

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
   
    parameters -- python dictionary containing:
                        Wz -- Weight matrix of the input filter gate, numpy array of shape (n_a, n_a + n_x)
                        Wr -- Weight matrix of the forget/reset gate, numpy array of shape (n_a, n_a + n_x)
                        Wh -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, xt, parameters)
    
    """
    # Retrieve parameters from "parameters"
    Wz = parameters["Wz"] # input filter gate weight
  
    Wr = parameters["Wr"] # update reset weight (notice the variable name)
    Wh = parameters["Wh"] # update hidden weight (notice the variable name)
    
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    ###########################
    ### START CODE HERE ###
    # 1.  Concatenate a_prev and xt (≈3 lines)
  
    concat = np.concatenate((a_prev, xt), axis=0)

    # 2. Compute values for zt, rt, ht_sl, a_next using the formulas given figure  (≈6 lines)
       
    zt = sigmoid(np.dot(Wz, concat))
    rt = sigmoid(np.dot(Wr, concat))
    ht_sl = np.tanh(np.dot(Wh, np.concatenate((rt * a_prev, xt), axis=0)))
    a_next = (1 - zt) * a_prev + zt * ht_sl

    # 3. Compute prediction of the GRU cell (≈1 line)
    yt_pred = np.dot(Wy, a_next) + by

    ### END CODE HERE ###
    ############################
    
    # store values needed for backward propagation in cache
    cache = (a_next, a_prev, zt, rt, ht_sl, xt, parameters)

    return a_next, yt_pred, cache


# **Test your implementation**
# 
# You should get:
# 
# * a_next[4] = 
#  [-0.994056   -0.54427401 -0.1301786   0.88059754  0.41074392 -1.04062004
#  -0.30944792  0.77150011  0.18802858  0.97770343]
# 
# * a_next.shape =  (5, 10)
# 
# * yt[1] = [1.05402090e-01 9.32570980e-01 4.63909785e-01 9.71900263e-01
#  2.74407670e-01 6.13722307e-01 3.03744741e-01 1.57571804e-01
#  1.10127750e-04 3.63304568e-01]
# 
# * yt.shape =  (2, 10)
# 
# * cache[1][3] =
#  [-0.75439794  1.25286816  0.51292982 -0.29809284  0.48851815 -0.07557171
#   1.13162939  1.51981682  2.18557541 -1.39649634]
# 
# * len(cache) =  7

# In[8]:


np.random.seed(1)
xt_tmp = np.random.randn(3,10)
a_prev_tmp = np.random.randn(5,10)

parameters_tmp = {}
parameters_tmp['Wz'] = np.random.randn(5, 5+3)

parameters_tmp['Wr'] = np.random.randn(5, 5+3)
parameters_tmp['Wh'] = np.random.randn(5, 5+3)

parameters_tmp['Wy'] = np.random.randn(2,5)
parameters_tmp['by'] = np.random.randn(2,1)

a_next_tmp, yt_tmp, cache_tmp = GRU_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)
print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = ", a_next_tmp.shape)

print("yt[1] =", yt_tmp[1])
print("yt.shape = ", yt_tmp.shape)
print("cache[1][3] =\n", cache_tmp[1][3])
print("len(cache) = ", len(cache_tmp))


# In[ ]:




