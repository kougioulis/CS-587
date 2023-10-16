#!/usr/bin/env python
# coding: utf-8

# # Build, Train & Test a Multilayer Neural Networks using TensorFlow
# 
# 
# ### Goals: 
# - Intro: build and train a feed forward neural network using the `TensorFlow` framework.
# - The SGD method will be used for training to apply automatic differentiation based on TensorFlow.
# - Tune the hyperparameters and modify the structure of your NN to achieve the highest accuracy.
# - Use Tensorboard to visualize the graph and results.
# 
# ### Dataset:
# - Digits: 10 class handwritten digits
# - It will automatically be downloaded once you run the provided code using the scikit-learn library.
# - Check for info in the following websites:
# - http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html
# - http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
# - http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits

# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
# display figures in the notebook
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()


# In[25]:


sample_index = 45
plt.figure(figsize=(3, 3))
plt.imshow(digits.images[sample_index], cmap=plt.cm.gray_r,
           interpolation='nearest')
plt.title("image label: %d" % digits.target[sample_index]);


# ### Preprocessing
# 
# - normalization of your input data
# - train/test split

# In[26]:


from sklearn import preprocessing
import numpy

#numpy.set_printoptions(threshold=numpy.nan)

####################
import sys
numpy.set_printoptions(threshold=sys.maxsize)

####################

data = np.asarray(digits.data, dtype='float32')
target = np.asarray(digits.target, dtype='int32')

X_train = data[0:1500,:]
y_train = target[0:1500]

X_test = data[1500:,:]
y_test = target[1500:]

# mean = 0 ; standard deviation = 1.0
scaler = preprocessing.StandardScaler()

# print(scaler.mean_)
# print(scaler.scale_)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Check that the train and test targets/labels are balanced within each set
plt.hist(y_train)  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram of train labels/targets")
plt.show()

plt.hist(y_test)  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram of test labels/targets")
plt.show()


# Let's display the one of the transformed sample (after feature standardization):

# In[27]:


sample_index = 150
plt.figure(figsize=(3, 3))
plt.imshow(X_train[sample_index].reshape(8, 8),
           cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("transformed sample\n(standardization)");


# The scaler objects makes it possible to recover the original sample:

# In[28]:


plt.figure(figsize=(3, 3))
plt.imshow(scaler.inverse_transform(X_train[sample_index]).reshape(8, 8),
           cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("original sample");


# In[29]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# ### TensorFlow is a symbolic graph computation engine, that allows automatic differentiation of each node
# - https://www.tensorflow.org 
# - https://www.tensorflow.org/tutorials/mnist/tf/
# 
# TensorFlow builds where nodes may be:
# - **constant:** constants tensors, such as a learning rate
# - **Variables:** any tensor, such as parameters of the models
# - **Placeholders:** placeholders for inputs and outputs of your models
# - many other types of nodes (functions, loss, ...)
# 
# The graph is symbolic, no computation is performed until a `Session` is defined and the command `run` or `eval` is invoked. TensorFlow may run this computation on (multiple) CPUs or GPUs

# In[30]:


import tensorflow as tf

print(tf. __version__) #1.15

#from tensorflow.python.client import device_lib

#print(device_lib.list_local_devices())

a = tf.constant(3)
b = tf.constant(2)
c = tf.Variable(0)
c = a + b
with tf.Session() as sess:
    print(sess.run(c))


# In[31]:


X = tf.placeholder("float32", name="input")
Y = X + tf.constant(3.0)
with tf.Session() as sess:
    print(sess.run(Y, feed_dict={X:2}))


# **Note: batches in inputs**
# - the first dimension of the input is usually kept for the batch dimension. A typical way to define an input placeholder with a 1D tensor of 128 dimensions, is:
# ```
# X = tf.placeholder("float32", shape=[None, 128])
# ```

# ## 1) Build a model using TensorFlow
# 
# - Using TensorFlow, build a simple model (one hidden layer)

# In[32]:


# helper functions

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def accuracy(y_pred, y=y_test):
    return np.mean(np.argmax(y_pred, axis=1) == y)  #######axis=1


# In[10]:


# hyperparams
batch_size = 32
hid_size = 1
learning_rate = 0.01
num_epochs = 10
input_size = X_train.shape[1]
output_size = 10

# input and output
X = tf.placeholder("float32", shape=[None, input_size])
y = tf.placeholder("int32", shape=[None])

# build the model and weights
W_h = init_weights([input_size, hid_size])
b_h = init_weights([hid_size])
W_o = init_weights([hid_size, output_size])
b_o = init_weights([output_size])
h = tf.nn.relu(tf.matmul(X,W_h)+b_h) #tf.nn.tanh, tf.nn.relu, tf.nn.signmoid
out_act = tf.matmul(h, W_o)+b_o

# build the loss, predict, and train operator
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_act, labels=y)
loss = tf.reduce_sum(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

predict = tf.nn.softmax(out_act)

#writer = tf.summary.FileWriter('events_log', sess.graph)

loss_summ = tf.summary.scalar("cross_entropy_loss", loss)
W_h_summ = tf.summary.histogram("weights", W_h)
b_h_summ = tf.summary.histogram("biases", b_h)

#merge all summaries into a single "operation" which we can execute in a session
merged_summary_op = tf.summary.merge_all()

# Initialization of all variables in the graph
init = tf.global_variables_initializer()


# ### 2) Train your model using SGD algorithm and check the generalization on the test set of your dataset.

# In[12]:


#Init your session, run training
#Render your graph and monitor your training procedure using TensorBoard

#%load_ext tensorboard
#%reload_ext tensorboard

# run training
with tf.Session() as sess: 
    sess.run(init)
    
    # For monitoring purposes
    writer = tf.summary.FileWriter('./events_log', sess.graph)
    #fw = tf.summary.FileWriter('./events_log')
    
    losses = []
    
    for e in range(num_epochs):
        for i in range(X_train.shape[0] // batch_size):
            idx, idxn = i * batch_size, min(X_train.shape[0]-1, (i+1) * batch_size)
            batch_xs, batch_ys = X_train[idx: idxn], y_train[idx: idxn]            
           
            _,l=sess.run([train_op, loss], feed_dict={X: batch_xs, y: batch_ys})
            
             #create a summary object and set its value
            summary = tf.Summary(value=[tf.Summary.Value(tag="cross_entropy_loss", simple_value=l)])
            writer.add_summary(summary, e * X_train.shape[0] // batch_size + i)
            
            losses.append(l)
            
        predicts_test = sess.run(predict, feed_dict={X: X_test})
        predicts_train = sess.run(predict, feed_dict={X: X_train})
        
        summary2 = tf.Summary(value=[tf.Summary.Value(tag="train_accuracy", simple_value=accuracy(predicts_train, y_train))])
        writer.add_summary(summary2,e)

        summary3 = tf.Summary(value=[tf.Summary.Value(tag="test_accuracy", simple_value=accuracy(predicts_test))])
        writer.add_summary(summary3,e)
        
        print("epoch: %d train accuracy: %0.3f test accuracy: %0.3f" % (e, accuracy(predicts_train, y_train), accuracy(predicts_test)))
        
plt.plot(losses);

writer.flush()
writer.close()

#IN CASE YOU MISSED IT 
#TODO: Do not forget to USE TENSORBOARD for graph rendereing + monitoring your training process 

#tensorboard --logdir ./events_log


# 
# ### 3) In order to maximize the accuracy on the given dataset try different settings for your model
# 
# Play around with the structure of your NN model and fine-tune its hyperparameters.
# 
# - A. Experiment with different hyperparameters (learning rate = $0.001$,..,$0.1$, batch size = $8$,..,$128$, size of hidden layers = $5$,..,$25$, number of epochs).
# - B. Try different activation functions (e.g., ReLU, TanH).
# - C. Try to add more hidden layers and increase their size.
# - D. Add L2 regularization (e.g., with regularization strength $10^{-4}$)
# 
# ### Bonus: A + 15% will be distributed to the top-performing models based on the accuracy on the test set (e.g if there are K submissions with equal top performance, each one will get a bonus 15%/K)

# In[34]:


#TODO: MAXimize the accuracy on the given dataset try different settings for your model

import random
import itertools

batch_size_list = [32, 64, 128]
hid_size_list = [15, 25, 30, 50]
learning_rate_list = [0.01, 0.025, 0.05, 0.1]
num_epochs_list = [20, 30, 50]
regularization_strength_list = [0.01, 0.001, 0.0001]
input_size = X_train.shape[1]
output_size = 10

X = tf.placeholder("float32", shape=[None, input_size])
y = tf.placeholder("int32", shape=[None])

max_accuracy = 0
best_batch_size = 0
best_hid_size = 0
best_learning_rate = 0
best_num_epochs = 0
best_regularization_strength = 0

def random_params():
    return (
        random.choice(batch_size_list),
        random.choice(hid_size_list),
        random.choice(learning_rate_list),
        random.choice(num_epochs_list),
        random.choice(regularization_strength_list),
    )

#perform random walk on 160 random hyperparameter configurations
params_list = [random_params() for _ in range(160)]

for batch_size, hid_size, learning_rate, num_epochs, regularization_strength in params_list:
    # build the model and weights
    W_h = init_weights([input_size, hid_size])
    b_h = init_weights([hid_size])
    W_o = init_weights([hid_size, output_size])
    b_o = init_weights([output_size])
    h = tf.nn.sigmoid(tf.matmul(X,W_h)+b_h)
    out_act = tf.matmul(h, W_o)+b_o + regularization_strength * tf.nn.l2_loss(W_h) + regularization_strength * tf.nn.l2_loss(W_o)
    
    # build the loss, predict, and train operator
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_act, labels=y)
    loss = tf.reduce_sum(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    predict = tf.nn.softmax(out_act)

    # Initialization of all variables in the graph
    init = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()
    
    with tf.Session() as sess: 
        sess.run(init)
        losses = []
        for e in range(num_epochs):
            for i in range(X_train.shape[0] // batch_size):
                idx, idxn = i * batch_size, min(X_train.shape[0]-1, (i+1) * batch_size)
                batch_xs, batch_ys = X_train[idx: idxn], y_train[idx: idxn]            
                _, l=sess.run([train_op, loss], feed_dict={X: batch_xs, y: batch_ys})
                losses.append(l)
                predicts_test = sess.run(predict, feed_dict={X: X_test})
                predicts_train = sess.run(predict, feed_dict={X: X_train})
                print("epoch: %d train accuracy: %0.3f test accuracy: %0.3f" % (e, accuracy(predicts_train, y_train), accuracy(predicts_test)))
        if accuracy(predicts_test, y_test) > max_accuracy:
            max_accuracy = accuracy(predicts_test)
            best_batch_size = batch_size
            best_hid_size = hid_size
            best_learning_rate = learning_rate
            best_num_epochs = num_epochs
            best_regularization_strength = regularization_strength
    file_writer = tf.summary.FileWriter('/tmp/tensor', sess.graph)                                                                        


# In[35]:


print("best batch size: %d, best hidden size: %d, best learning rate: %f, best regularization strength: %f, best num epochs: %d, best test set accuracy: %f" % (best_batch_size, best_hid_size, best_learning_rate, best_regularization_strength, best_num_epochs, max_accuracy))


# In[36]:


#Evaluation and TensorBoard plots of the optimal model selected in the previous code block

batch_size = 64
hid_size = 50
learning_rate = 0.05
num_epochs = 20
regularization_strength = 0.001
input_size = X_train.shape[1]
output_size = 10

X = tf.placeholder("float32", shape=[None, input_size])
y = tf.placeholder("int32", shape=[None])

W_h = init_weights([input_size, hid_size])
b_h = init_weights([hid_size])
W_o = init_weights([hid_size, output_size])
b_o = init_weights([output_size])
h = tf.nn.sigmoid(tf.matmul(X,W_h)+b_h)
out_act = tf.matmul(h, W_o)+b_o + regularization_strength * tf.nn.l2_loss(W_h) + regularization_strength * tf.nn.l2_loss(W_o)
    
# build the loss, predict, and train operator
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_act, labels=y)
loss = tf.reduce_sum(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

predict = tf.nn.softmax(out_act)

#writer = tf.summary.FileWriter('events_log', sess.graph)
#writer.add_graph(graph=sess.graph)

loss_summ = tf.summary.scalar("cross_entropy_loss", loss)
W_h_summ = tf.summary.histogram("weights", W_h)
b_h_summ = tf.summary.histogram("biases", b_h)

#merge all summaries into a single "operation" which we can execute in a session
merged_summary_op = tf.summary.merge_all()

# Initialization of all variables in the graph
init = tf.global_variables_initializer()

# run training
with tf.Session() as sess: 
    sess.run(init)
    
    # For monitoring purposes
    writer = tf.summary.FileWriter('./events_log', sess.graph)
    
    losses = []
    
    for e in range(num_epochs):
        for i in range(X_train.shape[0] // batch_size):
            idx, idxn = i * batch_size, min(X_train.shape[0]-1, (i+1) * batch_size)
            batch_xs, batch_ys = X_train[idx: idxn], y_train[idx: idxn]            
           
            _,l=sess.run([train_op, loss], feed_dict={X: batch_xs, y: batch_ys})
            
             #create a summary object and set its value
            summary = tf.Summary(value=[tf.Summary.Value(tag="cross_entropy_loss", simple_value=l)])
            writer.add_summary(summary, e * X_train.shape[0] // batch_size + i)
            
            losses.append(l)
            
            predicts_test = sess.run(predict, feed_dict={X: X_test})
            predicts_train = sess.run(predict, feed_dict={X: X_train})
        
            '''
            summary2 = tf.Summary(value=[tf.Summary.Value(tag="train_accuracy", simple_value=accuracy(predicts_train, y_train))])
            writer.add_summary(summary2,e)

            summary3 = tf.Summary(value=[tf.Summary.Value(tag="test_accuracy", simple_value=accuracy(predicts_test))])
            writer.add_summary(summary3,e)
            '''
        print("epoch: %d train accuracy: %0.3f test accuracy: %0.3f" % (e, accuracy(predicts_train, y_train), accuracy(predicts_test)))
        
plt.plot(losses);

writer.flush()
writer.close()


# In[ ]:




