#!/usr/bin/env python
# coding: utf-8

# # CS - 587: Exercise 3a
# ## Scope:
# The goal of this assignment is to get familiar with fine-tunning in a new dataset a Convolutional Neural Network (CNN) that has been trained in another dataset, taking advantage of transfer learning.
# 
# In your assignment you will be fine-tunning AlexNet a popular CNN architecture, that has been pretrained on ImageNet dataset. Your network will be finetuned for the task of recognizing art painting categories in a large dataset of art painting images, known as Wikiart.
# 
# The WikiArt dataset, which consists of 3000 images of paintings of arbitrary sizes from 10 different styles - Baroque, Realism, Expressionism, etc.
# 
# 

# In[1]:


#Cross-check all packages and dependencies

import sys
print(sys.version, sys.platform, sys.executable)


# In[2]:


get_ipython().system('python --version')


# In[3]:


import numpy
print(numpy.__path__)
print(numpy.__version__)


# In[4]:


import tensorflow as tf
print(tf.__version__)


# In[5]:


import os
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
import tensorflow as tf

import numpy as np
from models.AlexNet import AlexNet
from Utilities.datagenerator import ImageDataGenerator

from datetime import datetime
from tqdm import tqdm
import urllib

"""
Configuration settings
"""
weight_path= os.path.join('weights','bvlc_alexnet.npy')
general_path_weights = os.path.join('weights')

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Create parent path if it doesn't exist
if not os.path.isdir(general_path_weights): 
    os.mkdir(general_path_weights)
    
if os.path.isfile(weight_path) == False:
    print('Went it to download weights for AlexNet')
    print('Relax...')
    weight_file = urllib.request.urlretrieve('http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy',os.path.join('weights/bvlc_alexnet.npy'))
    print('Done with weights')
else:
    print('GOOD TO GO! Weights already downloaded and stored!')
    

tf.logging.set_verbosity(tf.logging.ERROR)

# Path to the textfiles for the trainings and validation set
#training_dirname = os.path.join('data','wikiart', 'train.txt')
training_dirname = os.path.join('/home','nikolas','Downloads','CS-587','Assignments','CS587_3rd_Assignment','CS587-Assignment_3','Utilities','data','train.txt')
print("Training dirname:", training_dirname)

#val_dirname = os.path.join('data','wikiart', 'test.txt')
val_dirname = os.path.join('/home','nikolas','Downloads','CS-587','Assignments','CS587_3rd_Assignment','CS587-Assignment_3','Utilities','data','test.txt')

# Path for tf.summary.FileWriter and to store model checkpoints
general_path = os.path.join('finetune_alexnet')
filewriter_path = os.path.join('finetune_alexnet','wikiart')
checkpoint_path = os.path.join('finetune_alexnet','CheckPoint')

# Create parent path if it doesn't exist
if not os.path.isdir(general_path): 
    os.mkdir(general_path)
# Create parent path if it doesn't exist
if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)


# In[6]:


# Learning params
learning_rate = 0.01
num_epochs = 10
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 10

#---------------LOOK AT ME----------------------------------------------------

#0.40625
train_layers = ['fc7','fc8'] # Change me if you want to try stuff

#0.39062
#train_layers = ['fc8'] # Change me if you want to try stuff

#------------------------------------------------------------------------------

# How often we want to write the tf.summary data to disk
#display_step = 1
display_step = 3


# In[7]:


tf.reset_default_graph() 


# # Pretrained Model
# For all of our image generation experiments, we will start with a convolutional neural network which was pretrained to perform image classification on ImageNet. We can use any model here, but for the purposes of this assignment we will use AlexNet

# In[8]:


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Link variable to model output
score = model.fc8

##################################################################################################
# TODO: Implement the (a) loss function (Soft-max Cross Entropy), (b) the optimization           #
# process using S Gradient Descent, (c) accuracy (using argmax). Create summaries in tensorboard #
# for the loss, the gradients of trainable variables (histogram form) and the accuracy.          #
#                                                                                                # 
# Hint: in order to take the gradients per variable use tf.gradients(,)                          #
##################################################################################################    
 
#pass

#part (a)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

#part (b)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss, var_list=var_list)
  
#part (c)
true_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(true_pred, tf.float32))

#creation of TF summaries
tf.summary.scalar("Loss", loss)
tf.summary.scalar("Accuracy", accuracy)

#gradients histogram
for variable in var_list:
    tf.summary.histogram(variable.name + '/gradients', tf.gradients(loss, variable))

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(training_dirname, 
                                     horizontal_flip = True, shuffle = True)
val_generator = ImageDataGenerator(val_dirname, shuffle = False) 

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

#print(train_batches_per_epoch)


# In[9]:


# Start Tensorflow session
with tf.Session() as sess:
    
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
  
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
  
    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)
  
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                    filewriter_path))
    # Loop over number of epochs
    for epoch in range(num_epochs):
        
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        
        step = 1
        
        for step in tqdm(range(train_batches_per_epoch), desc="AlexNet Training"):
        #for step in tqdm(range(train_batches_per_epoch)):
            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            
            #######################################################################################
            #TODO: Run the training operation, print the current loss value write the symmaries.  #
            #      The summarries must be written every 3 batches                                 #
            #######################################################################################
                              
            #pass
                 
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate})

            loss_value, train_acc = sess.run([loss, accuracy], feed_dict={x: batch_xs,
                                                                    y: batch_ys,
                                                                    keep_prob: 1.0})
            #obtain the summaries every display_step = 3 batches
            if step % display_step == 0:
                
                summary = sess.run(merged_summary, feed_dict={x: batch_xs, 
                                                              y: batch_ys, 
                                                              keep_prob: 1.0})
                #write the summaries to monitor
                writer.add_summary(summary, epoch*train_batches_per_epoch + step)

            print("Iteration {}, Mini-batch Loss= {:.5f}, Training Accuracy= {:.5f}".format(step*batch_size, loss_value, train_acc))
            
            step += 1
            
            #End of this task
            
        ############################################################
        #TODO: Validate the model on the ENTIRE validation set     #
        #      Print the final validation accuracy of your model   #
        # Do not forget to use validation batches                  #
        # use val_batches_per_epoch  variable                      #
        ############################################################   
        
        #pass
        
        # perform validation on the validation batches
        print("Starting validation...")
        validation_acc, validation_count = 0, 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            val_acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                y: batch_ty,
                                                keep_prob: 1.0})
        validation_acc += val_acc
        validation_count += 1
        validation_acc /= validation_count
        print("Validation Accuracy = {:.5f}".format(validation_acc))
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        
        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        
        print("{} Saving checkpoint of model...".format(datetime.now()))  
        
        #save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)  
        
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        


# In[ ]:




