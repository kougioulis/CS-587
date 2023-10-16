#!/usr/bin/env python
# coding: utf-8

# # CS - 587:  Exercise 4b
# 
# We will use a pretrained model, in our case a VGG_16 to compute class saliency maps as described in Section 3.1 of [2].
# 
# A saliency map tells us the degree to which each pixel in the image affects the classification score for that image. To compute it, we compute the gradient of the unnormalized score corresponding to the correct class (which is a scalar) with respect to the pixels of the image. If the image has shape (H, W, 3) then this gradient will also have shape (H, W, 3); for each pixel in the image, this gradient tells us the amount by which the classification score will change if the pixel changes by a small amount. To compute the saliency map, we take the absolute value of this gradient, then take the maximum value over the 3 input channels; the final saliency map thus has shape (H, W) and all entries are nonnegative.
# 
# [2] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. *"Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps"*, ICLR Workshop 2014.

# In[1]:


import numpy as np
import tensorflow as tf
from models import vgg16
from Utilities import utils
import matplotlib.pyplot as plt
from matplotlib import pylab as P
import os


# In[2]:


#Load the image to be processed
img = utils.load_image("flamingo1.jpg")

#Plot the image of this magestic creature
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()


# In[3]:


# Add some useful functions that will ease the visualization process
def VisualizeImageGrayscale(image_3d, percentile=99):
    #Returns a 3D tensor as a grayscale 2D tensor.

    #This method sums a 3D tensor across the absolute value of axis=2, and then
    #clips values at a given percentile.
 
    image_2d = np.sum(np.abs(image_3d), axis=2)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)
    
    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

def VisualizeImageDiverging(image_3d, percentile=99):
    #Returns a 3D tensor as a 2D tensor with positive and negative values.
    image_2d = np.sum(image_3d, axis=2)

    span = abs(np.percentile(image_2d, percentile))
    vmin = -span
    vmax = span

    return np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)


# ## Class Saliency Map
# 
# In order to compute the saliency map of a specific class the following steps are required:
# Following are the steps involved:
# 
# 1. Do a forward pass of the image through the network.
# 2. Calculate the scores for every class.
# 3. Enforce derivative of score S at last layer for all classes except class C to be 0. For C, set it to 1.
# 4. Backpropagate this derivative till the start.
# 5. Render the gradients and you have your Saliency Map!
# 
# More information about the role of each step can be found in:
# https://www.silversparro.com/single-post/2018/01/26/Understanding-Deep-Learning-Networks-using-Saliency-Maps

# In[4]:


##############################------PART  4 -------------##################################################################
#                                                                                                                         # 
# We will start with steps 3 and 4: we will define a class that enforces the derivative of an input (Score class)         # 
# to be 1 and backpropagates this derivative till the start.                                                              #
#                                                                                                                         #
###########################################################################################################################

class Saliency(object):
    #Base class for saliency masks. Alone, this class doesn't do anything.#
    def __init__(self, graph, session, y, x):
        #Constructs a SaliencyMask by computing dy/dx.

        #Args:
        #  graph: The TensorFlow graph to evaluate masks on.
        #  session: The current TensorFlow session.
        #  y: The output tensor to compute the SaliencyMask against. This tensor
        #      should be of size 1.
        #  x: The input tensor to compute the SaliencyMask against. The outer
        #      dimension should be the batch size.
    
        # y must be of size one, otherwise the gradient we get from tf.gradients
        # will be summed over all ys.
        
        size = 1
        for shape in y.shape:
            size *= shape
        assert size == 1
        
        self.graph = graph
        self.session = session
        self.y = y
        self.x = x
        
    def Mask_GEN(self, x_value, feed_dict={}):
        #Returns an unsmoothed mask.

        #Args:
        #  x_value: Input value, not batched.
        #  feed_dict: (Optional) feed dictionary to pass to the session.run call.
        
        raise NotImplementedError('A derived class should implemented GetMask()')
    
###################################################################################
# TODO: Implement the Gradient saliecy class so as to do the following:           #
#              a) Returns the backward pass (gradient ascent) of the activation   #
#                  w.r.t to the input image                                       # 
#                                                                                 #
###################################################################################    

class Get_Gradient(Saliency):
    #A SaliencyMask class that computes saliency masks with a gradient.

    def __init__(self, graph, session, y, x):
        
        ############################################################
        #       Start of your code #HINT: take a look at tf.gradients
        # Backward pass gradient must be stored in the gradients_node variable  
        
        super(Get_Gradient, self).__init__(graph, session, y, x)
        self.gradients_node = tf.gradients(y, x)[0]
        #pass
        #       End of your code
        ############################################################
        
    def Mask_GEN(self, x_value, feed_dict={}):
        #Returns a vanilla gradient mask.

        #Args:
        #  x_value: Input value, not batched.
        #  feed_dict: (Optional) feed dictionary to pass to the session.run call.
        
        feed_dict[self.x] = [x_value]
        return self.session.run(self.gradients_node, feed_dict=feed_dict)[0]


# In[5]:


#resize_image = img.reshape(1, 224, 224, 3) #----LOOK AT ME! I am the current CAPTAIN image now!!!-----#############
#SO FEED ME INTO THE VGG's stomach!------------------------------------------------------

resize_image = img
graph = tf.Graph()

with graph.as_default():
    
    sess = tf.Session(graph=graph)

    ######################################################################################
    # TODO: Implement entire Saliency computation pipeline by:                           #
    #              a) Computing steps 1 and 2                                            #
    #           TODO a.1) Do a forward pass of the image through the network.            #
    #           TODO a.2) Calculate the scores for every class.                          #
    #             b) Moving to steps 3 and 4                                             # 
    #           TODO  b.1) Take the logit for the specific class (the one that gives     #
    #                       the highest prediction)  -- should output 130 - make a print #
    #           TODO  b.2) Call Get_Gradient() to compute gradients                      #
    #           TODO  b.3) Call Get_Gradient.Mask_GEN for the SPECIFIC GRADIENTS to      #
    #                       compute the Mask of activations                              #
    #           TODO  b.4) Call VisualizeImageGrayscale to visualize the Saliency Map    #
    #                                                                                    #
    ######################################################################################    
    
    # START of code for Task a -  HINT: call vgg16.VGG16.build (INPUT TENSOR) to first build the model (load weights, etc.), 
    # to take the output of your model just call model.prob, where model is the VGG model that you have created.
  
    # Steps 1-2 of a, name the oupts of the model as logits
  
    # START
    

    # ENTER YOUR CODE HERE    
    #pass
    images = tf.placeholder(tf.float32, shape=(1, 224, 224, 3)) #create placeholder tensor
    
    model = vgg16.Vgg16()
    model.build(images) #build VGG16 on images tf tensor
  
    logits = model.prob #obtain final layer logits 

    # END 

    neuron_selector = tf.placeholder(tf.int32) 
    y = logits[0][neuron_selector]
  
    # TODO Step 1 of b
    # Define the operation that finds the class that gives me the highest score, use tf.argmax
    # RUN the operation, via session.run and input the placeholder of the image. Print the result! Should be 130.
 
    prediction = tf.argmax(logits, 1)
    prediction_class = sess.run(prediction, feed_dict={images: [resize_image]})[0]
    print("Prediction class: " + str(prediction_class))  # class 130 is flamingo, which is correct! (see https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)

    # TODO Step 2 of b
    # Use as input on the function, the graph, the current session and the placeholder of the input image.   
    #pass
    gradient_CS587 = Get_Gradient(graph, sess, y, images)
    
    # END of code... up to task b.2
    # Step 3 of b --- PARTIALLY IMPLEMENTED, just fill what I ask you in CAPS
 
    # Compute the vanilla mask.
    
    #mask_3d = gradient_CS587.Mask_GEN(img, feed_dict = {neuron_selector: YOUR OPERATION TO GET THE PREDICTED})
    mask_3d = gradient_CS587.Mask_GEN(img, feed_dict={neuron_selector: prediction_class})


# In[6]:


##############################################################################  
# ENTER THE CODE HERE FOR task b.4
#############################################################################
# Call the visualization methods to convert the 3D tensors to 2D grayscale.
# INPUT will be the mask_3d that you generated in step 3.b
# TO do the visuallization in grayscale call the function VisualizeImageGrayscale
# The method returns an image. TO SHOW IT use imshow function of matplotlib.

visualized_mask = VisualizeImageGrayscale(mask_3d)
plt.imshow(visualized_mask, cmap='gray')
plt.axis('off')
plt.show()

#pass

#END of code for task b.4


# In[5]:


######################################################################################
# TODO: Repeat the process for the rest of images, located in folder = MyImages      #
#              a) Load them and create the tensor to hold them (i.e. correct shape)  #
#              b) Repeat the steps in previous task for ALL new images               #
#              c) Show predicted class for each image and plot Saliency Map          #
#                                                                                    #
######################################################################################           
    
#image collection 
image_folder = "MyImages"
image_files = os.listdir(image_folder)

resized_images = []

graph = tf.Graph()

with graph.as_default():
    sess = tf.Session(graph=graph)

    images_tensor = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))

    model = vgg16.Vgg16()
    model.build(images_tensor)

    logits = model.prob

    neuron_selector = tf.placeholder(tf.int32)
    y = logits[0][neuron_selector]

    prediction = tf.argmax(logits, 1)

    for image_file in image_files:

        image_path = os.path.join(image_folder, image_file)
        image = utils.load_image(image_path)

        print("Plotting images...")    
        imgplot = plt.imshow(image)
        plt.axis('off')
        plt.show()

        resized_image = image

        prediction_class = sess.run(prediction, feed_dict={images_tensor: [resized_image]})[0]
        print("Prediction class for image", image_file, ":", prediction_class)

        gradient_CS587 = Get_Gradient(graph, sess, y, images_tensor)

        mask_3d = gradient_CS587.Mask_GEN(resized_image, feed_dict={neuron_selector: prediction_class})

        visualized_mask = VisualizeImageGrayscale(mask_3d)
        plt.imshow(visualized_mask, cmap='gray')
        plt.axis('off')
        plt.show()


# In[ ]:




