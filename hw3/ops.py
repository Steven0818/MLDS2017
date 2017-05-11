import tensorflow as tf
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from PIL import Image
import os

def z_fc_layer(name, input, n_neuron, img_size, add_bias=True, activation=tf.nn.relu, train_phase=True):
    
    in_size = input.get_shape().as_list()[1]
    n_chn = n_neuron // img_size // img_size
    
    weights = tf.get_variable('w_' + name, shape=[in_size, n_neuron], initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    h = tf.matmul(input, weights) 

    if add_bias:
        biases = tf.get_variable('b_' + name, shape=[n_neuron], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        h = h + biases
    
    h = tf.reshape(h, [-1, img_size, img_size, n_chn])
    h_bn = batch_norm(name, h, train_phase)
    
    return activation(h_bn, name='h_' + name)
    
    
def transpose_conv_layer(name, input , ksize, output_shape, strides=[1,2,2,1], add_bias=True, activation=tf.nn.relu, train_phase=True):
    weights = tf.get_variable('w_' + name, shape=ksize, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    conv = tf.nn.conv2d_transpose(input, weights, output_shape, strides=strides, padding='SAME')

    if add_bias:
        biases = tf.get_variable('b_' + name, shape=[ksize[2]], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        conv = tf.nn.bias_add(conv, biases)

    conv_bn = batch_norm(name, conv, train_phase)
    return activation(conv_bn, name='h_' + name)

    
def conv_layer(name, input, ksize, strides=[1,2,2,1], add_bias=True, do_bn=True, activation=tf.nn.relu, train_phase=True):
    
    weights = tf.get_variable('w_' + name, shape=ksize, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    conv = tf.nn.conv2d(input, weights, strides=strides, padding='SAME')

    if add_bias:
        biases = tf.get_variable('b_' + name, shape=[ksize[3]], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        conv = tf.nn.bias_add(conv, biases)

    if do_bn:
        conv = batch_norm(name, conv, train_phase)
    
    return activation(conv, name='h_' + name) 

def rf_fc_layer(name, input, n_neuron, add_bias=False):
    in_size = input.get_shape().as_list()[1]

    weights = tf.get_variable('w_' + name, shape=[in_size, n_neuron], initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    h = tf.matmul(input, weights)
    
    if add_bias:
        biases = tf.get_variable('b_' + name, shape=[n_neuron], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        h = h + biases
    
    return h
    
def batch_norm(name, input, train_phase):
    return tf.cond(train_phase,
            lambda: tf.contrib.layers.batch_norm(input, is_training=True,
                    updates_collections=None, center=True, scale=True, scope="bn_" + name),
            lambda: tf.contrib.layers.batch_norm(input, is_training=False,
                    updates_collections=None, center=True, scale=True, scope="bn_" + name, reuse = True))

def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)                    
                    
def unprocess_image(image, mean_pixel, norm):
    return image * norm + mean_pixel 
     
                    
def save_imshow_grid(images, result_dir, filename, grid_shape):
    
    img_shape = images.shape[1:]
    img_grid = np.zeros((img_shape[0]*grid_shape[0], img_shape[1]*grid_shape[1], 3))
    
    for x in range(grid_shape[0]):
        for y in range(grid_shape[1]):
            img_grid[x*img_shape[0]: (x+1)*img_shape[0], y*img_shape[1]: (y+1)*img_shape[1],:] = images[x*grid_shape[0]+y]
    
<<<<<<< HEAD
    cv2.imwrite(os.path.join(result_dir, filename), img_grid)
=======
    img = Image.fromarray(img_grid, 'RGB')
    img.save(os.path.join(result_dir, filename))
>>>>>>> replace cv with pil
