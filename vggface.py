from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import data

def vgg_face(data, input_maps):

    # read meta info
    meta = data['meta']
    classes = meta['classes']
    class_names = classes[0][0]['description'][0][0]
    normalization = meta['normalization']
    average_image = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
    image_size = np.squeeze(normalization[0][0]['imageSize'][0][0])
    input_maps = tf.image.resize_images(input_maps, size=[image_size[0], image_size[1]])

    # read layer info
    layers = data['layers']
    current = input_maps - np.array([129.1863, 104.7624, 93.5940]).reshape((1, 1, 1, 3))
    network = {}
    
    with tf.variable_scope('VGG-Face', reuse=tf.AUTO_REUSE):
        for layer in layers[0]:
            name = layer[0]['name'][0][0]
    
            if name == 'relu6' or name == 'fc6' or name == 'fc7' or name == 'relu7' or name == 'fc8' or name == 'prob' or name == 'pool5':
                continue
            else:
                layer_type = layer[0]['type'][0][0]
                if layer_type == 'conv':
                    if name[:2] == 'fc':
                        padding = 'VALID'
                    else:
                        padding = 'SAME'
                    stride = layer[0]['stride'][0][0]
                    kernel, bias = layer[0]['weights'][0][0]
                    bias = np.squeeze(bias).reshape(-1)
                    conv = tf.nn.conv2d(current, tf.constant(kernel),
                                        strides=(1, stride[0], stride[0], 1), padding=padding)
                    current = tf.nn.bias_add(conv, bias)
                elif layer_type == 'relu':
                    current = tf.nn.relu(current)
                elif layer_type == 'pool':
                    stride = layer[0]['stride'][0][0]
                    pool = layer[0]['pool'][0][0]
                    current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                             strides=(1, stride[0], stride[0], 1), padding='SAME')
                elif layer_type == 'softmax':
                    current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))
    
            network[name] = current
    return network

def face_embedding(vgg_weights, img):
    img = (img + 1.0) * 127.5
    if np.shape(img)[3] == 1:
        img = tf.tile(img, [1, 1, 1, 3])
    net = vgg_face(vgg_weights, img)
    return net['conv1_2'], net['conv2_2'], net['conv3_2'], net['conv4_2'], net['conv5_2']

def id_preserve(idnet_w, real, fake):
    real_conv1_2, real_conv2_2, real_conv3_2, real_conv4_2, real_conv5_2 = face_embedding(idnet_w, real)
    fake_conv1_2, fake_conv2_2, fake_conv3_2, fake_conv4_2, fake_conv5_2 = face_embedding(idnet_w, fake)
    conv1_2_loss = tf.reduce_mean(tf.abs(real_conv1_2 - fake_conv1_2)) / 224. / 224.
    conv2_2_loss = tf.reduce_mean(tf.abs(real_conv2_2 - fake_conv2_2)) / 112. / 112.
    conv3_2_loss = tf.reduce_mean(tf.abs(real_conv3_2 - fake_conv3_2)) / 56. / 56.
    conv4_2_loss = tf.reduce_mean(tf.abs(real_conv4_2 - fake_conv4_2)) / 28. / 28.
    conv5_2_loss = tf.reduce_mean(tf.abs(real_conv5_2 - fake_conv5_2)) / 14. / 14.
    vgg_loss = conv1_2_loss + conv2_2_loss + conv3_2_loss + conv4_2_loss + conv5_2_loss
    return vgg_loss
    

if __name__ == '__main__':
    data_tr = data.ImgDataPair('./exp_data/CK+/cross_validation1/train', 128, 32, 
                          pair=True, repeat=1)
    img, _, label = data_tr.get_next()
    vgg_path = './data/vgg-face.mat' # download from http://www.vlfeat.org/matconvnet/pretrained/
    vgg_weights = loadmat(vgg_path)
    output = face_embedding(vgg_weights, img)
    
    

