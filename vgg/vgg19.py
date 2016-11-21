import os 
import tensorflow as tf
from tensorflow.contrib.layers import layer_norm
import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]
num_classes = 5

class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, trainable=True, ln_mode=False):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.ln_mode = ln_mode

    def get_tr(self):
        one = np.ones(1000)
        self.tr_fc8 = tf.constant(one)
        self.tr_fc7 = self.bp_fc(self.tr_fc8, 1000, 4096, "tr_fc8")
        one = np.ones(4096)
        self.tr_fc6 = self.bp_fc(self.tr_fc7, 4096, 4096, "tr_fc7")

        self.tr_pool5 = self.bp_fc_2_pool(self.tr_fc6)
        self.tr_conv5_4 = self.unpool2x2(self.tr_pool5, 'tr_conv5_4')
        self.tr_conv5_3 = self.bp_conv(self.tr_conv5_4, 512, 512, 'tr_conv5_3')
        self.tr_conv5_2 = self.bp_conv(self.tr_conv5_3, 512, 512, 'tr_conv5_2')
        self.tr_conv5_1 = self.bp_conv(self.tr_conv5_2, 512, 512, 'tr_conv5_1')

        self.tr_pool4 = self.bp_conv(self.tr_conv5_1, 512, 512, 'tr_pool4')
        self.tr_conv4_4 = self.unpool2x2(self.tr_pool4, 'tr_conv4_4')
        self.tr_conv4_3 = self.bp_conv(self.tr_conv4_4, 512, 512, 'tr_conv4_3')
        self.tr_conv4_2 = self.bp_conv(self.tr_conv4_3, 512, 512, 'tr_conv4_2')
        self.tr_conv4_1 = self.bp_conv(self.tr_conv4_2, 512, 512, 'tr_conv4_1')

        self.tr_pool3 = self.bp_conv(self.tr_conv4_1, 512, 256, 'tr_pool3')
        self.tr_conv3_4 = self.unpool2x2(self.tr_pool3, tr_conv3_4)
        self.tr_conv3_3 = self.bp_conv(self.tr_conv3_4, 256, 256, 'tr_conv3_3')
        self.tr_conv3_2 = self.bp_conv(self.tr_conv3_3, 256, 256, 'tr_conv3_2')
        self.tr_conv3_1 = self.bp_conv(self.tr_conv3_2, 256, 256, 'tr_conv3_1')

        self.tr_pool2 = self.bp_conv(self.tr_conv3_1, 256, 128, 'tr_pool2')
        self.tr_conv2_2 = self.unpool2x2(self.tr_pool2, 'tr_conv2_2')
        self.tr_conv2_1 = self.bp_conv(self.tr_conv2_2, 128, 128, 'tr_conv2_1')

        self.tr_pool1 = self.bp_conv(self.tr_conv2_1, 128, 64, 'tr_pool1')
        self.tr_conv1_2 = self.unpool2x2(self.pool1, 'tr_conv1_2')
        self.tr_conv1_1 = self.bp_conv(self.tr_conv1_2, 64, 64, 'tr_conv1_1')
        self.tr_brg = self.bp_conv(self.tr_conv1_1, 64, 3, 'tr_brg')

    def build_net(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5') 

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 / (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, 0.5), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, 0.5), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)

        self.fc8 = self.fc_layer(self.relu7, 4096, num_classes, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
    
    def bp_fc(self, x, in_size, out_size, name):
        fc_name = '_'.join(name.split('_')[1:])
        with tf.variable_scope(fc_name):
            w, b = self.get_fc_var(in_size, out_size, fc_name)

            #TODO: add reshape
            tr = tf.matmul(x, w)

            return fc

    def bp_fc_2_pool(self, x, batch=1, height=7, width=7, channels=3):
        return tf.reshape(x, [batch, height, width, channels])

    def bp_conv(self, x, in_channel, out_channel, name, output_shape):
        conv_name = '_'.join(name.split('_')[1:])
        with tf.variable_scope(name):
            #filters, b = self.get_conv_var(3, out_channels, in_channels, name)
            #f_shape = filters.get_shape()
            filters = tf.ones([3, 3, out_channel, in_channel])
            deconv = tf.nn.conv2d_transpose(x, filters, output_shape, strides=[1, 1, 1, 1])
            return deconv

    def unpool2x2(input_, name, in_channels=3):
        """N-dimensional version of the unpooling operation from
        https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

        :param input_: A Tensor of shape [b, d0, d1, ..., dn, ch]
        :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
        """
        with tf.name_scope(name) as scope:
            sh = input_.get_shape().as_list()
            #dim = len(sh[1:-1])
            out = (tf.reshape(input_, [-1] + sh[-2:]))
            for i in range(2, 0, -1):
                out = tf.concat(i, [out, out])
            out_size = [-1] + [s * 2 for s in sh[1:-1]] + [in_channels]
            out = tf.reshape(out, out_size, name=scope)
            return out

    def avg_pool(self, input_, name):
        return tf.nn.avg_pool(input_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, input_, name):
        return tf.nn.max_pool(input_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, input_, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filters, b = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(input_, filters, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, b)
            output = tf.nn.relu(bias)
            if self.ln_mode:
                return layer_norm(output, center=True, scale=True, trainable=True)
            #tr_name = 'tr_' + name 
            #tr_weight = self.get_var(None, tr_name, None, tr_name)
            #tr_weight = tf.nn.moments(tr_weight, [1, 2])
            #tr_sum = tf.reduce_sum(tr_weight, [1, 2])
            #tr_weight = tf.sub(tf_weight, tf.tile(tr_sum, tr_weight.get_shape()[:-1]))
            #weighted_relu = tf.mul(tr_weight, relu)
            #return weighted_relu
            return output

    def fc_layer(self, input_, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            #x = tf.reshape(input_, [-1, in_size])  
            x = tf.reshape(input_, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        #initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if not data_dict.has_key(name):
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

    def get_bp_rate(self):
        return 