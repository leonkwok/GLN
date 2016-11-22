def bp_fc(self, x, in_size, out_size, name):
    fc_name = '_'.join(name.split('_')[1:])
    with tf.variable_scope(fc_name):
        w, b = self.get_fc_var(in_size, out_size, fc_name)

        #TODO: add reshape
        tr = tf.matmul(x, w)

        return fc

def bp_fc_2_pool(self, x, height=7, width=7, depth=512):
    return tf.reshape(x, [1, height, width, depth])

def bp_conv(self, x, output_shape, name,
            in_channel, out_channel):
    conv_name = '_'.join(name.split('_')[1:])
    with tf.variable_scope(name):
        #filters, b = self.get_conv_var(3, out_channels, in_channels, name)
        #f_shape = filters.get_shape()
        filter_ = tf.ones([3, 3, out_channel, in_channel])
        deconv = tf.nn.conv2d_transpose(x, filter_, output_shape, strides=[1, 1, 1, 1])
        return deconv

def unpool2x2(self, input_, name, in_channels=3):
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


