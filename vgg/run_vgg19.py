"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19
import utils

import numpy as np
import os


# configuration
# currently i assume the ImageNet dataset consists of tons of image files and a single corresponding label file
num_total_images = 0
num_classes = 0
num_batch_size = 10
path_dataset = "../../big_data/Imagenet_dataset/"
# path_dataset = "../../../dataset/ImageNet/"
learning_rate = 1e-05

# load image set
# at this stage, just load filename rather than real data
# img1 = utils.load_image("./test_data/tiger.jpeg")
dataset_images = list()
dataset_labels = list() # for test
for subdir in os.listdir(path_dataset):
    if subdir.startswith('.'):
        continue
    for image_file_name in os.listdir(path_dataset + subdir):
        image = path_dataset + subdir + '/' + image_file_name
        dataset_images.append(image)
        dataset_labels.append(subdir) # for test
        num_total_images += 1
num_classes = len(set(dataset_labels)) # for test
classes = list(set(dataset_labels)) # for test
for cls_i in range(len(classes)): # for test
    for i in range(len(dataset_labels)):
        if dataset_labels[i] == classes[cls_i]:
            dataset_labels[i] = [1 if j == cls_i else 0 for j in range(num_classes)]
# generate synset.txt (labels' text for printing)
# TODO: for test, we just have the first four categories
with open("./synset.txt", "w") as f:
    for cls in classes:
        f.write(cls + "\n")
    #for i in range(num_classes, num_classes): # for test
        #f.write("fake_class " + str(i) + "\n")
# for i in list(set(dataset_labels)):


# load label set
# assume each line of labelfile stand for a label
# img1_true_result = [1 if i == 292 else 0 for i in xrange(1000)]  # 1-hot result for tiger

# dataset_labels = list()
# with open(path_dataset + "image_net_label", "r") as f:
    # line = int(f.readlines())
    # dataset_labels.append(line) 
# num_classes = max(dataset_labels) + 1 # i assume the num of labels start from 0

if __name__=='__main__':
    sess = tf.InteractiveSession()

    images = tf.placeholder(tf.float32, [num_batch_size, 224, 224, 3])
    labels = tf.placeholder(tf.float32, [num_batch_size, num_classes]) # totally num_classes categories
    train_mode = tf.placeholder(tf.bool)

    #vgg = vgg19.Vgg19('./vgg19.npy')
    vgg = vgg19.Vgg19(num_batch_size, ln_mode=True, cln_mode=False)
    vgg.get_tr()
    vgg.build_net(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    # print vgg.get_var_count()
    #print(vgg.get_var_count())

    sess.run(tf.initialize_all_variables())

    # iterate until the whole dataset is nearly trained 
    # remain a proportion of images which is insufficient to construct one batch
    num_data_trained = 0
    for _ in range(10000):

        # a batch of data
        batch_images = list()
        batch_labels = list()

        batch_start = num_data_trained
        batch_end = batch_start + num_batch_size
        # construct a batch of training data (images & labels)
        for one_sample in range(batch_start, batch_end):
            # here we load real data
            image_file = utils.load_image(dataset_images[one_sample])
            # set the image batch size
            # batch1 = image_file.reshape((num_batch_size, 224, 224, 3))
            batch_images.append(image_file)
            print(image_file.shape, dataset_images[one_sample])
            batch_labels.append(dataset_labels[one_sample])

        # convert list into array
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        
        # TODO: official codes convert (num_batch_size, length, width, depth) into (num_batch_size, length * width * depth)
        # but our training function is different
        batch_images = batch_images.reshape((num_batch_size, 224, 224, 3))
        
        train_feed_dict = {
            images : batch_images,
            labels : batch_labels,
            train_mode : True
        }

        test_feed_dict = {images: batch_images, train_mode: False}

        # simple 1-step training, train with one image 
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(vgg.prob, labels))
        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(labels, 1))
        #TODO: PRINT THIS OUT
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(train, feed_dict=train_feed_dict)
        pred = sess.run(vgg.prob, feed_dict=test_feed_dict)
        for i in range(10):
            utils.print_prob(pred[i], './synset.txt')
        
        print(accuracy.eval(feed_dict={images: batch_images, labels: batch_labels, train_mode: False}))

        test_feed_dict = {
            images : batch_images,
            train_mode : False
        }

        # test classification
        # prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        #prob = sess.run(vgg.prob, feed_dict = test_feed_dict)
        # TODO: labels text file...
        #utils.print_prob(prob[0], './synset.txt')

        # num of training images for next epoch is less than our batch size
        #num_data_trained += num_batch_size
        #(num_data_trained + num_batch_size) > num_total_images:
        
    # TODO: train the last training images

    # test save
    vgg.save_npy(sess, './test-save.npy')