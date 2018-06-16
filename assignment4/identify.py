import tensorflow as tf
import os
import numpy as np
import cv2

from mnist_deep import deepnn

checkpoint_dir = './checkpoint'


def get_test_data():
    names = sorted(os.listdir('./process'))
    test_data = np.zeros((len(names), 28 * 28), dtype=np.float32)
    for i in range(len(names)):
        test_img = cv2.imread('./process/' + names[i], 0)
        # cv2.imwrite('./output/%d.jpg' % (i), test_img)
        test_img = test_img / 255
        test_data[i, :] = test_img.reshape(-1)

    return test_data


def identify_test():
    test_data = get_test_data()
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)
    # prediction
    with tf.name_scope('prediction'):
        prediction = tf.argmax(y_conv, 1)
        prediction = tf.cast(prediction, tf.int32)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        predictions = prediction.eval(feed_dict={x: test_data, keep_prob: 1.0})

        return predictions


if __name__ == '__main__':
    res = identify_test()
    print(res)
