#!/usr/bin/env python3

import argparse
import os

import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow.compat.v1 as tf
import gtc
from networks import net_utils as net
import numpy as np
from dataset_loaders import GetGTCDatasetLoader, DatasetInfo
from tf.keras.datasets import mnist

def top1_acc(labels, logits):
    return keras.metrics.top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=1)


def top5_acc(labels, logits):
    return keras.metrics.top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=5)



if __name__ == '__main__':
    # constants

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        required=True,
        help='Path to folders of labeled images'
    )
    parser.add_argument(
        '--input_model_path',
        type=str,
        default=None,
        required=True,
        help='Load gtc model at path'
    )
    args = parser.parse_args()

    _BASEDIR = '/home/gtc-tensorflow/lenetOutput/' # = args.input_model_path

    network_name = 'lenet'   #args.network_name
    dataset_name = 'mnist'  #args.dataset_name
    dataset_info = DatasetInfo(dataset_name)
    # DatasetLoader = GTCMNistDatasetLoader # This one is specifically for MNIST
    DatasetLoader = GetGTCDatasetLoader(dataset_name)  # More general. for any of the supported datasets.

    model_name = network_name + '_trained_on_' + dataset_name

    nb_classes = dataset_info.num_classes
    imsize = dataset_info.imsize

    config = net.NetConfig(image_size=(32, 32, 3), num_classes=10,
                           conv_quant='gtc', bn_quant='identity', act_quant='identity')

    x_placeholder = tf.compat.v1.placeholder(
        tf.float32, [None, config.image_size[0], config.image_size[1], config.image_size[2]])
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape from rank 3 tensor to rank 4 tensor
    x_train = np.reshape(x_train, (x_train.shape[0], args.image_size[0],
                                   config.image_size[1], config.image_size[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], config.image_size[0],
                                 config.image_size[1], config.image_size[2]))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print('Dataset Statistics')
    print('Training data shape', x_train.shape)
    print('Testing data shape', x_test.shape)

    # data generator
    datagen = ImageDataGenerator(rescale=1. / 255)
    print('Number of training samples', x_train.shape[0])
    print('Number of test samples', x_test.shape[0], '\n')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, args.num_classes)
    y_test = keras.utils.to_categorical(y_test, args.num_classes)

    json_file = _BASEDIR + 'bitnet/test_lenet.json'

    sess = tf.compat.v1.InteractiveSession()
    keras.backend.set_learning_phase(True)
    model = gtc.GTCModel()
    model.load_model_def(json_file)
    #model = LeNet(config, x_placeholder)
    model.load_weights(_BASEDIR + 'models/lenet/sample_weights.h5', sess=sess)
    
    validation_data_dir = args.dataset_path
    
    print('\n# Evaluate')
    result = model.evaluate(validation_generator)
    print(dict(zip(model.metrics_names, result)))
