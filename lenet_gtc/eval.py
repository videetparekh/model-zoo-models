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
from tensorflow.keras.datasets import mnist

tf.compat.v1.disable_v2_behavior()

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

    _BASEDIR = '/home/model-zoo-models/lenet_gtc/' # = args.input_model_path

    network_name = 'lenet'   #args.network_name
    dataset_name = 'mnist'  #args.dataset_name
    dataset_info = DatasetInfo(dataset_name)
    # DatasetLoader = GTCMNistDatasetLoader # This one is specifically for MNIST
    DatasetLoader = GetGTCDatasetLoader(dataset_name)  # More general. for any of the supported datasets.
    datagen_test = DatasetLoader(split='test')
    lambda_weights = net.LambdaWeights(hp=1, # must be 1 for GTC
                                        lp=0,    # must be 0 for GTC 
                                        distillation_loss=0.01,
                                        bit_loss=1e-5)
    datagen_test.build(
        featurewise_center=True,        # apply the same normalizations, but don't
        featurewise_std_normalization=True, # do any augmentations on the test set.
        lambda_weights=lambda_weights)

    model_name = network_name + '_trained_on_' + dataset_name

    nb_classes = dataset_info.num_classes
    imsize = dataset_info.imsize

    config = net.NetConfig(image_size=(32, 32, 3), num_classes=10,
                           conv_quant='gtc', bn_quant='identity', act_quant='identity')

    x_placeholder = tf.compat.v1.placeholder(
        tf.float32, [None, config.image_size[0], config.image_size[1], config.image_size[2]])
    (_, _), (x_test, y_test) = mnist.load_data()

    # # reshape from rank 3 tensor to rank 4 tensor
    # x_test = np.reshape(x_test, (x_test.shape[0], config.image_size[0],
    #                              config.image_size[1], config.image_size[2]))

    # x_test = x_test.astype('float32')
    # print('Dataset Statistics')
    # print('Testing data shape', x_test.shape)

    # # data generator
    # datagen = ImageDataGenerator(rescale=1. / 255)
    # print('Number of test samples', x_test.shape[0], '\n')

    # # convert class vectors to binary class matrices
    # y_test = keras.utils.to_categorical(y_test, args.num_classes)

    json_file = _BASEDIR + 'models/lenet_trained_on_mnist_final_weights.json'

    sess = tf.compat.v1.InteractiveSession()
    keras.backend.set_learning_phase(True)
    gtc_model = gtc.GTCModel()
    gtc_model.load_model_def(json_file)
    #model = LeNet(config, x_placeholder)
    gtc_model.load_weights(_BASEDIR + 'models/lenet_trained_on_mnist_final_weights.h5', sess=sess)
    
    # Create a keras model wrapper for the gtc model.
    keras_model = gtc_model.build_keras_model(lambda_weights, verbose=True)

    # Once trained, you can
    evaluate_on_test_dataset_again = True
    if evaluate_on_test_dataset_again:
        # evaluate on a single example
        datagen_test = DatasetLoader(split='test')
        datagen_test.build(
        featurewise_center=True,        # apply the same normalizations, but don't
        featurewise_std_normalization=True, # do any augmentations on the test set.
        lambda_weights=lambda_weights)
        test_dataLoader = datagen_test.flow(batch_size=1)
        num_samples_test = 3
        for j in range(num_samples_test):
            x,y = test_dataLoader[j]
            y_gt_label = int( np.argmax( y[0] ) )

            # for the keras_model wrapper, the predict() function calculates
            # both outputs (hp and lp), concatenated in a list
            y_pred = keras_model.predict(x)
            y_pred_hp, y_pred_lp = y_pred[0], y_pred[1]

            # You can also use the 'predict' method of the underlying gtcModel.
            # This accepts an argument 'hp', if hp=True, it uses the high-precision branch
            # if hp=False, it uses the low-precision branch.
            y_pred_hp2 = gtc_model.predict(x, hp=True)
            y_pred_lp2 = gtc_model.predict(x, hp=False)

            y_pred_label_hp = int(np.argmax(y_pred_hp))
            y_pred_label_lp = int(np.argmax(y_pred_lp))

            assert np.allclose(y_pred_hp, y_pred_hp2)
            assert np.allclose(y_pred_lp, y_pred_lp2)
            print('Sample %d' % (j+1))
            print(' Actual GT label:      %d. \n Predicted label (hp): %d. \n Predicted label (lp): %d\n' % (
                y_gt_label, y_pred_label_hp, y_pred_label_lp) )

        # After training you can also use the underlying gtcModel in a similar way
        test_dataLoader = datagen_test.flow(batch_size=batch_size, max_num_batches=val_steps, shuffle=False)
        print('Evaluating on test set using Keras wrapper model:')
        net.eval_classification_model_on_dataset(keras_model, test_dataLoader,
                                                hp=True, lp=True)

        test_dataLoader = datagen_test.flow(batch_size=batch_size, max_num_batches=val_steps, shuffle=False)
        print('Evaluating on test set using gtc model:')
        net.eval_classification_model_on_dataset(gtc_model, test_dataLoader,
                                                hp=True, lp=True)

