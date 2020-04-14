import gtc
import keras

#from mnist.config import config
from quantization import quant_utils as quant_utl
from networks import net_utils as net
import tensorflow.compat.v1 as tf
import os
from keras.datasets import mnist
import numpy as np

tf.compat.v1.disable_eager_execution()

# create LeNet model using keras Sequential
def LeNet(config, x_placeholder):

    network_name = net.get_network_full_name_with_tag(
        config, default='LeNet_GTC')

    num_layers = 5
    quantizers = quant_utl.QuantizerSet(num_layers=num_layers, config=config, prefix=network_name)

    model = gtc.GTCModel(name=network_name)
    model.add(x_placeholder)

    if config.quantize_input:
        model.add_linear_quantizer(config.num_bits_linear, quant_name='LQ_in',
                                   layer_name='input_quantized')


    net.add_conv_bn_act_layers(model, filters=6, kernel_size=5, use_bias=False, config=config,
                               quant_dict=quantizers.next(), base_name=network_name + 'conv1')
    model.add( gtc.MaxPooling2D( strides=2, pool_size=2, name=network_name + 'max_pool1'))

    net.add_conv_bn_act_layers(model, filters=16, kernel_size=5, use_bias=False, config=config,
                               quant_dict=quantizers.next(), base_name=network_name + 'conv2' )
    model.add( gtc.MaxPooling2D( strides=2, pool_size=2, name=network_name + 'max_pool2'))


    model.add(gtc.Flatten(name=network_name+'flatten'))

    q_dict_dense1 = quantizers.next()
    model.add(gtc.Dense(units=120,use_bias=False,
                        quantizer=q_dict_dense1['conv'], name=network_name + 'dense1'))
    model.add(gtc.Activation(activation='relu',
                             quantizer=q_dict_dense1['activation'], name=network_name + 'relu1'))
    #model.add(gtc.Dropout(0.5))

    q_dict_dense2 = quantizers.next()
    model.add(gtc.Dense(units=84, use_bias=False,
                        quantizer=q_dict_dense2['conv'], name=network_name + 'dense2'))
    model.add(gtc.Activation(activation='relu',
                             quantizer=q_dict_dense2['activation'], name=network_name + 'relu2'))


    if config.quantize_features:
        model.add_linear_quantizer(config.num_bits_linear, quant_name='LQ_feat',
                                   layer_name='features_quantized')

    q_dict_logits = quantizers.next()
    #model.add(gtc.Dropout(0.5))
    model.add(gtc.Dense(units=config.num_classes, use_bias=False,
                        quantizer=q_dict_logits['conv'], name=network_name + 'logits'))

    if config.quantize_logits_output:
        model.add_linear_quantizer(config.num_bits_linear, quant_name='LQ_logits',
                                   layer_name='logits_quantized')

    quantizers.confirm_done()
    return model




# create LeNet model using keras Sequential
def LeNet_v2(config, x_placeholder):

    network_name = net.get_network_full_name_with_tag(
        config, default='LeNet_GTC')
    network_name = ''
    num_layers = 5
    quantizers = quant_utl.QuantizerSet(num_layers=num_layers, config=config, prefix=network_name)

    model = gtc.GTCModel(name=network_name)
    model.add(x_placeholder)

    quant_dict1 = quant_dict=quantizers.next()
    model.add( gtc.Conv2D(filters=6, kernel_size=5, use_bias=True, padding='VALID',
                          quantizer=quant_dict['conv'], name='Conv1/Conv1'))
    if config.do_batchnorm:
        model.add(gtc.BatchNormalization(name='bn_Conv1/bn_Conv1', share=config.batchnorm_share))
    model.add( gtc.Activation('relu', quantizer=quant_dict1['activation']) )
    model.add( gtc.MaxPooling2D( strides=2, pool_size=2, name='max_pool1'))

    quant_dict2 = quantizers.next()
    model.add( gtc.Conv2D(filters=16, kernel_size=5, use_bias=True, padding='VALID',
                          quantizer=quant_dict2['conv'],name='Conv2/Conv2'))
    if config.do_batchnorm:
        model.add(gtc.BatchNormalization(name='bn_Conv2/bn_Conv2', share=config.batchnorm_share))
    model.add( gtc.Activation('relu', quantizer=quant_dict2['activation']) )
    model.add( gtc.MaxPooling2D( strides=2, pool_size=2, name='max_pool2') )


    model.add(gtc.Flatten())

    q_dict_dense1 = quantizers.next()
    model.add(gtc.Dense(units=120,use_bias=False,
                        quantizer=q_dict_dense1['conv'], name=network_name + 'Dense1/Dense1'))
    model.add(gtc.Activation(activation='relu',
                             quantizer=q_dict_dense1['activation'], name=network_name + 'dense_relu1'))

    q_dict_dense2 = quantizers.next()
    model.add(gtc.Dense(units=84, use_bias=False,
                        quantizer=q_dict_dense2['conv'], name=network_name + 'Dense2/Dense2'))
    model.add(gtc.Activation(activation='relu',
                             quantizer=q_dict_dense2['activation'], name=network_name + 'dense_relu2'))

    q_dict_logits = quantizers.next()
    model.add(gtc.Dense(units=config.num_classes, use_bias=True,
                        quantizer=q_dict_logits['conv'], name=network_name + 'Logits/Logits'))

    quantizers.confirm_done()
    return model

# create LeNet model using keras Sequential
def mlp_gtc(config, x_placeholder):

    network_name = net.get_network_full_name_with_tag(
        config, default='LeNet_GTC')
    network_name = ''
    num_layers = 5
    quantizers = quant_utl.QuantizerSet(num_layers=num_layers, config=config, prefix=network_name)

    model = gtc.GTCModel(name=network_name)
    model.add(x_placeholder)

    #quant_dict1 = quant_dict=quantizers.next()
    #model.add( gtc.Conv2D(filters=6, kernel_size=5, use_bias=True, padding='VALID',
    #                      quantizer=quant_dict['conv'], name='Conv1/Conv1'))
    #model.add( gtc.BatchNormalization(name='bn1/bn1', share=config.batchnorm_share))
    #model.add( gtc.Activation('relu', quantizer=quant_dict1['activation']) )
    #model.add( gtc.MaxPooling2D( strides=2, pool_size=2, name='max_pool1'))


    model.add(gtc.Flatten())

    #q_dict_dense1 = quantizers.next()
    #model.add(gtc.Dense(units=120,use_bias=False,
    #                    quantizer=q_dict_dense1['conv'], name=network_name + 'dense1/dense1'))
    #model.add(gtc.Activation(activation='relu',
    #                         quantizer=q_dict_dense1['activation'], name=network_name + 'dense_relu1'))

    q_dict_logits = quantizers.next()
    model.add(gtc.Dense(units=config.num_classes, use_bias=True,
                        quantizer=q_dict_logits['conv'], name=network_name + 'Logits/Logits'))

    #quantizers.confirm_done()
    return model


def main():
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

    learning_rate = tf.compat.v1.placeholder(tf.float32, name='lr')
    sess = tf.compat.v1.InteractiveSession()
    keras.backend.set_learning_phase(True)
    model = LeNet(config, x_placeholder)
    model.compile(verbose=True)
    #model.initialize(sess, initialize_weights=True)
    print('Compiled Model')
    model.print_model(sess=sess)
    if train_model:
        y_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, 10))


    test_save_hp_lp = True
    weights_file_hp = _BASEDIR + 'bitnet/test/lenet_hp.h5'
    weights_file_lp = _BASEDIR+ 'bitnet/test/lenet_lp.h5'
    if test_save_hp_lp:
        model.save_weights_for_keras_model(weights_file_hp, hp=True)
        model.save_weights_for_keras_model(weights_file_lp, hp=False)


    test_json_save = True
    if test_json_save:
        json_file = _BASEDIR + 'bitnet/test_lenet.json'
        model.save_model_def(json_file)

        model2 = gtc.GTCModel()
        model2.load_model_def(json_file)
        #model2

    test_load_save_weights = True
    if test_load_save_weights:
        tmp_save_file = _BASEDIR + 'models/lenet/sample_weights.h5'
        if os.path.exists(tmp_save_file):
            model.load_weights(tmp_save_file, sess=sess)
            model2.load_weights(tmp_save_file, sess=sess)
        else:
            model.save_weights(tmp_save_file, sess=sess)

    test_copy_model_weights = False
    if test_copy_model_weights:
        args2 = net.NetConfig(image_size=(32, 32, 3), num_classes=10, network_tag='prime')

        model2 = LeNet(args2, x_placeholder)
        model.compile(verbose=True)

        model2.copy_weights_from_model(model)
        model2.print_model(sess=sess)


    test_create_keras_model = True
    if test_create_keras_model:
        #args2 = net.NetConfig(image_size=(32, 32, 3), num_classes=10, network_tag='prime')

        #model2 = LeNet(args2, x_placeholder)
        #model.compile(verbose=True)
        model_file = _BASEDIR + 'bitnet/test/lenet.json'
        weights_file = _BASEDIR + 'bitnet/test/lenet.h5'
        model.save_weights(weights_file)
        model.save_model_def(model_file)

        model.save_weights_for_keras_model(weights_file_hp, hp=True)
        model.save_weights_for_keras_model(weights_file_lp, hp=False)

        keras_model_hp = gtc.singlePrecisionKerasModel(
            model_file=model_file, weights_file=weights_file_hp, hp=True, verbose=True)
        keras_model_hp.save(_BASEDIR + 'bitnet/test/hpmodel.hp')

        keras_model_lp = gtc.singlePrecisionKerasModel(
            model_file=model_file, weights_file=weights_file_lp, hp=False)
        keras_model_lp.save(_BASEDIR + 'bitnet/test/lpmodel.hp')


if __name__ == "__main__":
    _BASEDIR = '/home/gtc-tensorflow/lenetOutput/'
    
    main()

    print('Succesfully compiled LeNet' )
    print(' --> Done !')
