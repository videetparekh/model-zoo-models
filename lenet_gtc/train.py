import os
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import tensorflow.keras as keras
keras.backend.set_learning_phase(True)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger, LearningRateScheduler

from networks import net_utils as net

from datasets.dataset_loaders import GetGTCDatasetLoader, DatasetInfo

import networks
import numpy as np
#from leip.compress.training.config import config
from config import config
from gtc.gtcModel import singlePrecisionKerasModel


def main(_BASEDIR, args):
    print(args)
    # set meta params
    make_GTC_model = True
    #args.verbose_output = True
    #args.save_weights   = True
    batch_size = args.batch_size
    nb_epoch = args.num_epochs
    network_name = 'lenet'   #args.network_name
    dataset_name = 'mnist'  #args.dataset_name
    dataset_info = DatasetInfo(dataset_name)
    DatasetLoader = GetGTCDatasetLoader(dataset_name)  # More general. for any of the supported datasets.
    model_name = network_name + '_trained_on_' + dataset_name

    nb_classes = dataset_info.num_classes
    imsize = dataset_info.imsize

    # The lambda weights define how much weights are given to different portions of
    # the loss function. (e.g. hp=weight given to the cross-entropy loss of the
    # high-precision branch, usually set to 1. lp=weight given to the cross-entropy
    # loss of the low-precision branch, usually set to 0. to train the low-precision
    # branch, the distillation loss should be a nonzero (but small) value, eg. 0.01.
    # to add a cost to the number of bits (to encourage bit compression),
    # provide a non-zero bit_loss (should usually be very small, e.g. ~1e-5, but
    # the precise value should scale with the number of quantized layers in the network.)  )
    if make_GTC_model:
        lambda_weights = net.LambdaWeights(hp=1, # must be 1 for GTC
                                        lp=0,    # must be 0 for GTC 
                                        distillation_loss=args.lambda_distillation_loss,
                                        bit_loss=args.lambda_bit_loss)

        do_lp_network = lambda_weights.lp is not None or \
                        lambda_weights.distillation_loss is not None or \
                        lambda_weights.bit_loss is not None
    else:
        lambda_weights = None
        do_lp_network = False

    quantize_inputs_logits_outputs = False

    config = net.NetConfig(image_size=imsize,
                        stride2_use=[2,3,4],  # only needed for mobilenet-v2 on smaller datasets
                        include_preprocessing=False,
                        batchnorm_share='none',
                        do_batchnorm=True,
                        do_dropout=False,
                        identity_quantization=not do_lp_network,
                        do_weight_regularization=False,
                        num_classes=nb_classes,
                        quant_pre_skip=1,
                        quant_post_skip=1,
                        quantize_input=quantize_inputs_logits_outputs,
                        quantize_features=quantize_inputs_logits_outputs,
                        quantize_logits_output=quantize_inputs_logits_outputs,
                        conv_quant='gtc', bn_quant='identity', act_quant='identity')

    # load data

    datagen_train = DatasetLoader(split='train')
    datagen_train.build(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=0.0,   # optional dataset augmentation (mostly disabled for now)
        width_shift_range=0,
        height_shift_range=0,
        vertical_flip=False,
        horizontal_flip=False,
        # we have to supply the the lambda weights to the (customized) data generator.
        # Depending on which losses are not 'None', it duplicates the 'y' (target output)
        # appropriately so that the same target output is provided for the hp and
        # lp branches. it also provides a dummy vector of zeros for the distillation
        # and bitloss outputs (in keras, each output requires a target), if those
        # losses are not 'None'.
        lambda_weights=lambda_weights)



    datagen_test = DatasetLoader(split='test')
    datagen_test.build(
        featurewise_center=True,        # apply the same normalizations, but don't
        featurewise_std_normalization=True, # do any augmentations on the test set.
        lambda_weights=lambda_weights)


    optimizer=args.optimizer

    sess = tf.InteractiveSession()

    crossentropy_loss = keras.losses.categorical_crossentropy



    if make_GTC_model:
        print('Building a GTC model')
        x_placeholder = tf.placeholder(tf.float32, [None] + list(config.image_size[:3]))

        # use 'net_type' class to define a network type.
        # e.g.: net_type = net.NetworkType('resnet', num_layers=50)
        # or    net_type = net.NetworkType('vgg', num_layers=50)

        if network_name == 'lenet':
            net_type = net.NetworkType('lenet')
        elif network_name == 'mobilenet':
            net_type = net.MobilenetType(version=2)
        elif network_name == 'resnet18':
            net_type = net.NetworkType('resnet', num_layers=18, style='ic')
        else:
            raise ValueError('Undefined network')

        config.net_style = net_type.style
        gtc_model = net.GetNetwork(net_type)(config, x_placeholder)

        # Use print_model() to show the layers of the model.

        # Create a keras model wrapper for the gtc model.
        keras_model = gtc_model.build_keras_model(lambda_weights, verbose=True)

        gtc_model.initialize(sess, initialize_weights=True, verbose=False)

        if args.verbose_output :
            gtc_model.print_model()

        loss_use, metrics_use, loss_weights_use = \
            keras_model.gtcModel.lambda_weights.AugmentLossesAndMetrics(hp_lp_loss=crossentropy_loss, hp_lp_metric=net.acc)

        # Use keras summary() to show the keras view of the model.
        # In this view, all the gtc layers are wrapped up in a single layer
        # called 'gtc_model_keras_layer'
        keras_model.summary()


    else:
        print('Building a Pure keras model')
        gtc_model = None
        keras_model = None

        from networks.keras_lenet import LeNet_keras
        from networks.mobilenet_v2_keras import MobileNetV2 as MobileNetV2_keras

        if network_name == 'lenet':
            keras_model = LeNet_keras(config)
        elif network_name == 'mobilenet_v2':
            keras_model = MobileNetV2_keras(config=config)
        else:
            # define the an alternate keras model here
            pass

        assert keras_model is not None, "Please define an alternate keras model"

        keras_model.save_weights('test.h5')

        loss_use = crossentropy_loss
        loss_weights_use = None
        metrics_use = [net.acc]      # accuracy in percent

        keras_model.summary()


    # compile keras model
    keras_model.compile(optimizer=optimizer,
                        loss=loss_use,
                        loss_weights=loss_weights_use,
                        metrics=metrics_use)



    #LearningRateScheduler
    # Define
    callbacks = []
    os.makedirs(_BASEDIR + "models/", exist_ok=True)

    save_model_checkpoint = args.save_model_checkpoint
    if save_model_checkpoint:
        model_cp_path =  _BASEDIR + 'models/' + model_name + '_checkpoint.h5'
        monitor = 'val_lp_loss' if make_GTC_model else 'val_loss'
        callbacks.append(ModelCheckpoint(model_cp_path, monitor=monitor,
                                        save_best_only=True, save_weights_only=make_GTC_model))


    train_steps_per_epoch = int( np.ceil(len(datagen_train) / batch_size) )
    val_steps = int( np.ceil(len(datagen_test) / batch_size) )


    history = keras_model.fit_generator(
        datagen_train.flow(batch_size=batch_size, shuffle=True),
        steps_per_epoch=train_steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=callbacks,
        validation_data=datagen_test.flow(batch_size=batch_size),
        validation_steps= val_steps
        )
    # keras_ext = '' if make_GTC_model else '_keras'
    # q_act_ext = '_qact' if quantize_inputs_logits_outputs else ''

    # final_weights_file = _BASEDIR + 'weights/' + model_name + '_final_weights' + keras_ext + q_act_ext + '.h5'
    # keras_model.load_weights(final_weights_file)

    if make_GTC_model and args.verbose_output:
        # If you pass in the current tensorflow session (sess), print_model will also
        # display the quantization parameters
        gtc_model.print_model(sess=sess)


    save_weights = args.save_weights or args.print_layers_bits
    if save_weights:
        keras_ext = '' if make_GTC_model else '_keras'
        q_act_ext = '_qact' if quantize_inputs_logits_outputs else ''

        final_weights_file = _BASEDIR + 'models/' + model_name + '_final_weights' + keras_ext + q_act_ext + '.h5'
        keras_model.save_weights(final_weights_file)

        if make_GTC_model:
            final_weights_file_hp = final_weights_file.replace('.h5', '_hp.h5')
            final_weights_file_lp = final_weights_file.replace('.h5', '_lp.h5')
            model_file = final_weights_file.replace('.h5', '.json')

            gtc_model.save_model_def(model_file)
            gtc_model.save_weights_for_keras_model(final_weights_file_hp, hp=True)
            gtc_model.save_weights_for_keras_model(final_weights_file_lp, hp=False)

        # Alternatively, can also call using keras_model wrapper:
        #if make_GTC_model:
        #    gtc_model.save_weights(final_weights_file)

        # To Load weights, do:
        # keras_model.load_weights(final_weights_file)

        # After training you can also use the underlying gtcModel in a similar way
        test_dataLoader = datagen_test.flow(batch_size=batch_size, max_num_batches=val_steps, shuffle=False)
        print('Evaluating on test set using Keras wrapper model:')
        net.eval_classification_model_on_dataset(keras_model, test_dataLoader,
                                                hp=True, lp=True)

        test_dataLoader = datagen_test.flow(batch_size=batch_size, max_num_batches=val_steps, shuffle=False)
        print('Evaluating on test set using gtc model:')
        net.eval_classification_model_on_dataset(gtc_model, test_dataLoader,
                                                hp=True, lp=True)

    print_layers_bits = args.print_layers_bits
    if print_layers_bits:
        print('----- Number of bits used per each layer ------')
        bits_per_layer = gtc_model.get_bits_per_layer()
        for k in bits_per_layer:
            print('Layer ', k, ' with ',bits_per_layer[k].eval(),' bits') 

    savemodel = args.savemodel
    if (savemodel == 'HP') or (savemodel == 'BOTH'):
        keras_hp_model = singlePrecisionKerasModel(model_file, final_weights_file_hp, hp=True,
                              add_softmax = True, verbose=args.verbose_output)
        keras_hp_model.save(_BASEDIR + "models/hp_model.hd5")

    if (savemodel == 'LP') or (savemodel == 'BOTH'):
        keras_lp_model = singlePrecisionKerasModel(model_file, final_weights_file_lp, hp=False,
                              add_softmax = True, verbose=args.verbose_output)
        keras_lp_model.save(_BASEDIR + "models/lp_model.hd5")
        


if __name__ == "__main__":
################
    make_GTC_model = True           # uncomment this line to make GTC model.
    # make_GTC_model = False        # uncomment this line to make a pure keras model.

    network_name = 'lenet'         # uncomment this line to use lenet
    # network_name = 'mobilenet_v2'   # uncomment this line to use mobilenet-v2
    #network_name = 'resnet18'        # uncomment this line to use resnet18

    dataset_name = 'mnist'         # uncomment this line to use mnist
    #dataset_name = 'cifar10'        # uncomment this line to use cifar10

################
    _BASEDIR = '/home/model-zoo-models/lenet_gtc/'
    args = config(_BASEDIR,
                name_of_experiment='checking_regularization',
                batch_size=32,
                num_epochs=1,
                learning_rate=.0002,
                weight_decay=.0002,
                image_size=(28, 28),
                num_channels=1,
                lambda_bit_loss=1e-5,
                lambda_distillation_loss=0.01,
                )
    main(_BASEDIR, args)



