import os
import warnings
import numpy as np
from keras.layers import Input, Activation, Conv2D, Dense, Dropout, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add
from keras.models import Model
from keras import regularizers

# copied from keras.applications.MobilenetV2, but allows for smaller networks,
# by changing some of the stride-2 convolutions to stride-1

# build MobileNetV2 models
def MobileNetV2(input_shape=(32, 32, 3),
                alpha=1.0,
                depth_multiplier=1,
                include_top=True,
                pooling=None,
                classes=10,
                stride2_use=None,
                do_dropout=True,
                do_weight_reg=True,
                config=None):

    if config is not None:
        input_shape = config.image_size
        classes = config.num_classes
        stride2_use = config.stride2_use
        do_dropout = config.do_dropout
        do_weight_reg = config.do_weight_regularization


        # fileter size (first block)
    first_block_filters = _make_divisible(32 * alpha, 8)
    # input shape  (first block)
    img_input = Input(shape=input_shape)

    if stride2_use is None:
        stride2_use = list(range(5))  # default: do all 5 of the strided convolutions

        # convert list of indexes in which stride is 2 -> full list of 5 strides
        # eg. [2,3,4] -> [1, 1, 2, 2, 2]
    strides = [(2 if i in stride2_use else 1) for i in range(5)]

    conv_kwargs = dict(kernel_initializer="he_normal")
    if do_weight_reg:
        conv_kwargs['kernel_regularizer']=regularizers.l2(4e-5)

    # model architechture
    x = img_input


    x = Conv2D(first_block_filters, kernel_size=3, strides=strides[0], padding='same', use_bias=False, **conv_kwargs, name='Conv1')(x)  # 0
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16,  alpha=alpha, stride=1, expansion=1, block_id=0, conv_kwargs=conv_kwargs )

    x = _inverted_res_block(x, filters=24,  alpha=alpha, stride=strides[1], expansion=6, block_id=1, conv_kwargs=conv_kwargs )  #1
    x = _inverted_res_block(x, filters=24,  alpha=alpha, stride=1, expansion=6, block_id=2, conv_kwargs=conv_kwargs )

    x = _inverted_res_block(x, filters=32,  alpha=alpha, stride=strides[2], expansion=6, block_id=3, conv_kwargs=conv_kwargs )  #2     ***
    x = _inverted_res_block(x, filters=32,  alpha=alpha, stride=1, expansion=6, block_id=4, conv_kwargs=conv_kwargs )
    x = _inverted_res_block(x, filters=32,  alpha=alpha, stride=1, expansion=6, block_id=5, conv_kwargs=conv_kwargs )
    
    x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=strides[3], expansion=6, block_id=6, conv_kwargs=conv_kwargs )  #3    ***
    x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=1, expansion=6, block_id=7, conv_kwargs=conv_kwargs )
    x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=1, expansion=6, block_id=8, conv_kwargs=conv_kwargs )
    x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=1, expansion=6, block_id=9, conv_kwargs=conv_kwargs )
    if do_dropout:
        x = Dropout(rate=0.25)(x)

    x = _inverted_res_block(x, filters=96,  alpha=alpha, stride=1, expansion=6, block_id=10, conv_kwargs=conv_kwargs)
    x = _inverted_res_block(x, filters=96,  alpha=alpha, stride=1, expansion=6, block_id=11, conv_kwargs=conv_kwargs)
    x = _inverted_res_block(x, filters=96,  alpha=alpha, stride=1, expansion=6, block_id=12, conv_kwargs=conv_kwargs)
    if do_dropout:
        x = Dropout(rate=0.25)(x)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=strides[4], expansion=6, block_id=13, conv_kwargs=conv_kwargs)  #4   *** 2,3,4
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14, conv_kwargs=conv_kwargs)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15, conv_kwargs=conv_kwargs)
    if do_dropout:
        x = Dropout(rate=0.25)(x)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16, conv_kwargs=conv_kwargs)
    if do_dropout:
        x = Dropout(rate=0.25)(x)

    # define fileter size (last block)
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280


    x = Conv2D(last_block_filters, kernel_size=1, use_bias=False, **conv_kwargs, name='Conv_1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = ReLU(6., name='out_relu')(x)


    # top layer ("use" or "not use" FC)
    if include_top:
        x = GlobalAveragePooling2D(name='global_average_pool')(x)
        #x = Flatten()(x)
        x = Dense(classes, activation=None, use_bias=True, name='Logits')(x)
        x = Activation('softmax', name='softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # create model of MobileNetV2 (for CIFAR-10)
    model = Model(inputs=img_input, outputs=x, name='mobilenetv2_cifar10')
    return model



# define the calcuration of each 'Res_Block'
def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id,
                        conv_kwargs):
    prefix = 'block_{}_'.format(block_id)

    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs

    # Expand
    if block_id:
        x = Conv2D(expansion * in_channels, kernel_size=1, strides=1, padding='same', use_bias=False, activation=None, **conv_kwargs, name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same', **conv_kwargs, name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters, kernel_size=1, strides=1, padding='same', use_bias=False, activation=None, **conv_kwargs, name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)


    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x


# define the filter size
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
