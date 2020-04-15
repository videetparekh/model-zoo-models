import tensorflow.compat.v1 as tf
from gtc.kerasLayer import GTCKerasLayer
import tensorflow.keras as keras
from functools import partial

use_functional_API = True


class Pooling(GTCKerasLayer):
    """
    Applies pooling operation on the gtc layer
    """
    pass



class MaxPooling2D(Pooling):

    def __init__(self, pool_size=2, strides=1, quantizer=None,
                 padding='SAME', data_format='channels_last', name=None, **kwargs):

        keras_layer = keras.layers.MaxPooling2D(
            pool_size=pool_size, strides=strides, padding=padding,
            data_format=data_format)

        self._layer_config = dict(pool_size=pool_size, strides=strides,
                                  padding=padding, data_format=data_format)

        super(MaxPooling2D, self).__init__(
            keras_layer, quantizer=quantizer, name=name, **kwargs)




class AveragePooling2D(Pooling):

    def __init__(self, pool_size=2, strides=1, quantizer=None,
                 padding='SAME', data_format='channels_last', name=None, **kwargs):

        keras_layer = keras.layers.AveragePooling2D(
            pool_size=pool_size, strides=strides, padding=padding,
            data_format=data_format)

        self._layer_config = dict(pool_size=pool_size, strides=strides,
                                  padding=padding, data_format=data_format)

        super(AveragePooling2D, self).__init__(
            keras_layer, quantizer=quantizer, name=name, **kwargs)




class GlobalAveragePooling2D(Pooling):

    def __init__(self, quantizer=None,
                 data_format='channels_last', name=None, **kwargs):
        keras_layer = keras.layers.GlobalAveragePooling2D(
            data_format=data_format)

        self._layer_config = dict(data_format=data_format)

        super(GlobalAveragePooling2D, self).__init__(
            keras_layer, quantizer=quantizer, name=name, **kwargs)


class GlobalMaxPooling2D(Pooling):

    def __init__(self, quantizer=None,
                 data_format='channels_last', name=None, **kwargs):
        keras_layer = keras.layers.GlobalMaxPooling2D(
            data_format=data_format)

        self._layer_config = dict(data_format=data_format)

        super(GlobalMaxPooling2D, self).__init__(
            keras_layer, quantizer=quantizer, name=name, **kwargs)
