import tensorflow.compat.v1 as tf
from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict
import tensorflow.keras as keras
import numpy as np



def hard_sigmoid_relu6(x):
    # Note: there is already a keras/tensorflow implementation of hard_sigmoid,
    # which is x =  max(min( (0.2 * x) + 0.5,  1), 0)
    # This implementation takes advantage of the optimized versions of relu6 on many platforms.

    # tensorflow implementation of mobilenet-v3 hard-sigmoid
    return tf.nn.relu6(x+3)*0.16667  
    
    # this version would have been more consistent with definition of h-swish
    #return tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.) 


def swish(x, beta=1):
    # swish(x) = x * sigmoid(x)
    return (x * keras.activations.sigmoid(beta * x))

def hard_swish(x):
    # hard-swish(x) = x * hard_sigmoid(x)
    # from tensorflow implementation of mobilenet-v3 hswish
    return x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)  


new_activations = {'relu6': keras.layers.Activation(tf.nn.relu6),
                   'swish': keras.layers.Activation(swish),                   
                   'hsigmoid': keras.layers.Activation(hard_sigmoid_relu6),  # use hsigmoid to avoid overwriting native keras 'hard_sigmoid' activation
                   'hswish': keras.layers.Activation(hard_swish)}

import keras.utils.generic_utils
keras.utils.generic_utils.get_custom_objects().update(new_activations)



class Activation(GTCLayer):
    """
    Implements activation function for both gtc layers
    TODO quantizer
    """

    def __init__(self, activation, quantizer=None, name=None, trainable=False, **kwargs):

        self._activation = activation
        self._activation_layer = keras.layers.Activation(activation)
        self.supports_masking = True  # used to skip timesteps, for RNN models

        self._layer_config = dict(activation=activation)

        super(Activation, self).__init__(
            quantizer=quantizer, trainable=trainable, name=name, **kwargs)

    def call(self, inputs, **kwargs):  # note: ignores kwargs ?
        hp_input, lp_input = unpackGTCDict(inputs)

        hp_output = self._activation_layer(hp_input)
        lp_output = self._activation_layer(lp_input)
        if self._quantizer is not None:
            lp_output = self._quantizer.quantize(lp_output)

        return GTCDict(hp=hp_output, lp=lp_output)

    def number_of_bits(self, some_var):
        if isinstance(some_var, tf.Tensor):
            some_tensor = some_var
        else:
            some_tensor = some_var['hp']
        print('some tensor', some_tensor)
        return self._quantizer.number_of_bits(some_tensor)

    def quantization_error_for_output(self, some_var):
        if isinstance(some_var, tf.Tensor):
            some_tensor = some_var
        else:
            some_tensor = some_var['hp']
        return (self._quantizer.quantization_error(some_tensor), )

    def quantization_angle_difference_for_output(self, some_var):
        if isinstance(some_var, tf.Tensor):
            some_tensor = some_var
        else:
            some_tensor = some_var['hp']
        return (self._quantizer.quantization_angle_difference(some_tensor), )

