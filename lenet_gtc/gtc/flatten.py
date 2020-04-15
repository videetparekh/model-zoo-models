from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import backend as K


class Flatten(GTCLayer):
    """
    implements flatten operation on the gtc layer output
    """

    def __init__(self, name=None, **kwargs):
        """TODO: Docstring for __init__.
        :returns: Flattens the input tensor to vector
        Note: does not use keras.layers.Flatten() layer, or the lower-level
        keras.backend.flatten() function, as these do not preserve information
        about the final output shape. Instead, we use the K.reshape
        """
        trainable = kwargs.pop('trainable', False)
        super(Flatten, self).__init__(name=name, trainable=trainable, **kwargs)

    def build(self, input_shape):
        self._hp_output_size = np.prod(input_shape["hp"][1:])
        self._lp_output_size = np.prod(input_shape["lp"][1:])
        super(Flatten, self).build(input_shape)

    def call(self, inputs, **kwargs):

        hp_input, lp_input = unpackGTCDict(inputs, error_if_not_gtc_dict=True)
        hp_output = K.reshape(hp_input, [-1, self._hp_output_size])
        lp_output = K.reshape(lp_input, [-1, self._lp_output_size])

        return GTCDict(hp=hp_output, lp=lp_output)




# Pure Keras version of flatten (also uses reshape, like the GTC version)
# For using in building single-precision keras models from gtcModels
# keras.layers.Flatten() throws an error for inputs that are already flattened...?
# Use this replacement instead which uses reshape

class keras_Flatten(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(keras_Flatten, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self._output_size = np.prod(input_shape[1:])
        super(keras_Flatten, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self._output_size)
        return output_shape

    def call(self, x, **kwargs):
        y = K.reshape(x, [-1, self._output_size])
        return y



