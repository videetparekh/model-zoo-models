from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict
import numpy as np
from keras import backend as K
import keras

class L2Normalization(GTCLayer):
    """
    implements flatten operation on the gtc layer output
    """

    def __init__(self, gamma_init=20, name=None,  **kwargs):
        """TODO: Docstring for __init__.
        :returns: Flattens the input tensor to vector
        Note: does not use keras.layers.Flatten layer, as this does not record information about the final output shape.
        """
        if K.image_data_format() == 'channels_last':  # K.image_dim_ordering() == 'tf'
            self.axis = 3
        else:
            self.axis = 1
        trainable = kwargs.pop('trainable', False)
        self.gamma_init = gamma_init
        self._layer_config = {'gamma_init': gamma_init}
        super(L2Normalization, self).__init__(name=name, trainable=trainable, **kwargs)

    def build(self, input_shape):

        if isinstance(input_shape, dict) and ("gtc_tag" in input_shape):
            input_shape = input_shape['hp']
        input_shape_tuple = tuple(input_shape.as_list())

        #channels = int(input_shape[-1])

        gamma = self.gamma_init * np.ones((input_shape_tuple[self.axis],))
        self.gamma = K.variable(gamma, name='gamma_weights'.format(self.name))

        super(L2Normalization, self).build(input_shape)

    def call(self, inputs, **kwargs):

        hp_input, lp_input = unpackGTCDict(inputs, error_if_not_gtc_dict=True)

        hp_output = K.l2_normalize(hp_input, self.axis) * self.gamma
        lp_output = K.l2_normalize(lp_input, self.axis) * self.gamma

        return GTCDict(hp=hp_output, lp=lp_output)

    def get_hp_weights(self):
        return [self.gamma]





class keras_L2Normalization(keras.layers.Layer):
    '''
    pure keras version of the L2_Normalization layer
    '''

    def __init__(self, gamma_init=20, **kwargs):
        if K.image_data_format() == 'channels_last':  # K.image_dim_ordering() == 'tf'
            self.axis = 3
        else:
            self.axis = 1
        self.gamma_init = gamma_init
        super(keras_L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [keras.layers.InputSpec(shape=input_shape)]
        gamma = self.gamma_init * np.ones((input_shape[self.axis],))
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]
        super(keras_L2Normalization, self).build(input_shape)

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        return output * self.gamma

    def get_config(self):
        config = {
            'gamma_init': self.gamma_init
        }
        base_config = super(keras_L2Normalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
