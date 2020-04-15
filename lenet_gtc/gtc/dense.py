import tensorflow.compat.v1 as tf
from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict
from collections import OrderedDict
from tensorflow.python.ops import init_ops
import tensorflow.keras.layers
import tensorflow.keras.backend as K
import tensorflow.keras as keras

class Dense(GTCLayer):
    """
    Implements fully connected layer for GTC
    """

    def __init__(self,
                 units,
                 quantizer=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):

        self._units = units
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activity_regularizer = activity_regularizer
        self._kernel_constraint = kernel_constraint
        self._bias_constraint = bias_constraint
        self._trainable = trainable
        self._fused = False  # set to True if/when is merged with following bn_layer

        dense_args = dict(units=units,
                          activation=activation,
                          use_bias=use_bias,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          activity_regularizer=activity_regularizer,
                          kernel_constraint=kernel_constraint,
                          bias_constraint=bias_constraint)

        self._dense_layer = keras.layers.Dense(**dense_args,
                                               name='hp_weights')

        self._layer_config = dense_args

        #if not self._use_bias:
        #    self._bias_initializer = tf.initializers.zeros()

        trainable = kwargs.pop('trainable', trainable)
        super(Dense, self).__init__(
            trainable=trainable, quantizer=quantizer, name=name, **kwargs)

    def build(self, input_shape):

        if isinstance(input_shape, dict) and ("gtc_tag" in input_shape):
            input_shape = input_shape['hp']
        input_shape_tuple = tuple(input_shape.as_list())

        self.filter_shape = (input_shape[-1], self._units)

        self._dense_layer.build(input_shape_tuple)

        self.hp_weights = self._dense_layer.kernel

        self.lp_weights = self._quantizer.quantize(self.hp_weights, name="lp_weights")

        if self._use_bias:
            self.hp_bias = self._dense_layer.bias
            self.lp_bias = self._quantizer.quantize(self.hp_bias, name="lp_bias")
        else:
            self.hp_bias, self.lp_bias = None, None

    def call(self, inputs, **kwargs):

        hp_input, lp_input = unpackGTCDict(inputs)

        hp_output = self._dense(hp_input, quantized=False)
        lp_output = self._dense(lp_input, quantized=True)

        return GTCDict(hp=hp_output, lp=lp_output)

    def _dense(self, inputs, quantized=False):

        if quantized:
            weights = self.lp_weights
            bias = self.lp_bias
        else:
            weights = self.hp_weights
            bias = self.hp_bias

        dense_output = K.dot(inputs, weights)

        if self._use_bias:
            dense_output = K.bias_add(dense_output, bias)

        if self._activation is not None:
            dense_output = self._activation(dense_output)

        return dense_output


    def get_hp_weights(self):
        return self.remove_nones([self.hp_weights, self.hp_bias])

    def get_lp_weights(self):
        return self.remove_nones([self.lp_weights, self.lp_bias])

    def get_losses(self):
        return self._dense_layer._losses


