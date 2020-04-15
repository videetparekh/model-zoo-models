import tensorflow.compat.v1 as tf
from tensorflow.python.ops import init_ops
from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict
import quantization as quant
from collections import OrderedDict
import tensorflow.keras.layers
import numpy as np
import tensorflow.keras as keras
useZeroBiasIfUseBiasFalse = False   # some conv layers skip the bias term if followed by a batchnorm layer.

class Conv2d(GTCLayer):
    """
    Implements conv2d GTC layer
    """

    def __init__(self,
                 filters,               # use filters='depthwise' for depthwise separable conv2d
                 kernel_size,
                 quantizer=None,
                 bias_quantizer=None,   # by default(None), use the same quantizer for the bias as for the kernel
                 strides=1,             # keras default: (1,1)
                 padding='SAME',        # keras default: 'valid'
                 data_format='channels_last',  # keras default: None
                 dilation_rate=1,       # keras default: (1,1)
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', #tf.initializers.glorot_normal(),  # keras default: 'zeros'
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):

        if isinstance(dilation_rate, int):
            dilation_rate = (dilation_rate, dilation_rate)
        padding = padding.upper()

        assert not isinstance(filters, list), 'Filters must be an integer'
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._data_format = data_format
        self._dilation_rate = dilation_rate
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._kernel_constraint = kernel_constraint
        self._bias_constraint = bias_constraint
        self._trainable = trainable
        self._fused = False   # set to True if/when is merged with following bn_layer
        self._in_channels = None
        self._out_channels = None
        self._kwargs = kwargs.copy()
        if not self._use_bias:
            self._bias_initializer = tf.initializers.zeros()

        self._do_depthwise_conv = filters=='depthwise'

        # handle bias initialized to specific numpy array
        # For the config.json file, we save the raw numpy array.
        # But when initializing the conv layer, we replace it with
        # a function that returns a keras.constant()
        bias_initializer_for_config = bias_initializer
        
        if isinstance(bias_initializer, (list, np.ndarray)):
            bias_array = np.array(bias_initializer.copy())
            def numpy_init(shape, dtype=None):
                assert bias_array.shape == shape, "shape mismatch"
                return keras.backend.constant(bias_array, dtype=dtype)
            bias_initializer = numpy_init

        conv_args_base = dict(kernel_size=kernel_size, strides=strides,
                         padding=padding, data_format=data_format,
                         dilation_rate=dilation_rate, activation=activation,
                         use_bias=use_bias,
                         #kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         #kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint)

        # Arguments for a standard conv layer, and also for saving to config
        conv_layer_args = conv_args_base.copy()
        conv_layer_args['filters'] = filters
        conv_layer_args['kernel_initializer'] = kernel_initializer
        conv_layer_args['kernel_regularizer'] = kernel_regularizer
        conv_layer_args['dilation_rate'] = dilation_rate

        if not self._do_depthwise_conv:
            self._conv_layer = keras.layers.Conv2D(**conv_layer_args)
            self._conv_op = tf.nn.conv2d

        elif self._do_depthwise_conv:
            dwconv_layer_args = conv_args_base.copy()
            dwconv_layer_args['depthwise_initializer']=kernel_initializer
            dwconv_layer_args['depthwise_regularizer']=kernel_regularizer
            self._conv_layer = keras.layers.DepthwiseConv2D(**dwconv_layer_args)
            self._conv_op = tf.nn.depthwise_conv2d

        self._strides = self._conv_layer.strides

        if tuple(dilation_rate) == (1, 1):
            dilation_kwargs = dict()
        else:
            dilations_ext = [1, self._dilation_rate[0], self._dilation_rate[1], 1]
            dilation_kwargs = dict(dilations=dilations_ext)
        self._dilation_kwargs = dilation_kwargs

        if 'input_shape' in kwargs and kwargs['input_shape'] is None:
            kwargs.pop('input_shape', None)   # (remove 'input_shape' argument if is None)

        super(Conv2d, self).__init__(
            trainable=trainable, quantizer=quantizer, name=name, **kwargs)

        conv_layer_args['bias_initializer'] = bias_initializer_for_config
        self._layer_config = conv_layer_args.copy()
        self._layer_config['bias_quantizer'] = bias_quantizer

        # The parent-class initialization handles string/None quantizer input
        # Once that is done, we deal with the bias quantizer.
        if bias_quantizer is None:
            self._bias_quantizer = self._quantizer
            self._bias_quantizer_type = 'same'  # same as weights quantizer
        else:
            if isinstance(bias_quantizer, str):
                self._bias_quantizer = quant.GetQuantization(
                    bias_quantizer)(name=bias_quantizer.upper()[0])
            else:
                assert isinstance(bias_quantizer, quant.Quantization)
                self._bias_quantizer = bias_quantizer
            self._bias_quantizer_type = 'independent' # independent from weights quantizer


    def build(self, input_shape):

        if isinstance(input_shape, dict) and ("gtc_tag" in input_shape):
            input_shape = input_shape['hp']
        input_shape_tuple = tuple(input_shape.as_list())

        channels = int(input_shape[-1])
        self._in_channels = channels # at build time, see how many input channels
        if self._do_depthwise_conv:
            self._out_channels = self._in_channels # at build time, see how many input channels
        else:
            self._out_channels = self._filters  # at build time, see how many input channels

        self._conv_layer.build(input_shape_tuple)
        if self._do_depthwise_conv:
            self.hp_weights = self._conv_layer.depthwise_kernel
        else:
            self.hp_weights = self._conv_layer.kernel


        self.lp_weights = self._quantizer.quantize(self.hp_weights, name="lp_weights")

        if self._use_bias:
            self.hp_bias = self._conv_layer.bias
            self.lp_bias = self._bias_quantizer.quantize(self.hp_bias, name="lp_bias")

        else:
            if useZeroBiasIfUseBiasFalse:

                self.hp_bias = self.add_weight(
                    name='hp_bias',
                    dtype=tf.float32,
                    shape=(self._out_channels, ),
                    initializer=self._bias_initializer,
                    regularizer=self._bias_regularizer,
                    constraint=self._bias_constraint,
                    trainable=False)
                self.lp_bias = self._bias_quantizer.quantize(self.hp_bias, name="lp_bias")

            else:
                self.hp_bias = None
                self.lp_bias = None

        super(Conv2d, self).build(input_shape)

    def call(self, inputs, **kwargs):
        hp_input, lp_input = unpackGTCDict(inputs)

        hp_output = self._conv2d(hp_input, quantized=False)
        lp_output = self._conv2d(lp_input, quantized=True)

        return GTCDict(hp=hp_output, lp=lp_output)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, dict) and ("gtc_tag" in input_shape):
            input_shape = input_shape['hp']

        test_tensor = tf.constant(input_shape)
        out_tensor = self._conv2d(test_tensor)
        return out_tensor.shape


    def _conv2d(self, inputs, quantized=False):

        if quantized:  # low-precision weights (which we hope will track the high-precision, fused weights)
            weights = self.lp_weights
            bias = self.lp_bias
        else:
            weights = self.hp_weights
            bias = self.hp_bias

        x = self._conv_op(
            inputs,
            filters=weights,
            strides=[1, self._strides[0], self._strides[1], 1],
            padding=self._padding,
            **self._dilation_kwargs)

        if self._use_bias or useZeroBiasIfUseBiasFalse:
            x = tf.nn.bias_add(x, bias)

        return x

    def _save_params(self):
        return [self.lp_weights, self.lp_bias]


    def get_hp_weights(self):
        return self.remove_nones([self.hp_weights, self.hp_bias])

    def get_lp_weights(self):
        return self.remove_nones([self.lp_weights, self.lp_bias])


    def get_losses(self):
        return self._conv_layer._losses







class Conv2d_depthwise(Conv2d):
    """
    Implements conv2d GTC layer
    """

    def __init__(self,
                 kernel_size,
                 quantizer=None,
                 strides=1,             # keras default: (1,1)
                 padding='SAME',        # keras default: 'valid'
                 data_format='channels_last',  # keras default: None
                 dilation_rate=1,       # keras default: (1,1)
                 activation=None,
                 use_bias=True,
                 kernel_initializer=tf.initializers.glorot_normal(),  # keras default: 'glorot_uniform'
                 bias_initializer=tf.initializers.glorot_normal(),  # keras default: 'zeros'
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):

        super(Conv2d_depthwise, self).__init__(
            filters='depthwise',
            kernel_size=kernel_size,
            quantizer=quantizer,
            strides=strides,  # keras default: (1,1)
            padding=padding,  # keras default: 'valid'
            data_format=data_format,  # keras default: None
            dilation_rate=dilation_rate,  # keras default: (1,1)
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,  # keras default: 'glorot_uniform'
            bias_initializer=bias_initializer,  # keras default: 'zeros'
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs)

