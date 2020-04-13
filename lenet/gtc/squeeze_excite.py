import tensorflow as tf
from tensorflow.python.ops import init_ops
from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict
from collections import OrderedDict
import keras.layers
import keras.backend as K


class SqueezeExcite(GTCLayer):
    """
    Implements a squeeze & excite block
    Squeeze and Excitation Networks: https://arxiv.org/abs/1709.01507
    """

    def __init__(self,
                 squeeze_factor=None,
                 filters_squeeze=None,
                 divisible_by=8,
                 inner_activation_fn='relu',
                 gating_fn='sigmoid',
                 quant_dict=None,
                 name=None,
                 use_bias=True,
                 independent_lp_weights=False,
                 **kwargs):


        if 'input_shape' in kwargs and kwargs['input_shape'] is None:
            kwargs.pop('input_shape', None)   # (remove 'input_shape' argument if is None)

        assert (squeeze_factor is None) != (filters_squeeze is None), "Specify either squeeze_factor, or filters_squeeze, but not both"

        if quant_dict is None:
            quant_dict = {'conv': None, 'activation': None}

        self._squeeze_factor = squeeze_factor
        self._filters_squeeze = filters_squeeze
        self._divisible_by = divisible_by
        self._inner_activation_fn = inner_activation_fn
        self._gating_fn = gating_fn
        self._quantizer_weights = quant_dict['conv']
        self._quantizer_activation = quant_dict['activation']
        self._independent_lp_weights = independent_lp_weights
        self._use_bias = use_bias

        super(SqueezeExcite, self).__init__(
            quantizer=self._quantizer_weights, name=name, **kwargs)

    def build(self, input_shape):
        # print('is in conv2d', input_shape)
        if isinstance(input_shape, dict) and ("gtc_tag" in input_shape):
            input_shape = input_shape['hp']
        input_shape_tuple = tuple(input_shape.as_list())
        channels = int(input_shape[-1])

        if self._squeeze_factor is not None:
            squeeze_channels = channels / self._squeeze_factor
        else:
            squeeze_channels = self._filters_squeeze

        squeeze_channels = _make_divisible(squeeze_channels, divisor=self._divisible_by)

        #channel_axis = -1 # if K.image_data_format() == "channels_last" else 1
        #filters = input_shape_tuple[channel_axis]
        self._se_shape = (1, 1, channels)
        self._se_shape_squeeze = (1, 1, squeeze_channels)
        input_shape1 = (None,) + self._se_shape
        input_shape2 = (None,) + self._se_shape_squeeze

        self._hp_kernels = []
        self._lp_kernels = []
        self._hp_biases = []
        self._lp_biases = []
        self._activations = []

        self._GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D()
        self._Reshape = keras.layers.Reshape(self._se_shape)
        with tf.name_scope(self.name + '/Conv/'):
            self._Dense_1_hp = keras.layers.Dense(squeeze_channels, activation=self._inner_activation_fn,
                                                  kernel_initializer='he_normal', use_bias=self._use_bias, name=self.name + '/Conv')
            self._Dense_1_hp.build(input_shape1)

        with tf.name_scope(self.name + '/Conv_1/'):
            self._Dense_2_hp = keras.layers.Dense(channels, activation=self._gating_fn, kernel_initializer='he_normal',
                                              use_bias=self._use_bias, name=self.name + '/Conv_1')

            self._Dense_2_hp.build(input_shape2)

        hp_bias_1, hp_bias_2, lp_bias_1, lp_bias_2 = None, None, None, None
        hp_kernel_1 = self._Dense_1_hp.kernel
        hp_kernel_2 = self._Dense_2_hp.kernel
        if self._use_bias:
            hp_bias_1 = self._Dense_1_hp.bias
            hp_bias_2 = self._Dense_2_hp.bias

        #TODO: decide which weights get called for hp_weights() and lp_weights()
        if self._independent_lp_weights:
            self._Dense_1_lp = keras.layers.Dense(channels // self._squeeze_factor, kernel_initializer='he_normal', use_bias=False)
            self._Dense_2_lp = keras.layers.Dense(channels, activation=self, kernel_initializer='he_normal', use_bias=False)
            self._Dense_1_hp.build(input_shape1)
            self._Dense_2_hp.build(input_shape2)
            lp_kernel_1 = self._quantizer_weights.quantize(self._Dense_1_lp.kernel, name="lp_kernel_1")
            lp_kernel_2 = self._quantizer_weights.quantize(self._Dense_1_lp.kernel, name="lp_kernel_2")
            if self._use_bias:
                lp_bias_1 = self._Dense_1_lp.bias
                lp_bias_2 = self._Dense_2_lp.bias

        else:

            lp_kernel_1 = self._quantizer_weights.quantize(hp_kernel_1, name="lp_kernel_1")
            lp_kernel_2 = self._quantizer_weights.quantize(hp_kernel_2, name="lp_kernel_2")
            if self._use_bias:
                lp_bias_1 = self._quantizer_weights.quantize(hp_bias_1, name="lp_bias_1")
                lp_bias_2 = self._quantizer_weights.quantize(hp_bias_2, name="lp_bias_2")


        self._hp_kernels.append(hp_kernel_1)
        self._hp_kernels.append(hp_kernel_2)
        self._hp_biases.append(hp_bias_1)
        self._hp_biases.append(hp_bias_2)

        self._lp_kernels.append(lp_kernel_1)
        self._lp_kernels.append(lp_kernel_2)
        self._lp_biases.append(lp_bias_1)
        self._lp_biases.append(lp_bias_2)

        self._activations.append(keras.layers.Activation(self._inner_activation_fn))
        self._activations.append(keras.layers.Activation(self._gating_fn))

        super(SqueezeExcite, self).build(input_shape)

    def call(self, inputs, **kwargs):
        hp_input, lp_input = unpackGTCDict(inputs)

        hp_output = self._squeeze_excite(hp_input, quantized=False)
        lp_output = self._squeeze_excite(lp_input, quantized=True)

        return GTCDict(hp=hp_output, lp=lp_output)


    def _squeeze_excite(self, input, quantized=False):

        se = self._GlobalAveragePooling2D(input)
        if quantized:
            se = self._quantizer_activation.quantize(se)

        se = self._Reshape(se)
        se = self._dense(se, id=0, quantized=quantized )
        se = self._dense(se, id=1, quantized=quantized )

        output = keras.layers.multiply([input, se])

        if quantized:
            output = self._quantizer_activation.quantize(output)

        return output


    def _dense(self, inputs, id, quantized=False):

        if quantized:
            kernel = self._lp_kernels[id]
            bias = self._lp_biases[id]
        else:
            kernel = self._hp_kernels[id]
            bias = self._hp_biases[id]

        dense_output = K.dot(inputs, kernel)

        if self._use_bias:
            dense_output = K.bias_add(dense_output, bias)

        if self._activations[id] is not None:
            dense_output = self._activations[id](dense_output)

        if quantized:
            dense_output = self._quantizer_activation.quantize(dense_output)

        return dense_output



    def _save_params(self):
        return [self.quantized_weights, self.quantized_bias]

    def quantization_error_for_output(self, some_tensor):
        return None

    def quantization_angle_difference_for_output(self, some_tensor):
        return None


    def get_hp_weights(self):
        if self._use_bias:
            return self._hp_kernels + self._hp_biases
        else:
            return self._hp_kernels

    def get_lp_weights(self):
        if self._use_bias:
            return self._lp_kernels + self._lp_biases
        else:
            return self._lp_kernels





def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v