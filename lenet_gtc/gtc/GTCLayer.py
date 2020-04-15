# this file is in the core branch
import tensorflow.compat.v1 as tf
#from tensorflow.python.keras.layers import InputSpec
from tensorflow.keras.layers import InputSpec
import quantization as quant
from collections import OrderedDict
import tensorflow.keras as keras

#class GTCLayer(tensorflow.keras.layers.Layer):
class GTCLayer(keras.layers.Layer):
    """
    Abstract layer for GTC layers
    """

    def __init__(self, trainable=True, quantizer=None, name=None, **kwargs):
        """TODO: Docstring for __init__.

        :tensor_op: TODO
        :quantizer: TODO
        add initializer and any params of add_weight that you need?
        """
        if quantizer is None:
            quantizer = quant.IdentityQuantization(name='I')
        elif isinstance(quantizer, str):
            quantizer = quant.from_str(quantizer)
        elif isinstance(quantizer, quant.Quantization):
            pass
        else:
            raise ValueError('Invalid input for quantizer : %s' % str(quantizer))

        self._quantizer = quantizer

        # self.input_spec = [
        #     InputSpec(dtype=tf.float32),
        #     InputSpec(dtype=tf.float32),
        #     InputSpec(dtype=tf.constant)
        # ]

        # Allow inputs to this layer to be specified during layer creation,
        # either as keyword argument 'layer_input' or 'layer_inputs'
        self._layer_inputs = kwargs.pop('layer_inputs', None)
        if self._layer_inputs is None:
            self._layer_inputs = kwargs.pop('layer_input', None)
        if isinstance(self._layer_inputs, str):
            self._layer_inputs = [self._layer_inputs]

        # Allow outputs from this layer to be specified during layer creation,
        # either as keyword argument 'layer_output' or 'layer_outputs'
        self._layer_outputs = kwargs.pop('layer_outputs', None)
        if self._layer_outputs is None:
            self._layer_outputs = kwargs.pop('layer_output', None)
        if isinstance(self._layer_outputs, str):
            self._layer_outputs = [self._layer_outputs]

        # filter out any remaining kwargs that are not relevant for the super-class, keras.layers.Layer
        allowed_kwargs = {'input_shape', 'batch_input_shape','batch_size','weights','activity_regularizer'}
        kwargs_allowed = {k:v for k,v in kwargs.items() if k in allowed_kwargs}
        kwargs_ignored = [k for k in kwargs.keys() if k not in allowed_kwargs]
        if bool(kwargs_ignored):
            print('Warning: GTCLayer : ignoring these keyword args: %s' % ','.join(kwargs_ignored))

        if not hasattr(self, '_layer_config'):
            self._layer_config = {}

        super(GTCLayer, self).__init__(
            name=name, trainable=trainable, **kwargs_allowed)

    def build(self, input_shape):
        super(GTCLayer, self).build(input_shape)


    def compute_output_shape(self, input_shape):
        # by default, assume output_shape is same as input shape
        if isinstance(input_shape, dict) and ("gtc_tag" in input_shape):
            input_shape = input_shape['hp']
        return input_shape

    # Put default/blank implementations of these methods so they don't have to be
    # defined for subclasses if all they return is None

    def quantization_error_for_output(self, some_tensor):
        return None

    def quantization_angle_difference_for_output(self, some_tensor):
        return None

    def number_of_bits(self, some_tensor):
        if self.get_hp_weights():
            return self._quantizer.number_of_bits(self.get_hp_weights())
        return None

    def quantization_error_for_weights(self):
        if self.get_hp_weights():
            return tuple(self._quantizer.quantization_error(w) for w in self.get_hp_weights())
        return None

    def quantization_angle_difference_for_weights(self):
        if self.get_hp_weights():
            return tuple(self._quantizer.quantization_angle_difference(w) for w in self.get_hp_weights())
        return None

    def get_hp_weights(self):
        return []

    def get_lp_weights(self):
        return []


    def get_hp_trainable_weights(self):  # by default, all hp weights are trainable
        if self.trainable:
            return self.get_hp_weights()     # (except for batchnorm moving mean/var)
        else:
            return []

    def get_hp_non_trainable_weights(self):
        if self.trainable:
            return []
        else:
            return self.get_hp_weights()

    def get_lp_trainable_weights(self):  # by default, all lp-weights are NON-trainable
        return []               # (since they are quantized versions of the hp-weights)

    def get_lp_non_trainable_weights(self):
        return self.get_lp_weights()


    def get_trainable_weights(self):
        return self.get_hp_trainable_weights() + self.get_lp_trainable_weights()

    def get_non_trainable_weights(self):
        return self.get_hp_non_trainable_weights() + self.get_lp_non_trainable_weights()


    def get_losses(self): # activity & weight-regularization losses
        return []

    def get_config(self):
        config = dict(quantizer=str(self._quantizer),
                      layer_inputs=self._layer_inputs,
                      #layer_outputs=self._layer_outputs,
                      )
        # The parent class (tf.keras.layers.Layer) config contains the:
        #  'trainable', 'name', 'dtype', '_batch_input_shape'  parameters
        config.update(super(GTCLayer, self).get_config())

        layer_config = self._layer_config.copy()
        config.update(layer_config)
        config.pop('dtype', None)

        return config


    @classmethod
    def remove_nones(cls, lst):
        return [x for x in lst if x is not None]



def GTCDict(hp, lp):
    d = OrderedDict()
    d['gtc_tag'] = tf.constant("gtc")
    d['hp'] = hp
    d['lp'] = lp
    return d

def unpackGTCDict(gtc_dict, error_if_not_gtc_dict=False):
    if isinstance(gtc_dict, dict) and ("gtc_tag" in gtc_dict):
        hp_input = gtc_dict["hp"]
        lp_input = gtc_dict["lp"]
    else:
        if error_if_not_gtc_dict:
            raise ValueError('Not a gtc layer, use tf version')
        else:
            hp_input = gtc_dict
            lp_input = gtc_dict #FIXME should we make the copy?

    return hp_input, lp_input
