from quantization.quant import Quantization
import tensorflow as tf

_default_intercept_bnd_int=0.0001  # default initialization value
_default_scale_bnd_int=0.1         # default initialization value


class BoundedIntegerQuantization(Quantization):
    """ x = clip(0, round(scale*x + intercept), 2^(beta-1))
    beta is based on the maximum number of bits allowed
    scale is the scaling parameter
    intercept is the intercept
    This is some sense bounds the relu to limited range
    beta = 8, meaning 8 bit Integer allowed
    """

    def __init__(self, train=True, scale=_default_scale_bnd_int,
                 intercept=_default_intercept_bnd_int, name=None):
        quant_type_name = type(self).__name__
        if bool(name):  # append the name of the quantizer, if not empty
            quant_type_name += '/' + name

        with tf.variable_scope(quant_type_name):
            self.scale = tf.Variable(initial_value=float(scale), name="scale")
            self.intercept = tf.Variable(initial_value=float(intercept), name="intercept")
            super().__init__(train, name=name, quant_type='boundedinteger')

    def quantize(self, weights, name=None):
        pass

    def __str__(self):
        return 'BoundedIntegerQuantization'
       