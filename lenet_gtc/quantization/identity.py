from quantization.quant import Quantization
import tensorflow.compat.v1 as tf

class IdentityQuantization(Quantization):
    """Trivial quantization --- used to implement tf layers in gtc"""
    def __init__(self, output_dtype=tf.float32, train=True, variable_scope='', name=None):
        super(IdentityQuantization, self).__init__(
            output_dtype, train, variable_scope, name, quant_type='identity')

    @Quantization.scoped_output('quantized_tensor')
    def quantize(self, some_tensor, epsilon=1e-6, name=None):
        return tf.cast(some_tensor, dtype=self._output_dtype, name=name)

    def trainable_parameters(self):
        return []

    @Quantization.scoped_output('encoding_number_of_bits')
    def number_of_bits(self, some_tensor, epsilon=1e-6):
        return self._bitmap[self._output_dtype]

