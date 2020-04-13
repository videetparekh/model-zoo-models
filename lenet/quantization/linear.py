from quantization.quant import Quantization
import tensorflow as tf

num_bits_initial_default = 8.0

class LinearQuantization(Quantization):
    """LinearQuantization with variable bits
    Implements Bitnet linear quantization as in Raghavan et. al. 2017"""

    def __init__(self, num_bits=num_bits_initial_default, output_dtype=tf.float32,
                 train=True, variable_scope='', epsilon=1e-6, name=None):
        """__init__
        Creates one tf.Variable scalar for quantization function
        """
        if num_bits is None:
            num_bits = num_bits_initial_default
        with tf.variable_scope(variable_scope):
            quant_type_name = type(self).__name__
            if bool(name):  # append the name of the quantizer, if not empty
                quant_type_name += '/' + name

            with tf.variable_scope(quant_type_name):
                if train:
                    self.num_bits = tf.Variable(initial_value=float(num_bits), name="slope",
                                                constraint=lambda t: tf.clip_by_value(t, 1, 32.))
                else:    
                    self.num_bits = tf.constant(float(num_bits), name="slope")
        super().__init__(output_dtype, train, variable_scope, name,
                         epsilon=epsilon, quant_type='linear')

    @Quantization.scoped_output('alpha_min_tensor')
    def alpha(self, some_tensor):
        return tf.reduce_min(some_tensor)

    @Quantization.scoped_output('beta_max_tensor')
    def beta(self, some_tensor):
        return tf.reduce_max(some_tensor)

    @Quantization.scoped_output('delta_separation_of_bins')
    def delta(self, some_tensor):
        alpha = self.alpha(some_tensor)
        beta = self.beta(some_tensor)
        return tf.where(tf.equal(alpha,beta),
                        1. ,
                        (beta - alpha) / (1. * 2**self.differentiable_round(self.num_bits)))

    @Quantization.scoped_output('bins')
    def bins(self, some_tensor):
        alpha = self.alpha(some_tensor)
        bin_idx = tf.range(start=0., limit=1.+ 2.**self.differentiable_round(self.num_bits))
        delta = self.delta(some_tensor)
        ret = alpha + bin_idx * delta
        return ret

    @Quantization.scoped_output('quantized_tensor')
    def quantize(self, some_tensor, name=None):
        """quantize
        Equation (1)
        """
        bin_idx = self.rounded_term(some_tensor)
        alpha = self.alpha(some_tensor)
        delta = self.delta(some_tensor)
        return tf.cast(alpha + delta * bin_idx, dtype=self._output_dtype, name=name)

    @Quantization.scoped_output('rounded_bin_idx')
    def rounded_term(self, some_tensor):
        """Eq (1)"""
        alpha = self.alpha(some_tensor)
        delta = self.delta(some_tensor)
        return self.differentiable_round((some_tensor - alpha) / (1. * delta))

    def trainable_parameters(self):
        return [self.num_bits]

    @Quantization.scoped_output('encoding_number_of_bits')
    def number_of_bits(self, some_tensor):
        return tf.minimum(self.num_bits, self._bitmap[self._output_dtype])

