from quantization.quant import Quantization
import tensorflow.compat.v1 as tf

default_intercept = 0.0001
default_slope = 1.0

class GTCQuantization(Quantization):
    """GTCQuantization
    Implements Generalized Ternary quantization as in Parajuli et. al. 2018"""

    def __init__(self, slope=default_slope, intercept=default_intercept,
                 output_dtype=tf.float32, train=True, variable_scope='',
                 epsilon=1e-6, name=None):
        """__init__
        Creates two tf.Variable scalars for quantization function
        :raises AssertionError if epsilon not at least 1e-6
        """
        # We know 1e-6 works, anything larger must be fine
        assert epsilon >= 1e-6
        with tf.compat.v1.variable_scope(variable_scope):
            quant_type_name = type(self).__name__
            if bool(name):  # append the name of the quantizer, if not empty
                quant_type_name += '/' + name

            with tf.compat.v1.variable_scope(quant_type_name):
                self.q_intercept = tf.Variable(
                    initial_value=intercept, name="intercept")
                self.q_slope = tf.Variable(
                    initial_value=slope, name="slope")
        super().__init__(output_dtype, train, variable_scope, epsilon=epsilon,
                         name=name, quant_type='gtc')

    @Quantization.scoped_output('exponent_term_of_two')
    def exp_term(self, some_tensor):
        inner_term = self.q_intercept + \
                     self.q_slope * self.log2(some_tensor)
        return inner_term

    @Quantization.scoped_output('rounded_exponent_term_of_two')
    def rounded_exp_term(self, some_tensor):
        """rounded_exp
        Equation (6)
        :param some_tensor: tf.Variable
        """
        inner_term = self.exp_term(some_tensor)
        rounded_result = self.differentiable_round(inner_term)
        return rounded_result

    @Quantization.scoped_output('quantized_tensor')
    def quantize(self, some_tensor,  name=None):
        """quantize
        Equation (5)
        """
        exp_term = self.rounded_exp_term(some_tensor)
        quantized_tensor = self.sign_with_zero(some_tensor) * (2 ** exp_term)
        self.quantized_some_tensor = tf.cast(quantized_tensor,
                                             dtype=self._output_dtype, name=name)
        return self.quantized_some_tensor

    @Quantization.scoped_output('log2_of_abs')
    def log2(self, some_tensor):
        """log2
        Log base 2 of elementwise absolute value of tensor
        implemented using tf.log
        Computes \\(log_2(|some_tensor|+epsilon)\\)
        :param some_tensor: tf.Variable
        """
        # We know 1e-6 works, anything larger must be fine
        numerator = tf.compat.v1.log(tf.abs(some_tensor) + self._epsilon)
        denominator = tf.compat.v1.log(tf.compat.v1.constant(2. + self._epsilon, dtype=numerator.dtype))
        return numerator / denominator

    @Quantization.scoped_output('sign_with_zero')
    def sign_with_zero(self, some_tensor):
        """sign_with_zero
        Calculate the piece-wise constant sign function that is zero for
        values in (-epsilon, +epsilon), otherwise returns the sign
        :param some_tensor: tf.Variable
        """
        pos_mask = tf.where(tf.greater_equal(some_tensor, self._epsilon),
                            tf.ones_like(some_tensor),
                            tf.zeros_like(some_tensor))
        neg_mask = tf.where(tf.less_equal(some_tensor, -1. * self._epsilon),
                            -1. * tf.ones_like(some_tensor),
                            tf.zeros_like(some_tensor))
        return neg_mask + pos_mask

    def trainable_parameters(self):
        return [self.q_intercept, self.q_slope]

    @Quantization.scoped_output('bins')
    def bins(self, some_tensor):
        exp_term = self.rounded_exp_term(some_tensor)
        quantized_tensor = self.sign_with_zero(some_tensor) * (2 ** exp_term)
        uniqs, _ = tf.unique(tf.reshape(quantized_tensor, [-1]))
        return uniqs

    # NOTE output is due to the range of the set of exponents
    # how to interpret this? Sign bit is always included
    # So this discourages large range, encourages compact set of exponents
    @Quantization.scoped_output('encoding_number_of_bits')
    def number_of_bits(self, some_tensor):
        """number_of_bits
        Equation (10)
        :param some_tensor: tf.Variable
        :returns tf.Variable
        """
        # [collect_exponents for w in list_tensors]
        # [take_min_max for w in list_tensors]
        # take min of min and max of max over list

        tensor0 = some_tensor[0] if isinstance(some_tensor, list) else some_tensor
        assert isinstance(tensor0, (tf.Variable, tf.Tensor))
        assert not 'Quantization' in tensor0.name, "Pass in HP tensor, not quantized tensor"
        # Make sure we are not receiving a quantized tensor
        some_tensor = some_tensor if isinstance(some_tensor,
                                                list) else [some_tensor]
        exp_term = [
            self.rounded_exp_term(tensor_var)
            for tensor_var in some_tensor
        ]
        #print('exp term', exp_term)
        ## TODO: Check with aswin, there are bugs here!!!!!
        min_exponent = tf.reduce_min(
            [tf.reduce_min(tensor_var) for tensor_var in exp_term])
        max_exponent = tf.reduce_max(
            [tf.reduce_max(tensor_var) for tensor_var in exp_term])
        range_exponent = max_exponent - min_exponent + 1
        log_range_exponent = self.log2(range_exponent)
        nbits = 1 + self.differentiable_ceil(log_range_exponent)
        return tf.minimum(nbits, self._bitmap[self._output_dtype])
