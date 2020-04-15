# this file is in the core branch
import tensorflow.compat.v1 as tf

# NOTE all classes in this file currently
# assume that the input tensors
# are NOT batched i.e. applicable for
# weight quantization and not for activations/inputs


class Quantization:
    """Abstract class for Quantization functions that work with gtc.layers"""
    def __init__(self, output_dtype=tf.float32,
                 train=True, variable_scope='', name=None,
                 quant_type=None, epsilon=1e-6):
        """
        :param output_dtype: cast quantized tensor to this dtype
        :param train: sets self._train which is used elsewhere
        :param variable_scope: namespace to use for quantization parameters
        Typically the layer name
        output_dtype one of tf.float16, tf.float32, tf.float64, or
        tf.int8, tf.int16, tf.int32, tf.int64"""
        self._train = train
        self._variable_scope = variable_scope
        self._output_dtype = output_dtype
        self._epsilon = epsilon
        self._bitmap = {
            tf.float32: 32,
            tf.float64: 64,
            tf.float16: 16,
            tf.int8: 8,
            tf.int16: 16,
            tf.int32: 32,
            tf.int64: 64,
            tf.uint8: 8,
            tf.uint16: 16,
            tf.uint32: 32,
            tf.uint64: 64
        }
        assert output_dtype in self._bitmap
        self.name = name
        self.quant_type = quant_type

    @classmethod
    def scoped_output(cls, some_name):
        """decorator for functions of the form
        f(self: Quantization, some_tensor, *args, **kwargs)
        to properly define the variable scope in tf
        called as @Quantization.scoped_output(some_name)
        names the output of the function as
        self._variable_scope/some_tensor.name/some_name
        """
        def name_wrapper(tf_function):
            def func_wrapper(self, some_tensor, *args, **kwargs):
                ret = tf_function(self, some_tensor, *args, **kwargs)
                with tf.compat.v1.variable_scope(self._variable_scope):
                    some_tensor0 = some_tensor[0] if isinstance(some_tensor, list) else some_tensor
                    with tf.compat.v1.variable_scope(some_tensor0.name.replace(':', '_')):
                        with tf.compat.v1.variable_scope(type(self).__name__):
                            ret = tf.identity(ret, name=some_name)
                            # print('adding', some_name, 'to',
                            #      tf.get_variable_scope().name, 'result', ret.name)
                            return ret

            return func_wrapper

        return name_wrapper

    def differentiable_round(self, some_tensor):
        """differentiable_round
        Wrapper around tf.Round whose gradient is the identity function
        :param some_tensor: tf.Variable
        """
        default_graph = tf.compat.v1.get_default_graph()
        with default_graph.gradient_override_map({"Round": "Identity"}):
            return tf.round(some_tensor)

    def differentiable_ceil(self, some_tensor):
        default_graph = tf.compat.v1.get_default_graph()
        with default_graph.gradient_override_map({"Ceil": "Identity"}):
            return tf.ceil(some_tensor)

    def quantize(self, some_tensor, name=None):
        """quantize
        The main function for quantization any tensor
        :param some_tensor: A tf.Variable
        :returns Quantized tf.Variable
        """
        raise NotImplementedError("Overriding classes should implement this.")

    def trainable_parameters(self):
        """trainable_parameters
        :returns List of tf.Variables for trainable quantization parameters"""
        raise NotImplementedError("Overriding classes should implement this.")

    def number_of_bits(self, some_tensor):
        """number_of_bits
        :returns tf.Variable for number of bits"""
        raise NotImplementedError("Overriding classes should implement this.")

    def quantization_error(self, some_tensor):
        quantized_tensor = self.quantize(some_tensor)

        def nice(x):
            return tf.reshape(tf.cast(x, dtype=tf.float32), [-1])

        diff = nice(some_tensor) - nice(quantized_tensor)
        return tf.norm(diff) / tf.cast(tf.size(diff), dtype=diff.dtype)

    def quantization_angle_difference(self, some_tensor):
        quantized_tensor = self.quantize(some_tensor)

        def nice(x):
            return tf.reshape(tf.cast(x, dtype=tf.float32), [-1])

        unit_tensor_1 = nice(some_tensor)
        unit_tensor_1_norm = tf.math.l2_normalize(unit_tensor_1)
        unit_tensor_2 = nice(quantized_tensor)
        unit_tensor_2_norm = tf.math.l2_normalize(unit_tensor_2)
        dot = tf.tensordot(unit_tensor_1_norm, unit_tensor_2_norm, 1)
        return dot / tf.cast(tf.size(dot), dtype=dot.dtype)

    def __str__(self):
        name_str = self.name if self.name is not None else ''
        return self.quant_type + '.' + name_str

