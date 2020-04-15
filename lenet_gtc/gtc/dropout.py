import tensorflow.compat.v1 as tf
from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict
from collections import OrderedDict
from tensorflow.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import nn_ops
import math
import tensorflow.keras as keras

if tf.__version__ == '1.12.0':
    use_keep_prob = True
else:
    use_keep_prob = False

class Dropout(GTCLayer):
    """
    Implements Dropout for GTCLayer
    """

    def __init__(self,
                 rate,
                 quantizer=None,
                 noise_shape=None,
                 seed=None,  # use this random seed for all dropout calls
                 name=None,  # (results in the same dropout mask being used each time). For debugging only.
                 **kwargs):

        self._rate = rate
        self._noise_shape = noise_shape
        self._seed = seed
        self.support_masking = True  # used for recurrent nets

        self._layer_config = dict(rate=rate,
                                  noise_shape=noise_shape,
                                  seed=seed)

        trainable = kwargs.pop('trainable', False)
        super(Dropout, self).__init__(
            trainable=trainable, quantizer=quantizer, name=name, **kwargs)

    def _get_noise_shape(self, inputs):
        if self._noise_shape is None:
            return self._noise_shape

        return nn_ops._get_noise_shape(
            inputs, self._noise_shape)  # TODO: get  more details

    def call(self, inputs, training=None):
        # original_training_value = training

        if training is None:
            training = K.learning_phase()

        hp_input, lp_input = unpackGTCDict(inputs)

        dropout_mask = self._get_dropout_mask(hp_input, training)

        hp_output = self._dropout(training, hp_input, dropout_mask)
        lp_output = self._dropout(training, lp_input, dropout_mask)

        return GTCDict(hp=hp_output, lp=lp_output)

    def _get_dropout_mask(self, hp_input, training):

        all_true = tf.cast(K.ones_like(hp_input), tf.bool)
        if use_keep_prob:
            rate_kwargs = dict(keep_prob=1-self._rate)
        else:
            rate_kwargs = dict(rate=self._rate)

        def bool_mask():
            masked_ones= tf.nn.dropout(
                K.ones_like(hp_input),
                **rate_kwargs,
                noise_shape=self._get_noise_shape(hp_input),
                seed=self._seed)
            return tf.cast(masked_ones, tf.bool)

        dropout_mask = tf_utils.smart_cond(
            training,
            bool_mask,          # during training
            lambda: all_true)   # during inference
        return dropout_mask

    def _dropout(self, training, inputs, dropout_mask):

        output = tf_utils.smart_cond(
            training,
            # during training: mask inputs, then multiply by 1/rate to compensate
            lambda: tf.where(dropout_mask, inputs, tf.zeros_like(inputs)) * (1/self._rate),
            lambda: array_ops.identity(inputs))  # during inference: just return original inputs

        return output

