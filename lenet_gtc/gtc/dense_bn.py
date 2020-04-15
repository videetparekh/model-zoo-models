
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import init_ops
from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict
from collections import OrderedDict
import tensorflow.keras as keras
import tensorflow.keras.layers

import tensorflow.keras.backend as K 
class Dense_bn(GTCLayer):
    """
    Implements fused conv2d + batch norm GTC layer
    """

    def __init__(self,
                 units,
                 quantizer=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=tf.initializers.glorot_normal(),  # keras default: 'glorot_uniform'
                 bias_initializer=tf.initializers.glorot_normal(), # keras default: 'zeros'
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 activity_regularizer=None, 
                 name=None,

                 axis=-1,  # batch norm parameters
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 adjustment=None,  # TODO: check use

                 lp_conv_bn_fused = True, # default: True: hp conv and batchnorm weights are separate, lp is merged.
                                          #          False: hp conv and batchnorm weights are separate, lp is is also separate.
                                          #                (For convenince/sanity check: to mimic having two completely separate layers.)
                 check_fused_outputs=True,
                 **kwargs):

        self._units = units
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._kernel_constraint = kernel_constraint
        self._bias_constraint = bias_constraint
        self._trainable = trainable
        
        self._lp_conv_bn_fused = lp_conv_bn_fused
        self._check_fused_outputs = check_fused_outputs
        if not self._use_bias:
            self._bias_initializer = tf.initializers.zeros()

        assert center == True and scale == True, "not using center/scale is currently not supported"


        self._dense_layer_hp = keras.layers.Dense(units=self._units,
                                               activation=activation,
                                               use_bias=use_bias,
                                               kernel_initializer=kernel_initializer,
                                               bias_initializer=bias_initializer,
                                               kernel_regularizer=kernel_regularizer,
                                               bias_regularizer=bias_regularizer,
                                               kernel_constraint=kernel_constraint,
                                               bias_constraint=bias_constraint,
                                               name='hp_weights')

        self._dense_op = K.dot


        self._batch_normalizer_hp = keras.layers.normalization.BatchNormalization(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            name='hp_bnorm'
        )

        if not self._lp_conv_bn_fused:
            self._batch_normalizer_lp = keras.layers.normalization.BatchNormalization(
                axis=axis,
                momentum=momentum,
                epsilon=epsilon,
                center=center,
                scale=scale,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                moving_mean_initializer=moving_mean_initializer,
                moving_variance_initializer=moving_variance_initializer,
                beta_regularizer=beta_regularizer,
                gamma_regularizer=gamma_regularizer,
                beta_constraint=beta_constraint,
                gamma_constraint=gamma_constraint,
                name='lp_bnorm'
            )

        if 'input_shape' in kwargs and kwargs['input_shape'] is None:
            kwargs.pop('input_shape', None)   # (remove 'input_shape' argument if is None)

        super(Dense_bn, self).__init__(
            trainable=trainable, quantizer=quantizer, name=name, **kwargs)

    def build(self, input_shape):
        
        if isinstance(input_shape, dict) and ("gtc_tag" in input_shape):
            input_shape = input_shape['hp']


        # Step 1. build dense layer
        input_shape_tuple = tuple(input_shape.as_list())
        self._dense_layer_hp.dtype = tf.float32
        self.filter_shape = (input_shape[-1], self._units)

        self._dense_layer_hp.build(input_shape_tuple)


        self.hp_weights = self._dense_layer_hp.kernel


        self.hp_bias = None
        self.lp_bias = None
        self.hp_fused_bias = None

       #  nfilters = channels if self._do_depthwise_conv else self._filters
       # TODO - Check if the shape is correct or not 
        bn_input_shape_tuple = input_shape_tuple[:-1] + (self._units,)

        if self._use_bias:
            self.hp_bias = self._dense_layer_hp.bias


        # Step 2. build batch norm layer

        self._batch_normalizer_hp.build(bn_input_shape_tuple)
        if self._lp_conv_bn_fused:
            # Step 3:  Fused batch-norm weights and bias (adapted from https://tehnokv.com/posts/fusing-batchnorm-and-conv/)
            # 3a. Prepare filters
            bn = self._batch_normalizer_hp
            bn_weight = tf.divide(bn.gamma, tf.sqrt(bn.epsilon + bn.moving_variance)) # [c_out] -> [c_out, c_out]

            # TODO - Check the shape 
            conv_weight = tf.transpose( tf.reshape(self.hp_weights, [-1, self._units]) ) # [k , k , c_in , c_out] ->  [k*k*c_in, c_out]
            bn_weight_diag = tf.diag(bn_weight)#  [c_out] -> [c_out, c_out]
            self.hp_fused_weights = tf.reshape( tf.transpose( tf.matmul(bn_weight_diag, conv_weight)), self.hp_weights.shape)

            # 3b. Prepare spatial bias
            #b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
            self.hp_fused_bias = bn.beta - tf.multiply(bn_weight, bn.moving_mean)
            if self._use_bias:
                self.hp_fused_bias += tf.multiply( bn_weight, self.hp_bias)  # add to fused_bias if use_bias==True, otherwise stands alone. (assumes center == True)


            # Step 4: Now we can define the quantized weights/bias
            self.lp_fused_weights = self._quantizer.quantize(self.hp_fused_weights, name='lp_weights')
            self.lp_fused_bias = self._quantizer.quantize(self.hp_fused_bias, name='lp_bias')

            self.lp_bn_weight = None
            self.lp_bn_bias = None

            self.hp_bn_gamma = bn.gamma
            self.hp_bn_beta = bn.beta
            self.hp_bn_moving_mean = bn.moving_mean
            self.hp_bn_moving_variance = bn.moving_variance
            print('trainable vars', self.trainable_weights)
        else: # if NOT self._lp_conv_bn_fused

            self._batch_normalizer_lp.build(bn_input_shape_tuple)

            self._batch_normalizer_lp.gamma = self._quantizer.quantize(self._batch_normalizer_hp.gamma, name='bnorm_lp_gamma')
            self._batch_normalizer_lp.beta = self._quantizer.quantize(self._batch_normalizer_hp.beta, name='bnorm_lp_beta')

            self.lp_bn_weights = self._batch_normalizer_lp.gamma
            self.lp_bn_bias = self._batch_normalizer_lp.beta

            self.lp_weights = self._quantizer.quantize(self.hp_weights, name="lp_weights")
            if self._use_bias:
                self.lp_bias = self._quantizer.quantize(self.hp_bias, name="lp_bias")

        super(Dense_bn, self).build(input_shape_tuple)  # the actual input_shape value is not used.



    def call(self, inputs, fused=False, training=None, **kwargs):

        hp_input, lp_input = unpackGTCDict(inputs)

        if not fused:
            # Step 1:  Pass HP input through convolutional layer
            hp_output = self._dense(hp_input, quantized=False, fused=False)

            # Step 2:  Pass HP input through batch- norm layer
            hp_output_use = self._batch_normalizer_hp(hp_output, training=training)

        else:   # use fused weights (support for sanity check. Normally we keep the conv and bn separate until test time.
                # but can use this to confirm that the merging is done correctly)
            # Step 1+2: Use fused weights
            hp_output_use = self._dense(hp_input, quantized=False, fused=True)

            if self._check_fused_outputs:
                hp_output_conv = self._dense(hp_input, quantized=False, fused=False)
                hp_output_conv_bn = self._batch_normalizer_hp(hp_output_conv, training=training)
                fuse_mean_error = tf.reduce_mean( tf.abs( hp_output_conv_bn - hp_output_use ) )

                # fuse_mean_error = tf.Print(fuse_mean_error, [fuse_mean_error], 'fuse error : ')
                #hp_output_use += fuse_mean_error * 0
                print('Fuse mean error', fuse_mean_error)
                assert_op = tf.debugging.assert_less(fuse_mean_error, 1e-5)
                with tf.control_dependencies([assert_op]):
                    fuse_error_mean = tf.identity(fuse_mean_error, name='check')
                    hp_output_use += fuse_error_mean * 0

        # 3. Get LP-outputs (can only calculate this after quantized weights have been defined)

        #lp_output = self._conv2d(lp_input, quantized=True, fused=False)
        #lp_output = self._dense(lp_input, quantized=True, fused=self._lp_conv_bn_fused)

        # Step 3b:  Pass LP input through batch-norm layer, if not using fused weights.
        #if not self._lp_conv_bn_fused:
        #    lp_output = self._batch_normalizer_lp(lp_output, training=training)

        return GTCDict(hp=hp_output_use, lp=lp_input)


    def _dense(self, inputs, quantized=False, fused=False):

        if quantized:  # low-precision weights (which we hope will track the high-precision, fused weights)
            if self._lp_conv_bn_fused:
                weights = self.lp_fused_weights
                bias = self.lp_fused_bias
            else:
                weights = self.lp_weights
                bias = self.lp_bias

        else:
            if fused:  # fused (conv+batch-norm) high-precision weights
                weights = self.hp_fused_weights
                bias = self.hp_fused_bias
            else:      # non-fused (conv only) high-precision weights
                weights = self.hp_weights
                bias = self.hp_bias

        dense_output = self._dense_op(inputs, weights)
        
        if self._use_bias or (self._batch_normalizer_hp.center and fused):
            dense_output = K.bias_add(dense_output, bias)

        if self._activation is not None:
            dense_output = self._activation(dense_output)

        return dense_output

    def _save_params(self):
        if self._lp_conv_bn_fused:
            return [self.lp_fused_weights, self.lp_fused_bias]
        else:
            return [self.lp_weights, self.lp_bias, self.lp_bn_weight, self.lp_bn_bias]

    def quantization_error_for_output(self, some_tensor):
        return None

    def quantization_angle_difference_for_output(self, some_tensor):
        return None



    def get_hp_weights(self):
        conv_weights = self.remove_nones([self.hp_weights, self.hp_bias])
        bn_weights = [self.hp_bn_gamma, self.hp_bn_beta, self.hp_bn_moving_mean, self.hp_bn_moving_variance]

        all_hp_weights = conv_weights + bn_weights
        return all_hp_weights

    def get_hp_fused_weights(self):
        return [self.hp_fused_weights, self.hp_fused_bias]



    def get_lp_weights(self):

        if self._lp_conv_bn_fused:
            lp_weights = [self.lp_fused_weights, self.lp_fused_bias]
        else:
            lp_weights = [self.lp_fused_weights, self.lp_fused_bias,  self.lp_weights]
            if self._use_bias:
                lp_weights.append(self.lp_bias)
            raise ValueError('not implemented yet')

        return self.lp_fused_weights #FIXME


    def quantization_error_for_weights(self):
        return tuple(self._quantizer.quantization_error(w) for w in self.get_hp_weights())

    def quantization_error_for_fused_weights(self):
        return tuple(self._quantizer.quantization_error(w) for w in self.get_hp_fused_weights())

    def quantization_angle_difference_for_weights(self):
        return tuple(self._quantizer.quantization_angle_difference(w) for w in self.get_hp_weights())
