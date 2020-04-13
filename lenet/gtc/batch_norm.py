from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict
import keras
import keras.layers
import tensorflow as tf

momentum_default = 0.99
epsilon_default = 1e-3

# In theory, using the keras dynamic learning phase should work fine, and allow you to
# set K.learning_phase to True or False and have the batch-norm switch training and 
# evaluation modes.
# In practice, having dynamic learning phase causes network performance to crash
# shortly after training starts (this might be a bug in keras, but I'm not sure)
# Thus: to avoid this issue creeping into my experiments, I make sure to 
# K.set_learning_phase(True) (when building a network for training)
# or K.set_learning_phase(False) (when building a network for evaluation) 
# *before* building the network. 
# (Setting the flag below to `False` checks that this is done.)
use_dynamic_learning_phase_when_building = False


class BatchNormalization(GTCLayer):
    """
    Implements Batch Normalization for GTC Layers.
    Can be shared or independent, depending on input arguments
    """

    def __init__(
            self,
            quantizer=None,
            axis=-1,  # batch norm parameters
            momentum=momentum_default,
            epsilon=epsilon_default,
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
            share=None, # can be 'all', 'none', or a list containing a subset of ['running_mean', 'running_variance, 'beta', 'gamma']
                        # or can be 'beta_gamma' (default shared)
                        # if specified, overwrites the individual arguments below
            share_running_mean=False,
            share_running_variance=False,
            share_beta=False,
            share_gamma=False,

            share_type=None, # alternative way to specify type: you can specify share_type='shared' instead of share=['beta', 'gamma']
                             # or type='indep' instead of share='none' to replicate previous 'indep' and 'shared' batch norms
            **kwargs):

        bn_args = dict(
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
        )

        self.center = center
        self.scale = scale

        self._batch_normalizer_hp = keras.layers.normalization.BatchNormalization(
            **bn_args, name='bnorm_hp' )

        self._batch_normalizer_lp = keras.layers.normalization.BatchNormalization(
            **bn_args, name='bnorm_lp')


        all_weights = ['gamma', 'beta',  'running_mean', 'running_variance']
        if share_type is not None:
            if share_type == 'shared':
                share = ['beta', 'gamma']
            elif share_type == 'indep':
                share = []
            else:
                raise ValueError('type should be "shared" or "indep" or left blank, not %s' % share_type)

        if share is not None:
            if share == 'all':
                share = all_weights
            elif share == 'none':
                share = []
            elif share in ['beta_gamma', 'gamma_beta']:
                share = ['beta', 'gamma']
            elif isinstance(share, list):
                assert all( [w in all_weights for w in share] )
            else:
                raise ValueError('Invalid "share" argument : %s' % str(share))

            share_beta = 'beta' in share
            share_gamma = 'gamma' in share
            share_running_mean = 'running_mean' in share
            share_running_variance = 'running_variance' in share

        share_bools = [share_gamma, share_beta,  share_running_mean, share_running_variance]
        self._share_beta = share_beta
        self._share_gamma = share_gamma
        self._share_running_mean = share_running_mean
        self._share_running_variance = share_running_variance
        self._share = share
        self._share_bools = share_bools

        # in addition to the boolean flags, we'll also keep a 'self._share' attribute
        # which contains a list of the shared weights (converted from 'all' or 'none' if necessary)

        self._share = [s for b,s in zip(share_bools, all_weights) if b ]
        self._not_share = [s for b,s in zip(share_bools, all_weights) if not b ] # useful for when loading pretrained weights

        self._quantizer = quantizer

        self._fused = False  # switch to True if/when this layer is fused with a preceding conv layer

        self._layer_config = bn_args
        self._layer_config['share'] = self._share

        trainable = kwargs.pop('trainable', True)
        super(BatchNormalization, self).__init__(
            trainable=trainable, quantizer=quantizer, **kwargs)

    def build(self, input_shape):

        if isinstance(input_shape, dict) and ("gtc_tag" in input_shape):
            input_shape = input_shape['hp']

        input_shape_tuple = tuple(input_shape.as_list())
        self._batch_normalizer_hp.build(input_shape_tuple)
        self._batch_normalizer_lp.build(input_shape_tuple)

        if self._share_beta:
            self._batch_normalizer_lp.beta = self._quantizer.quantize(
                self._batch_normalizer_hp.beta, name='bnorm_lp_beta')

        if self._share_gamma:
            self._batch_normalizer_lp.gamma = self._quantizer.quantize(
                self._batch_normalizer_hp.gamma, name='bnorm_lp_gamma')

        if self._share_running_mean:
            self._batch_normalizer_lp.moving_mean = self._quantizer.quantize(
                self._batch_normalizer_hp.moving_mean, name='bnorm_lp_moving_mean')

        if self._share_running_variance:
            self._batch_normalizer_lp.moving_variance = self._quantizer.quantize(
                self._batch_normalizer_hp.moving_variance, name='bnorm_lp_moving_variance')

        # Define the hp and lp (quantized) weights/bias
        self.hp_weights = self._batch_normalizer_hp.gamma
        self.hp_bias = self._batch_normalizer_hp.beta
        self.hp_moving_mean = self._batch_normalizer_hp.moving_mean
        self.hp_moving_variance = self._batch_normalizer_hp.moving_variance

        self.lp_weights = self._batch_normalizer_lp.gamma
        self.lp_bias = self._batch_normalizer_lp.beta
        self.lp_moving_mean = self._batch_normalizer_lp.moving_mean
        self.lp_moving_variance = self._batch_normalizer_lp.moving_variance


    def call(self, inputs, training=None):

        # Before using K.set_learning_phase(...), the K.learning_phase returns a tensor 
        # Afterwards, it returns a python bool

        if training is None:
            learn_phase = keras.backend.learning_phase()
            if use_dynamic_learning_phase_when_building:
                assert isinstance(learn_phase, tf.Tensor), \
                    "Please use dynamic learning phase (ie. use K.set_learning_phase(val) only after network is built)"
            else:
                assert isinstance(learn_phase, bool), \
                    "Please use static learning phase (ie. use K.set_learning_phase(val) before network is built)"
        else:
            if use_dynamic_learning_phase_when_building:
                assert isinstance(training, tf.Variable), \
                    "Please use dynamic learning phase (ie. pass in tf.Variable for 'training' argument)"
            else:
                assert isinstance(training, bool), \
                    "Please use static learning phase (ie. pass in boolean for 'training' argument)"


        hp_input, lp_input = unpackGTCDict(inputs)

        hp_output_bn = self._batch_normalizer_hp(hp_input, training=training)

        lp_training = training
        if self._share_running_mean or self._share_running_variance:
            lp_training = False # do not add direct updates to lp moving mean/var,
                                # since are inherited from hp.
        lp_output_bn = self._batch_normalizer_lp(lp_input, training=lp_training)

        return GTCDict(hp=hp_output_bn, lp=lp_output_bn)

    def reset_batch_norm_updates(self):
        self._batch_normalizer_hp._updates = []
        self._batch_normalizer_lp._updates = []

    def get_batch_norm_updates(self):
        hp_updates = self._batch_normalizer_hp._updates
        lp_updates = self._batch_normalizer_lp._updates
        return hp_updates + lp_updates

    def get_hp_weights(self):
        hp_weights = [self.hp_weights, self.hp_bias,
                      self.hp_moving_mean, self.hp_moving_variance]
        return self.remove_nones(hp_weights)

    def get_hp_trainable_weights(self):
        if self.trainable:
            return self.remove_nones([self.hp_weights, self.hp_bias])
        else:
            return []

    def get_hp_non_trainable_weights(self):
        wgts = self.remove_nones([self.hp_moving_mean, self.hp_moving_variance])
        if not self.trainable:
            wgts += self.remove_nones([self.hp_weights, self.hp_bias])
        return wgts

    def get_lp_weights(self):
        lp_weights = [self.lp_weights, self.lp_bias,
                      self.lp_moving_mean, self.lp_moving_variance]
        return self.remove_nones(lp_weights)


    def get_lp_trainable_weights(self):
        lp_trainable = []
        if not self._share_gamma and self.trainable:  # trainable if is independent of HP (NOT shared)
            lp_trainable.append(self.lp_weights)
        if not self._share_beta and self.trainable:   # trainable if is independent of HP (NOT shared)
            lp_trainable.append(self.lp_bias)

        return self.remove_nones(lp_trainable)

    def get_lp_non_trainable_weights(self):

        lp_non_trainable = []
        if self._share_gamma or not self.trainable:  # not trainable if is shared with HP
            lp_non_trainable.append(self.lp_weights)
        if self._share_beta or not self.trainable:   # not trainable if is shared with HP
            lp_non_trainable.append(self.lp_bias)

        # These two are always non-trainable, even when shared with HP
        lp_non_trainable.append(self.lp_moving_mean)
        lp_non_trainable.append(self.lp_moving_variance)

        return self.remove_nones(lp_non_trainable)



