from gtc.GTCLayer import GTCDict, unpackGTCDict, GTCLayer
import keras.layers
import tensorflow.keras.backend as K

class Reshape(GTCLayer):
    """
    implements Reshape operation on the gtc layer output
    """
    def __init__(self, target_shape, quantizer=None, name=None, **kwargs):
        # Target shape should not include batch dimension.
        # Target shape can have a maximum of one unknown dimension
        # (indicated by size of -1 for one of the dimensinos)
        self._target_shape = target_shape
        self._layer_config = dict(target_shape=target_shape)

        trainable = kwargs.pop('trainable', False)

        super(Reshape, self).__init__(trainable=trainable,
                                      name=name, quantizer=quantizer, **kwargs)

    def call(self, inputs, **kwargs):

        hp_input, lp_input = unpackGTCDict(inputs, error_if_not_gtc_dict=False)

        # The output tensor of a keras.layers.Reshape loses some information about the shape.
        # So we call it directly on the input to make it calculate the output shape,
        # but we then use this output shape directly with K.reshape().
        # This makes more information about the shape available to downstream layers.

        keras_layer = keras.layers.Reshape(self._target_shape)
        reshaped_output = keras_layer(hp_input)

        # use keras_layer.output_shape instead of reshaped_output.shape
        # output_shape = tuple([-1] + sample_reshaped_output.shape[1:].as_list()   )
        output_shape = tuple([(x if x is not None else -1) for x in
                              keras_layer.output_shape])  # replace None, with -1 (should be maximum of 1 occurence)

        hp_output = K.reshape( hp_input, output_shape )
        lp_output = K.reshape( lp_input, output_shape )
        if self._quantizer is not None:
            lp_output = self._quantizer.quantize(lp_output, name=self.name + '_requantize')

        return GTCDict(hp=hp_output, lp=lp_output)
