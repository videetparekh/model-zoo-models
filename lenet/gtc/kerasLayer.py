from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict


class GTCKerasLayer(GTCLayer):
    """
    Applies an arbitrary keras layer (that doesn't have any trainable weights) 
    on the gtc layer input
    """

    def __init__(self, keras_layer, quantizer=None, name=None, **kwargs):
        """
        TODO: Docstring for __init__.
        """

        #self._args = kwargs
        self._keras_layer = keras_layer
        self._quantizer = quantizer

        if not hasattr(self, '_layer_config'):
            self._layer_config = dict(keras_layer=keras_layer)

        trainable = kwargs.pop('trainable', False)
        super(GTCKerasLayer, self).__init__(name=name, quantizer=quantizer,
                                            trainable=trainable, **kwargs)

    def build(self, input_shape, **kwargs):

        super(GTCKerasLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        hp_input, lp_input = unpackGTCDict(inputs, error_if_not_gtc_dict=False)

        hp_output = self._keras_layer(hp_input)
        lp_output = self._keras_layer(lp_input)
        if self._quantizer is not None:
            lp_output = self._quantizer.quantize(lp_output, name=self.name + '_requantize')

        return GTCDict(hp=hp_output, lp=lp_output)




