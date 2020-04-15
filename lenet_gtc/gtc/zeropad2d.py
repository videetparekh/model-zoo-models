from gtc.kerasLayer import GTCKerasLayer
from gtc.GTCLayer import GTCLayer, unpackGTCDict, GTCDict
import tensorflow.keras.layers

class ZeroPadding2D(GTCLayer):

    def __init__(self, padding=None, quantizer=None, name=None, **kwargs):
        self._padding = padding
        self._layer_config = dict(padding=padding)
        #keras_layer = keras.layers.ZeroPadding2D(padding, name=name, **kwargs)
        super(ZeroPadding2D, self).__init__(name=name, quantizer=quantizer, **kwargs)


    def call(self, inputs, **kwargs):

        hp_input, lp_input = unpackGTCDict(inputs, error_if_not_gtc_dict=False)

        keras_layer = keras.layers.ZeroPadding2D(self._padding, name=self._name)
        #print(' >>>>  ZeroPad module : %s : hp_input = %s\n' % (self.name, hp_input.name))

        hp_output = keras_layer(hp_input)
        lp_output = keras_layer(lp_input)
        if self._quantizer is not None:
            lp_output = self._quantizer.quantize(lp_output, name=self.name + '_requantize')

        return GTCDict(hp=hp_output, lp=lp_output)
