from gtc.kerasLayer import GTCKerasLayer
import tensorflow.keras.layers

# Note: cannot name this file lambda.py

class Lambda(GTCKerasLayer):
    """
    implements an arbitrary lambda operation on the gtc layer output
    """
    def __init__(self, lambda_func, quantizer=None, name=None, **kwargs):
        keras_layer = keras.layers.Lambda(lambda_func, **kwargs)

        self._layer_config = dict(lambda_func=lambda_func)

        super(Lambda, self).__init__(keras_layer, name=name, quantizer=quantizer, **kwargs)
