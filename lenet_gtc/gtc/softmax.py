from gtc.kerasLayer import GTCKerasLayer
import tensorflow.keras.layers

class Softmax(GTCKerasLayer):
    def __init__(self, axis=-1, name=None, quantizer=None, **kwargs):
        keras_layer = keras.layers.Softmax(axis)
        super(Softmax, self).__init__(keras_layer, quantizer=quantizer, name=name, **kwargs)

