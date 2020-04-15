from gtc.kerasLayer import GTCKerasLayer
import tensorflow.keras.layers
import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K


class Resize(GTCKerasLayer):
    def __init__(self, new_shape, interpolation='bilinear', quantizer=None, name=None, **kwargs):
        assert interpolation in ['nearest', 'bilinear'], "Invalid interpolation"

        keras_lambda_func = keras.layers.Lambda(lambda x: resize(x, new_shape, interpolation))
        self._layer_config = dict(new_shape=new_shape, interpolation=interpolation)

        super(Resize, self).__init__(keras_lambda_func, name=name, quantizer=quantizer, **kwargs)




# Pure Keras version of Resize
# For using in building single-precision keras models from gtcModels

class keras_Resize(tensorflow.keras.layers.Layer):
    # Keras layer that does resizing (for building with a pure-keras model),
    # that mirrors the job that the Resize(GTCKerasLayer) does for a GTCModel.

    def __init__(self, new_shape, interpolation='bilinear', name=None, **kwargs):
        self.interpolation = interpolation
        self.new_shape = new_shape
        super(keras_Resize, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        super(keras_Resize, self).build(input_shape)

    def call(self, x, **kwargs):
        y = resize(x, self.new_shape, self.interpolation)
        return y

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.new_shape[0], self.new_shape[1]) + input_shape[3:]
        return output_shape



def resize(x, new_size, interpolation):
    cur_size = x.shape.as_list()[1:]
    height_factor = new_size[0] / cur_size[0]
    width_factor = new_size[1] / cur_size[1]

    y = keras.backend.resize_images(
        x, height_factor, width_factor,
        data_format=keras.backend.image_data_format(),
        interpolation=interpolation)
    return y
