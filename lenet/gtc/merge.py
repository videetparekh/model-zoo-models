from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict
import tensorflow.compat.v1 as tf
import keras.layers
from functools import partial

use_functional_API = True

class _Merge(GTCLayer):
    """
    Generic merge layer for elementwise merge functions.  Based on the keras.layers._Merge superclass
    """

    def __init__(self, keras_merge_layer, keras_kwargs=None, layer_inputs=None, name=None, quantizer=None, **kwargs):
        """
        Validate layer_inputs, which can be a list of tensors, or names of tensors that will be resolved during at model.compile()
        """
        assert layer_inputs is not None, "Must provide inputs to merge layer"
        assert isinstance(layer_inputs, (list, tuple)) and len(layer_inputs) >= 2,\
            "Must provide at least two inputs to merge layer"

        inputs_names_list = []
        for input in layer_inputs:
            if isinstance(input, str):
                inputs_names_list.append(input)
            elif isinstance(input, tf.keras.layers.Layer):
                inputs_names_list.append(input.name)
            else:
                raise ValueError('Each input must either be a string or a keras layer')

        if keras_kwargs is None:
            keras_kwargs = {}

        kwargs_extended = dict(**kwargs, layer_inputs=inputs_names_list)
        self._keras_merge_layer = keras_merge_layer
        self._keras_kwargs = keras_kwargs
        self._quantizer = quantizer

        self._layer_config = dict(**keras_kwargs)
        trainable = kwargs_extended.pop('trainable', False)

        super(_Merge, self).__init__(name=name, quantizer=quantizer,
                                     trainable=trainable, **kwargs_extended)

    def build(self, input_shapes_list):
        if isinstance(input_shapes_list, dict) and ("gtc_tag" in input_shapes_list):
            input_shapes_list = input_shapes_list['hp']
        input_shape_tuple_list = [tuple(shp.as_list()) for shp in input_shapes_list]

        if not use_functional_API:
            # build
            self._keras_merge_layer.build(input_shape_tuple_list)

        super(_Merge, self).build(input_shapes_list)

    def call(self, inputs, **kwargs):

        hp_input, lp_input = unpackGTCDict(inputs, error_if_not_gtc_dict=True)

        if use_functional_API:
            hp_output = self._keras_merge_layer(hp_input, **self._keras_kwargs)
            lp_output = self._keras_merge_layer(lp_input, **self._keras_kwargs)
        else:
            hp_output = self._keras_merge_layer.call(hp_input)
            lp_output = self._keras_merge_layer.call(lp_input)


        if self._quantizer is not None:
            lp_output = self._quantizer.quantize(lp_output, name=self.name + "_requantize")

        return GTCDict(hp=hp_output, lp=lp_output)




class Add(_Merge):
    def __init__(self, layer_inputs=None, quantizer=None, name=None, **kwargs):
        if use_functional_API:
            keras_merge_layer = keras.layers.add
        else:
            keras_merge_layer = keras.layers.Add()
        keras_kwargs = {}
        super(Add, self).__init__(keras_merge_layer, keras_kwargs, layer_inputs,
                                  quantizer=quantizer, name=name, **kwargs)


class Subtract(_Merge):
    def __init__(self, layer_inputs=None, quantizer=None, name=None, **kwargs):
        if use_functional_API:
            keras_merge_layer = keras.layers.subtract
        else:
            keras_merge_layer = keras.layers.Subtract()
        keras_kwargs = {}
        super(Subtract, self).__init__(keras_merge_layer, keras_kwargs, layer_inputs,
                                       quantizer=quantizer, name=name, **kwargs)


class Multiply(_Merge):
    def __init__(self, layer_inputs=None, quantizer=None, name=None, **kwargs):
        if use_functional_API:
            keras_merge_layer = keras.layers.multiply
        else:
            keras_merge_layer = keras.layers.Multiply()
        keras_kwargs = {}
        super(Multiply, self).__init__(keras_merge_layer, keras_kwargs, layer_inputs,
                                       quantizer=quantizer, name=name, **kwargs)

class Average(_Merge):
    def __init__(self, layer_inputs=None, quantizer=None, name=None, **kwargs):
        if use_functional_API:
            keras_merge_layer = keras.layers.average
        else:
            keras_merge_layer = keras.layers.Average()
        keras_kwargs = {}
        super(Average, self).__init__(keras_merge_layer, keras_kwargs, layer_inputs,
                                       quantizer=quantizer, name=name, **kwargs)

class Maximum(_Merge):
    def __init__(self, layer_inputs=None, quantizer=None, name=None, **kwargs):
        if use_functional_API:
            keras_merge_layer = keras.layers.maximum
        else:
            keras_merge_layer = keras.layers.Maximum()
        keras_kwargs = {}
        super(Maximum, self).__init__(keras_merge_layer, keras_kwargs, layer_inputs,
                                      quantizer=quantizer, name=name, **kwargs)

class Minimum(_Merge):
    def __init__(self, layer_inputs=None, quantizer=None, name=None, **kwargs):
        if use_functional_API:
            keras_merge_layer = keras.layers.minimum
        else:
            keras_merge_layer = keras.layers.Minimum()
        keras_kwargs = {}
        super(Minimum, self).__init__(keras_merge_layer, keras_kwargs, layer_inputs,
                                      quantizer=quantizer, name=name, **kwargs)

class Concatenate(_Merge):
    def __init__(self, layer_inputs, axis=-1, quantizer=None, name=None, **kwargs):
        if use_functional_API:
            keras_merge_layer = keras.layers.concatenate
        else:
            keras_merge_layer = keras.layers.Concatenate(axis=axis)
        keras_kwargs = dict(axis=axis)

        super(Concatenate, self).__init__(keras_merge_layer, keras_kwargs, layer_inputs,
                                          quantizer=quantizer, name=name, **kwargs)

class Dot(_Merge):
    def __init__(self, layer_inputs=None, axes=None, normalize=False, quantizer=None, name=None, **kwargs):
        if use_functional_API:
            keras_merge_layer = keras.layers.dot
        else:
            keras_merge_layer = keras.layers.Dot(axes=axes, normalize=normalize)
        keras_kwargs = dict(axes=axes, normalize=normalize)

        super(Dot, self).__init__(keras_merge_layer, keras_kwargs, layer_inputs,
                                  quantizer=quantizer, name=name, **kwargs)


