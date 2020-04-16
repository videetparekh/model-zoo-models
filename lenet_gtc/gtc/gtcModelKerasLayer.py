import tensorflow.compat.v1 as tf
import tensorflow.keras as keras


class GTCModelKerasLayer(keras.layers.Layer):
    """
    A Keras layer class to encapsulate all the internals of the gtcModel
    inside a single real keras layer, from which we can then create a keras model,
    on which we can call the keras .fit function.
    Note that this inherits from keras.layers.Layer,
    not from tf.keras.layers.Layer, as the other GTCLayers do.
    Since this is only a wrapper class, which does not actually create any of
    the layers, we should pass in the following arguments:
     1. 'get_weights_function': retrieve all the trainable weights
        (#1b. optional: also retrieve all the nontrainable weights
        (to display in summary with model.summary() )
     2. 'get_layer_updates_function' retrieve all the layer updates
         (e.g. for batch-norm updates)
     3. 'get_losses_function': retrieve the losses (weight/activity regularizers)
     4. Either:
        4a. 'call_function', which does the forward pass on an input, running a
         keras-input though all the gtcLayers of the gtcModel
         OR
        4b. 'outputs': the symbolic tensor output of this function
     5. optional: the output_shape, if known (not necessary if 'outputs' provided)
         If it is not passed, then call_function should be defined. The output
         shape will be calculated by running the forward pass and inspecting the
        shape of the output
    The gtcModel class implements convenient functions to be directly passed
    for these functions.
    """

    def __init__(self, get_weights_function=None,
                 get_layer_updates_function=None,
                 get_losses_function=None,
                 call_function=None, outputs=None, # provide one of these two
                 output_shape=None, **kwargs):
        self._get_weights_function = get_weights_function
        self._get_layer_updates_function = get_layer_updates_function
        self._get_losses_function = get_losses_function
        self._call_function = call_function
        self._outputs = outputs
        self._output_shape = output_shape
        super(GTCModelKerasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add references to trainable weights, so backprop can update these weights
        # if self._get_weights_function is not None:
        #     self.trainable_weights += self._get_weights_function(trainable=True, non_trainable=False)
        #     self.non_trainable_weights += self._get_weights_function(trainable=False, non_trainable=True)
        # --------- These references ARE READ ONLY in TF 2.0 you can not modify them
        # --------- At the previous TF 1.13 it was necessary to add those lines

        if self._get_losses_function is not None:
            # TODO: distinguish between weight regularizers (input independent)
            # and activity regularizers (input dependent)
            self.add_loss(self._get_losses_function())

        super(GTCModelKerasLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        if self._outputs is not None:
            y = self._outputs   # this is the easiest
        else:
            assert self._call_function is not None
            y = self._call_function(x)

        # Add layer updates so batch-norm running mean/var gets updated
        if self._get_layer_updates_function is not None:
            updates = self._get_layer_updates_function()
            self.add_update(updates, inputs=x)

        return y


    def compute_output_shape(self, input_shape):
        # 1. Either return explicitly defined output-shape
        if self._output_shape is not None:
            return self._output_shape

        def output_shape_for_elem(x):
            if len(x.shape) > 1: # For tensor outputs
                return (None,) + tuple(x.shape.as_list()[1:])
            else: # For scalar outputs (e.g. distillation/bit loss)
                return tuple(x.shape.as_list())

        # 2. Or calculate it, either from:
        # 2a. explicitly provided output tensor
        if self._outputs is not None:
            test_output = self._outputs

        # 2b. or using input and call-function
        else:
            if isinstance(input_shape, dict) and ("gtc_tag" in input_shape):
                input_shape = input_shape['hp']
            if isinstance(input_shape, tuple):
                input_shape_tuple = input_shape
            else:
                input_shape_tuple = tuple(input_shape.as_list())

            sample_input_shape = (1,) + input_shape_tuple[1:]
            test_input = tf.zeros(sample_input_shape)

            test_output = self._call_function(test_input)


        if isinstance(test_output, (list, tuple)):
            output_shape = [output_shape_for_elem(x) for x in test_output]
        else:
            output_shape = output_shape_for_elem(test_output)

        return output_shape


        #return [(input_shape[0],) + self._output_shape[1:], ]


