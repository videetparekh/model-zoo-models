from gtc.kerasLayer import GTCKerasLayer
import keras.layers
import tensorflow as tf
import numpy as np




class Preprocess(GTCKerasLayer):
    def __init__(self, op_list=None, defaults=None,
                 quantizer=None, name=None, **kwargs):

        if defaults is not None:
            op_list = getDefaultOpsList(defaults)
            # contains_any = lambda x, strs: any([str in x for str in strs])
        else:
            assert op_list is not None, "Define op_list if not providing 'defaults'"

        self._op_list = op_list

        self._layer_config = dict(op_list=op_list, defaults=defaults)

        op_names_concat = '_'.join([op if isinstance(op, str) else op[0] for op in op_list])

        # Define a single lambda function that performs all the operations:
        preprocess_func = keras.layers.Lambda(lambda x: apply_ops(x, self._op_list),
                                              name=op_names_concat)
        if name is None:
            name = 'Preprocess'

        layer_name = name
        if op_names_concat not in layer_name:
            layer_name = name + '_' + op_names_concat

        super(Preprocess, self).__init__(preprocess_func, name=layer_name,
                                         quantizer=quantizer, **kwargs)


# Pure Keras version of Preprocess
# For using in building single-precision keras models from gtcModels
class keras_Preprocess(keras.layers.Layer):

    def __init__(self, op_list=None, defaults=None,
                 quantizer=None, name=None, **kwargs):

        if defaults is not None:
            op_list = getDefaultOpsList(defaults)
            # contains_any = lambda x, strs: any([str in x for str in strs])
        else:
            assert op_list is not None, "Define op_list if not providing 'defaults'"

        self._op_list = op_list
        super(keras_Preprocess, self).__init__(**kwargs)

    def build(self, input_shape):
        super(keras_Preprocess, self).build(input_shape)

    def call(self, x, **kwargs):
        y = apply_ops(x, self._op_list)
        return y






def getDefaultOpsList(defaults):
    # Allows common preprocessing operations
    #    op_list should be a list of ops, provided as tuples containing
    #    name of op and value (if necessary)
    #    e.g.:   [ ('divide', 127.5), ('subtract', 1.0), ('swap_channels') ]
    #           would divide the input by 127.5, subtract 1.0, and then
    #           reverse channels (RGB->BGR)
    #    Alternatively, provide a 'defaults' argument to use the default
    #    preprocessing operations from a particular network
    #    e.g, 'mobilenet_tf', or 'mobilenet_keras',


    defaults_lower = defaults.lower()
    if defaults_lower in ['mobilenet_tf']: # , 'mobilenet_ssd_tf']:
        # for mobilenet-v1, v2 (tensorflow) checkpoints
        # ### [removed: and for mobilenet_ssd v1,v2*,v3*(tensorflow)]
        op_list = [('divide', 128.0), ('subtract', 1.0)]

    elif defaults_lower in ['mobilenet_keras', 'mobilenet_ssd_tf']:
        # for mobilenet v1,v2 (keras.applications),
        # and for mobilenet_ssd v1,v2*,v3*(tensorflow)
        # For mobilenet_ssd (tensorflow): the docs say to divide by 128.0,
        # but in the frozen graph the division is by 127.5
        # To be able to match the graph for checking activations, we'll use 127.5
        op_list = [('divide', 127.5), ('subtract', 1.0)]


    elif defaults_lower in ['vgg_keras', 'resnet_keras']:
        # resnet (keras.applications) or vgg (keras.applications):
        # switch to BGR, then subtract imagenet channelwise (BGR) mean:
        op_list = [('swap_channels', ), ('subtract', [103.939, 116.779, 123.68] )]

    elif defaults_lower in ['vgg_ssd_keras']:
        # Using RGB order, swap channels
        op_list = [('subtract', [123, 117, 104]),  ('swap_channels', )]

    elif defaults_lower in ['mobilenet_ssd_keras']:
        # keras implementation of mobilenet-v1-ssd,
        # https://github.com/ManishSoni1908/Mobilenet-ssd-keras
        # original also had ('swap_channels', ), because was loading with
        # opencv, which loads in BGR format.
        op_list = [('subtract', 127.5), ('divide', 127.5), ('swap_channels', ) ]

    elif defaults_lower in ['resnet_ic']:
        # resnet (image-classifiers):  no preprocessing:
        op_list = []


    else:
        raise ValueError('Unrecognized preprocessing default spec.')

    return op_list



def apply_ops(x, op_list):
    for op in op_list:
        if isinstance(op, (list, tuple)) and len(op) == 1:
            op = op[0]

        if isinstance(op, (list, tuple)) and len(op) == 2:
            op_name = op[0]
            value = op[1]

        elif isinstance(op, str):
            op_name = op
            value = None
        else:
            raise ValueError('Invalid op format')

        if op_name == 'multiply':
            x *= np.array(value, dtype=np.float32)
        elif op_name == 'divide':
            x /= np.array(value, dtype=np.float32)
        elif op_name == 'add':
            x += np.array(value, dtype=np.float32)
        elif op_name == 'subtract':
            x -= np.array(value, dtype=np.float32)
        elif op_name == 'swap_channels':
            x = x[..., ::-1]
        else:
            raise ValueError('Unrecognized operation : %s' % op[0])
    return x


