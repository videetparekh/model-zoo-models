"""
A custom Keras layer to generate anchor boxes.

Copyright (C) 2017 Pierluigi Ferrari

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import division
import numpy as np
import keras.backend as K
#from keras.engine.topology import InputSpec
#from keras.engine.topology import Layer
from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict
import tensorflow.compat.v1 as tf
from ssd import ssd_utils as ssd_utl
import keras

class AnchorBoxes(GTCLayer):
    """
    A Keras layer to create an output tensor containing anchor box coordinates
    and variances based on the input tensor and the passed arguments.

    A set of 2D anchor boxes of different aspect ratios is created for each spatial unit of
    the input tensor. The number of anchor boxes created per unit depends on the arguments
    `aspect_ratios` and `two_boxes_for_ar1`, in the default case it is 4. The boxes
    are parameterized by the coordinate tuple `(xmin, xmax, ymin, ymax)`.

    The purpose of having this layer in the network is to make the model self-sufficient
    at inference time. Since the model is predicting offsets to the anchor boxes
    (rather than predicting absolute box coordinates directly), one needs to know the anchor
    box coordinates in order to construct the final prediction boxes from the predicted offsets.
    If the model's output tensor did not contain the anchor box coordinates, the necessary
    information to convert the predicted offsets back to absolute coordinates would be missing
    in the model output. The reason why it is necessary to predict offsets to the anchor boxes
    rather than to predict absolute box coordinates directly is explained in `README.md`.

    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
        or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.

    Output shape:
        5D tensor of shape `(batch, height, width, n_boxes, 8)`. The last axis contains
        the four anchor box coordinates and the four variance values for each box.
    """

    def __init__(self,
                 ssd_config=None,
                 scale_idx=None,
                 **kwargs):
        """
        Some of these arguments are explained in more detail in the documentation of the `SSDBoxEncoder` class.

        Arguments:
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generated anchor boxes
                as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            aspect_ratios (list, optional): The list of aspect ratios for which default boxes are to be
                generated for this layer. Defaults to [0.5, 1.0, 2.0].
            two_boxes_for_ar1 (bool, optional): Only relevant if `aspect_ratios` contains 1.
                If `True`, two default boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor. Defaults to `True`.
            limit_boxes (bool, optional): If `True`, limits box coordinates to stay within image boundaries.
                Defaults to `True`.
            variances (list, optional): A list of 4 floats >0 with scaling factors (actually it's not factors but divisors
                to be precise) for the encoded predicted box coordinates. A variance value of 1.0 would apply
                no scaling at all to the predictions, while values in (0,1) upscale the encoded predictions and values greater
                than 1.0 downscale the encoded predictions. If you want to reproduce the configuration of the original SSD,
                set this to `[0.1, 0.1, 0.2, 0.2]`, provided the coordinate format is 'centroids'. Defaults to `[1.0, 1.0, 1.0, 1.0]`.
            coords (str, optional): The box coordinate format to be used. Can be either 'centroids' for the format
                `(cx, cy, w, h)` (box center coordinates, width, and height) or 'minmax' for the format
                `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids'.
            normalize_coords (bool, optional): Set to `True` if the model uses relative instead of absolute coordinates,
                i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates. Defaults to `False`.
        """
        if K.backend() != 'tensorflow':
            raise TypeError("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))

        assert ssd_config is not None
        assert scale_idx is not None

        self.scale_idx = scale_idx
        self.config = ssd_config
        self.box_generator = ssd_utl.SSDAnchorBoxGenerator(ssd_config)

        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, dict) and ("gtc_tag" in input_shape):
            input_shape = input_shape['hp']
        input_shape_tuple = tuple(input_shape.as_list())

        super(AnchorBoxes, self).build(input_shape_tuple)

    def call(self, inputs, mask=None):
        """
        Return an anchor box tensor based on the shape of the input tensor.

        Note that this tensor does not participate in any graph computations at runtime. It is being created
        as a constant once during graph creation and is just being output along with the rest of the model output
        during runtime. Because of this, all logic is implemented as Numpy array operations and it is sufficient
        to convert the resulting Numpy array into a Keras tensor at the very end before outputting it.

        Arguments:
            inputs (tensor): 4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
                or `(batch, height, width, channels)` if `dim_ordering = 'tf'`. The input for this
                layer must be the output of the localization predictor layer.
        """
        hp_input, lp_input = unpackGTCDict(inputs)

        boxes_tensor = anchor_boxes_layer_call(self, hp_input)

        return GTCDict(hp=boxes_tensor, lp=boxes_tensor)


    def get_config(self):
        config = {
            'scale_idx': self.scale_idx,
            'ssd_config': self.config,
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





class keras_AnchorBoxes(keras.layers.Layer):
    """
    Same as AnchorBoxes class above, but for a pure-keras model.
    """

    def __init__(self,
                 ssd_config=None,
                 scale_idx=None,
                 **kwargs):

        if K.backend() != 'tensorflow':
            raise TypeError("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))

        assert ssd_config is not None
        assert scale_idx is not None

        self.scale_idx = scale_idx
        self.config = ssd_config
        self.box_generator = ssd_utl.SSDAnchorBoxGenerator(ssd_config)
        self.boxes_per_location = self.config.n_boxes[self.scale_idx]
        super(keras_AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        super(keras_AnchorBoxes, self).build(input_shape)

    def call(self, inputs, mask=None):
        boxes_tensor = anchor_boxes_layer_call(self, inputs)
        assert boxes_tensor._shape_tuple()[3] == self.boxes_per_location
        return boxes_tensor

    def compute_output_shape(self, input_shape):
        if K.image_data_format() == 'channels_last': #K.image_dim_ordering() == 'tf':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        else: #
            batch_size, feature_map_channels, feature_map_height, feature_map_width = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.boxes_per_location, self.config.anchors_size)


    def get_config(self):
        config = {
            'scale_idx': self.scale_idx,
            'ssd_config': self.config,
        }
        base_config = super(keras_AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




# Core `call` function for both GTC AnchorBoxes layer and keras AnchorBoxes layers

def anchor_boxes_layer_call(self, inputs):

    # We need the shape of the input tensor
    assert( K.image_data_format() == 'channels_last')  # K.image_dim_ordering() == 'tf'
    #batch_size, feature_map_height, feature_map_width, feature_map_channels =  x._keras_shape
    batch_size, feature_map_height, feature_map_width, feature_map_channels = tuple(inputs.shape.as_list())

    feature_map_size = (feature_map_height, feature_map_width)

    boxes_tensor = self.box_generator.generate_boxes_for_scale(
        self.scale_idx, feature_map_size)

    if self.config.add_variances_to_anchors:
        # Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
        # as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor) # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor += self.config.variances  # Long live broadcasting
        # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

    # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
    # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
    boxes_tensor = np.expand_dims(boxes_tensor, axis=0)

    #boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(hp_input)[0], 1, 1, 1, 1))

    boxes_tensor = K.constant(boxes_tensor)
    boxes_tensor = K.tile(boxes_tensor, (K.shape(inputs)[0], 1, 1, 1, 1))

    return boxes_tensor
