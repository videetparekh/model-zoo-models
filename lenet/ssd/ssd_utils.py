'''
Includes:
* Function to compute IoU similarity for axis-aligned, rectangular, 2D bounding boxes
* Function to perform greedy non-maximum suppression
* Function to decode raw SSD model output
* Class to encode targets for SSD model training

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
'''

from __future__ import division
import numpy as np
import tensorflow as tf
from scipy.special import softmax, expit as sigmoid
import time
#from tqdm import tqdm
import cv2
import os
from collections import OrderedDict
import imageio
#from networks import net_types as net  # causes
from ssd.post_processing_tf import post_processing
from ssd.post_processing_tf.faster_rcnn_box_coder import FasterRcnnBoxCoder
import ssd.post_processing_tf.visualization_utils as vis_util
#import ssd.ssd_info as ssd
import keras
import keras.backend as K
from ssd import ssd_info as ssd

default_labels_seq = ('class_id', 'xmin', 'ymin', 'xmax', 'ymax')
seq2labelsformat = lambda seq: OrderedDict([ (name, (seq.index(name) if name in seq else -1)) for name in default_labels_seq])
#seq2labelsformat = lambda seq: {name: (seq.index(name) if name in seq else -1) for name in default_labels_seq}


ssd_scales_voc = (0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05)  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
ssd_scales_coco = (0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05)  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets

ssd_scales_uniform = (0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1) # uniformly spaced from 0.2 to 0.95
ssd_aspect_ratios_mobilenet = ([1.001, 2.0, 0.5],
                               [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                               [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                               [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                               [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                               [1.0, 2.0, 0.5, 3.0, 1.0/3.0])  # The anchor box aspect ratios used in the original SSD300; the order matters
ssd_aspect_ratios_vgg = ([1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5])  # The anchor box aspect ratios used in the original SSD300; the order matters


# ssd_dataset_mean = (123.0, 117, 104)
ssd_dataset_mean = (127.5, 127.5, 127.5)

steps_vgg = (8, 16, 32, 64, 100, 300)
steps_caffemodel = (16, 32, 64, 100, 150, 300)  # steps used by caffe-model MobilentSSD (0.74 mAP on Pascal for v1-keras model)

coord_variances = (0.1, 0.1, 0.2, 0.2)



class SSDConfig():
    '''
    Configuration class for SSD models. Convenient packaging of all parameters
    into a single class. The parameters here determine:
       * details about the SSD anchor boxes
       * how the ground truth labels are encoded in the anchor boxes
       * hyper-parameters for the SSD loss function
       * parameters for non-max-suppression during decoding
    For convenience, you can pass in a net_type, and the default configurations for
    that network will be used. Supported net_types:
        VGGSSDType() : (the default ssd300 network with VGG backbone)
        MobilenetSSDType(mobilenet_version=1, style='keras')  # Caffe Mobilenet-SSD configuration
        MobilenetSSDType(mobilenet_version=[1,2,3], style='tf')  # Tensorflow Mobilenet-SSD configuration

    Arguments:
        img_height (int): The height of the input images.
        img_width (int): The width of the input images.
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
                min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. Note that you should set the scaling factors
            such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
            to detect. Must be >0.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the
            largest will be linearly interpolated. Note that the second to last of the linearly interpolated
            scaling factors will actually be the scaling factor for the last predictor layer, while the last
            scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
            if `two_boxes_for_ar1` is `True`. Note that you should set the scaling factors
            such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
            to detect. Must be greater than or equal to `min_scale`.
        scales (list, optional): A list of floats >0 containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used. If a list is passed,
            this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
            Note that you should set the scaling factors such that the resulting anchor box sizes correspond to
            the sizes of the objects you are trying to detect.
        aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all prediction layers. Note that you should set the aspect ratios such
            that the resulting anchor box shapes roughly correspond to the shapes of the objects you are trying to detect.
        aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
            If a list is passed, it overrides `aspect_ratios_global`. Note that you should set the aspect ratios such
            that the resulting anchor box shapes very roughly correspond to the shapes of the objects you are trying to detect.
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratios lists that contain 1. Will be ignored otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
            using the scaling factor for the respective layer, the second one will be generated using
            geometric mean of said scaling factor and next bigger scaling factor.
        steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
            pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
            the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
            If no steps are provided, then they will be computed such that the anchor box center points will form an
            equidistant grid within the image dimensions.
        offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either floats or tuples of two floats. These numbers represent for each predictor layer how many
            pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
            as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
            of the step size specified in the `steps` argument. If the list contains floats, then that value will
            be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
            `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
        clip_boxes (bool, optional): If `True`, limits the anchor box coordinates to stay within image boundaries.
        variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
            its respective variance value.
        matching_type (str, optional): Can be either 'multi' or 'bipartite'. In 'bipartite' mode, each ground truth box will
            be matched only to the one anchor box with the highest IoU overlap. In 'multi' mode, in addition to the aforementioned
            bipartite matching, all anchor boxes with an IoU overlap greater than or equal to the `pos_iou_threshold` will be
            matched to a given ground truth box.
        pos_iou_threshold (float, optional): The intersection-over-union similarity threshold that must be
            met in order to match a given ground truth box to a given anchor box.
        neg_iou_limit (float, optional): The maximum allowed intersection-over-union similarity of an
            anchor box with any ground truth box to be labeled a negative (i.e. background) box. If an
            anchor box is neither a positive, nor a negative box, it will be ignored during training.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.
        coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input format
            of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
            and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): If `True`, the encoder uses relative instead of absolute coordinates.
            This means instead of using absolute tartget coordinates, the encoder will scale all coordinates to be within [0,1].
            This way learning becomes independent of the input image size.
        background_id (int, optional): Determines which class ID is for the background class.

        '''

    def __init__(self,
                 # image-size, image preprocessing
                 img_height=None, img_width=None, img_channels=None, imsize=None,
                 label_set=None,
                 n_classes=None,
                 condensed_labels=None, # for COCO, whether the network predicts condensed class indexes [1..80] or original [1..90]
                 class_activation_in_network=None,
                 net_type=None,

                 add_nms_to_network=False,

                 # scales
                 min_scale=None,
                 max_scale=None,
                 scales=ssd_scales_uniform,

                 # aspect ratios
                 aspect_ratios_global=None,
                 aspect_ratios=ssd_aspect_ratios_vgg,
                 n_predictor_layers=None,
                 two_boxes_for_ar1=True,
                 add_extra_ar1_box_at_end=False,
                 reduce_boxes_in_lowest_layer=True,
                 reduce_scale_of_lowest_square_box=False,  # scale of first square box = 0.1 instead of 0.2
                 steps=steps_caffemodel,
                 # The space between two adjacent anchor box center points for each predictor layer.
                 offsets=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                 # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
                 limit_boxes=False,
                 # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
                 box_xy_order='xy',
                 border_pixels='half',
                 matching_type='multi', # can be 'multi' or 'bipartite'
                 background_id=0,
                 decoding_border_pixels='include',

                 # variances=(0.1, 0.1, 0.2, 0.2),  # The variances by which the encoded target coordinates are scaled as in the original implementation
                 variances=coord_variances,
                 add_variances_to_anchors=True,
                 pred_coords='centroids',  # coordinate format of the predictions
                 anchor_coords='centroids',  # coordinate format of the anchor boxes
                 # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
                 normalize_coords=True,

                 # ssd box encoding:
                 encode_pos_iou_threshold=0.5,
                 encode_neg_iou_threshold=0.3,

                 # postprocessing
                 loss_neg_pos_ratio=3,
                 loss_n_neg_min=0,
                 loss_alpha=1.0,

                 matching_iou_thresh=0.5,
                 score_thresh=0.2,
                 nms_iou_thresh=0.5,
                 max_size_per_class=10,
                 max_total_size=100,
                 class_activation=None,  # default for keras: 'softmax',
                 ):

        import networks.net_utils_core as net

        assert net_type is not None, "Please define ssd network type"
        # net_type = net.SSDNetType(mobilenet_version=1, ssd_lite=False, style='keras')
        if isinstance(net_type, str):
            net_type = net.NetworkType.from_str(net_type)

        base_net = net_type.base_net_type
        mobilenet_version = None
        if base_net.name == 'mobilenet': # isinstance(base_net, net.MobilenetType):
            mobilenet_version = base_net.version

        imsize = (320, 320, 3) if mobilenet_version in [3] else (300, 300, 3)
        if net_type.imsize is not None:
            imsize = net_type.imsize
            if isinstance(imsize, int):
                imsize = (imsize, imsize, 3)

            if len(imsize) == 2:
                imsize = tuple(imsize) + (3,)


            # assert isinstance(net_type, net.SSDNetType)
        if 'keras' in net_type.style:
            if label_set is None:
                label_set = 'pascal' # default for keras models
            if condensed_labels is None:
                condensed_labels = True  # for COCO, whether the network predicts condensed class indexes [1..80] or original [1..90]

            add_extra_ar1_box_at_end = False
            reduce_scale_of_lowest_square_box = False
            add_variances_to_anchors = True
            limit_boxes = False

            if class_activation_in_network is None:
                class_activation_in_network = True  # default for keras
            box_xy_order = 'xy'
            pred_coords = 'centroids'  # coordinate format of the predictions
            anchor_coords = 'centroids'  # coordinate format of the anchor boxes


            if base_net.name.lower() == 'mobilenet': # isinstance(base_net, net.MobilenetType):
                aspect_ratios = [ar_list.copy() for ar_list in ssd_aspect_ratios_mobilenet]
                scales = ssd_scales_uniform
                #steps = None  # Different from Mobilenet-ssd-keras code
                steps = steps_caffemodel
                reduce_boxes_in_lowest_layer = True
            elif base_net.name.lower() == 'vgg':
                aspect_ratios = [ar_list.copy() for ar_list in ssd_aspect_ratios_vgg]
                scales = ssd_scales_voc
                steps = steps_vgg  # Different from Mobilenet-ssd-keras code
                reduce_boxes_in_lowest_layer = False

            if class_activation is None:
                class_activation = 'softmax'

        elif net_type.style == 'tensorflow':  # defaults for all tensorflow models:
            if label_set is None:
                label_set = 'coco' # default for tensorflow models
            if condensed_labels is None:
                condensed_labels = False  # for COCO, whether the network predicts original coco indices [1..90] or condensed class indexes [1..80]

            add_extra_ar1_box_at_end=True
            reduce_boxes_in_lowest_layer = True ## SAME ##
            reduce_scale_of_lowest_square_box = True
            add_variances_to_anchors = False
            limit_boxes = False     ## SAME ##
            box_xy_order = 'yx'
            steps = None  # Compute steps automatically
            pred_coords = 'centroids'  # coordinate format of the predictions # (not sure about mobilenet-version 3)
            anchor_coords = 'corners' if mobilenet_version in [1, 2] else 'centroids'
            aspect_ratios = [ar_list.copy() for ar_list in ssd_aspect_ratios_mobilenet]
            aspect_ratios[0][0] = 1.0  # 1.001 --> 1.0
            if class_activation_in_network is None:
                class_activation_in_network = False  # default for tensorflow model


            if class_activation is None:
                class_activation = 'sigmoid'

            mimic_keras = False
            #label_set = 'pascal'
            #n_classes = 20

            #class_activation = 'softmax'

            if mimic_keras:

                #add_extra_ar1_box_at_end = False  ## EXPECT A DIFFERENCE ##  ** STAY WITH TF DEFAULT **
                reduce_boxes_in_lowest_layer = True  ## (SAME as keras anyway) ##
                reduce_scale_of_lowest_square_box = False  ## EXPECT A DIFFERENCE ##
                add_variances_to_anchors = True
                limit_boxes = False      ## (SAME as keras anyway) ##

                # if class_activation_in_network is None:
                class_activation_in_network = True  # default for keras
                #box_xy_order = 'xy'    ## EXPECT A DIFFERENCE ##
                pred_coords = 'centroids'   # coordinate format of the predictions
                anchor_coords = 'centroids'
                steps = steps_caffemodel
                # aspect_ratios = [ar_list.copy() for ar_list in ssd_aspect_ratios]  ## EXPECT A DIFFERENCE ##  ** STAY WITH TF DEFAULT **

                class_activation = 'softmax'


            #add_nms_to_network = True
        else:
            raise ValueError('Unsupported style for SSD : %s' % net_type.style)



        if label_set == 'pascal':
            n_classes = 20
        elif label_set == 'coco':
            assert isinstance(condensed_labels, bool)
            if condensed_labels:
                n_classes = 80  # labels go from 1..80
            else:
                n_classes = 90  # labels go from 1..90, skipping 10 indices.


        if n_predictor_layers is not None:
            assert n_predictor_layers == len(aspect_ratios)
        else:
            n_predictor_layers = len(aspect_ratios)
        # validate image size

        if class_activation == 'sigmoid':
            nms_iou_thresh = 0.6
            matching_iou_thresh = 0.5
            max_size_per_class = 100
            max_total_size = 100
            score_thresh = 0.01

        elif class_activation == 'softmax':
            nms_iou_thresh = 0.45
            matching_iou_thresh = 0.5
            max_size_per_class = 200
            max_total_size = 400
            score_thresh = 0.01
        else:
            raise ValueError('unknown class activation function : %s' % class_activation)


        assert not ((img_height is not None and img_width is not None and img_channels is not None) and (
                    imsize is not None)), \
            "Either specify img_height, img_width & img_channels OR specify im_size (not both)"

        if imsize is not None:
            if len(imsize) == 2:
                img_height, img_width = imsize
                img_channels = 3
            elif len(imsize) == 3:
                img_height, img_width, img_channels = imsize

        elif img_height is None:
            default_size = (300, 300, 3)
            img_height, img_width, img_channels = default_size

        # aspect ratios
        if aspect_ratios_global is None and aspect_ratios is None:
            raise ValueError(
                "`aspect_ratios_global` and `aspect_ratios` cannot both be None. At least one needs to be specified.")
        if aspect_ratios:
            if len(aspect_ratios) != n_predictor_layers:
                raise ValueError(
                    "It must be either aspect_ratios is None or len(aspect_ratios) == {}, but len(aspect_ratios) == {}.".format(
                        n_predictor_layers, len(aspect_ratios)))

        # scales
        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
        if scales:
            if len(scales) != n_predictor_layers + 1:
                raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                    n_predictor_layers + 1, len(scales)))
            for scl_idx, scale in enumerate(scales):
                if scl_idx == n_predictor_layers:
                    scale = np.sqrt(scales[scl_idx - 1] * scales[scl_idx])
                if (scale < 0) or (scale > 1):
                    #raise ValueError("each `scale` must be in [0, 1] but `this_scale` == {}".format(scale))
                    print(" WARNING: each `scale` must be in [0, 1] but `this_scale` == {}".format(scale))

        else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
            scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

        # steps
        if (not (steps is None)) and (len(steps) != n_predictor_layers):
            raise ValueError("You must provide at least one step value per predictor layer.")
        if steps is None:
            steps = [None] * n_predictor_layers

        # offsets
        if offsets is not None:
            if isinstance(offsets, (float, int)):
                offsets = [offsets]*n_predictor_layers
            elif len(offsets) != n_predictor_layers:
                raise ValueError("You must provide at least one offset value per predictor layer.")
        elif offsets is None:
            offsets = [None] * n_predictor_layers

        # variances
        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        # Coords
        valid_coords = ['minmax', 'centroids', 'corners']
        if pred_coords not in valid_coords:
            raise ValueError("Unexpected value for `pred_coords`. Supported values are 'minmax', 'corners' and 'centroids'.")
        if anchor_coords not in valid_coords:
            raise ValueError("Unexpected value for `anchor_coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

        if box_xy_order not in ['xy', 'yx']:
            raise ValueError("box_xy_order must be either 'xy' or 'yx'")

        # Compute the number of boxes to be predicted per cell for each predictor layer.
        # We need this so that we know how many channels the predictor layers need to have.
        n_boxes = []
        for scale_idx, ar in enumerate(aspect_ratios):
            num_boxes_this_scale = len(ar)
            if (1 in ar) & two_boxes_for_ar1 and \
                    (scale_idx>0 or not reduce_boxes_in_lowest_layer):
                num_boxes_this_scale += 1 # +1 for the second box for aspect ratio 1
            n_boxes.append(num_boxes_this_scale)

        self.net_type = net_type

        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.imsize = (img_width, img_height, img_channels)
        self.label_set = label_set
        self.n_classes = n_classes  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
        self.condensed_labels = condensed_labels
        self.class_activation_in_network = class_activation_in_network

        # self.divide_by_stddev_first = divide_by_stddev_first
        # self.subtract_mean = subtract_mean
        # self.divide_by_stddev = divide_by_stddev
        # self.swap_channels = swap_channels  # The color channel order in the original SSD is BGR

        self.add_nms_to_network = add_nms_to_network

        self.n_predictor_layers = n_predictor_layers
        if isinstance(scales, str) and scales.lower() == 'voc':
            scales = ssd_scales_voc
        elif isinstance(scales, str) and scales.lower() == 'coco':
            scales = ssd_scales_coco

        self.scales = scales
        self.aspect_ratios = aspect_ratios  # The anchor box aspect ratios used in the original SSD300; the order matters
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.add_extra_ar1_box_at_end = add_extra_ar1_box_at_end
        self.reduce_boxes_in_lowest_layer = reduce_boxes_in_lowest_layer
        self.reduce_scale_of_lowest_square_box = reduce_scale_of_lowest_square_box
        self.box_xy_order = box_xy_order
        self.border_pixels = border_pixels
        self.matching_type = matching_type
        assert background_id == 0, "non-zero background id not supported"
        self.background_id = background_id

        self.steps = steps
        self.offsets = offsets
        self.limit_boxes = limit_boxes  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries

        self.variances = variances
        self.pred_coords = pred_coords
        self.anchor_coords = anchor_coords
        self.normalize_coords = normalize_coords
        self.add_variances_to_anchors = add_variances_to_anchors
        self.n_boxes = n_boxes

        if add_variances_to_anchors:
            self.anchors_size = 8
        else:
            self.anchors_size = 4

        self.loss_neg_pos_ratio = loss_neg_pos_ratio
        self.loss_n_neg_min = loss_n_neg_min
        self.loss_alpha = loss_alpha

        self.encode_pos_iou_threshold = encode_pos_iou_threshold
        self.encode_neg_iou_threshold = encode_neg_iou_threshold

        # Post processing
        #self.box_coder_scales = list(1.0/v for v in variances)

        self.score_thresh = score_thresh
        self.nms_iou_thresh = nms_iou_thresh
        self.matching_iou_thresh = matching_iou_thresh
        self.max_size_per_class = max_size_per_class
        self.max_total_size = max_total_size
        self.class_activation = class_activation
        self.decoding_border_pixels = decoding_border_pixels

        n_classes = n_classes + 1
        n_variances = len(variances) if add_variances_to_anchors else 0
        n_box_coords = 4
        n_anchor_coords = 4
        self.box_vec_size = n_classes + n_variances + n_box_coords + n_anchor_coords

    def copy(self):
        new_config = SSDConfig(net_type=self.net_type)
        for k, v in self.__dict__.items():
            setattr(new_config, k, v)
        return new_config

    def get_config(self):
        cfg = self.__dict__.copy()
        attrs_remove = ['n_boxes', 'anchors_size', 'box_vec_size',
                        'img_height', 'img_width', 'img_channels']
        for attr in attrs_remove:
            cfg.pop(attr, None)

        cfg['net_type'] = str(self.net_type)
        cfg['variances'] = self.variances.tolist()
        return cfg


predictor_sizes_mobilenet_300 = [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]
predictor_sizes_vgg_300       = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]

class SSDAnchorBoxGenerator:
    '''
    Helper class for generating anchor boxes.
    Used by the SSDBoxEncoder when creating anchor boxes for y (targets) during training
    And also by the keras AnchorBox layer for the SSD network.
    The anchor box generation code logic is consolidated here to ensure
    consistency between the boxes generated in both places.
    Supports both the caffe defaults (and its keras adaptation)
        https://github.com/chuanqi305/MobileNet-SSD and
        https://github.com/ManishSoni1908/Mobilenet-ssd-keras
    as well as the tensorflow defaults
        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
    '''

    def __init__(self, ssd_config):
        self.config = ssd_config

        pass

    def generate_boxes_for_scale(self, scale_idx, feature_map_size,
                                 diagnostics=False):
        '''
        Compute an array of the spatial positions and sizes of the anchor boxes for one predictor layer
        of size `feature_map_size == [feature_map_height, feature_map_width]`.

        Arguments:
            feature_map_size (tuple): A list or tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
            aspect_ratios (list): A list of floats, the aspect ratios for which anchor boxes are to be generated.
                All list elements must be unique.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generate anchor boxes
                as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            diagnostics (bool, optional): If true, the following additional outputs will be returned:
                1) A list of the center point `x` and `y` coordinates for each spatial location.
                2) A list containing `(width, height)` for each box aspect ratio.
                3) A tuple containing `(step_height, step_width)`
                4) A tuple containing `(offset_height, offset_width)`
                This information can be useful to understand in just a few numbers what the generated grid of
                anchor boxes actually looks like, i.e. how large the different boxes are and how dense
                their spatial distribution is, in order to determine whether the box grid covers the input images
                appropriately and whether the box sizes are appropriate to fit the sizes of the objects
                to be detected.

        Returns:
            A 4D Numpy tensor of shape `(feature_map_height, feature_map_width, n_boxes_per_cell, 4)` where the
            last dimension contains `(xmin, xmax, ymin, ymax)` for each anchor box in each cell of the feature map.
        '''
        # Compute box width and height for each aspect ratio.
        config = self.config
        assert isinstance(config, SSDConfig)

        aspect_ratios = config.aspect_ratios[scale_idx]
        this_scale = config.scales[scale_idx]
        next_scale = config.scales[scale_idx+1]
        this_steps = config.steps[scale_idx]
        this_offsets = config.offsets[scale_idx]


        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(config.img_height, config.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []

        for ar in aspect_ratios:
            scale_use = this_scale
            if scale_idx == 0 and config.reduce_scale_of_lowest_square_box and ar == 1:
                scale_use = scale_use / 2.0

            box_width = scale_use * size * np.sqrt(ar)
            box_height = scale_use * size / np.sqrt(ar)
            wh_list.append((box_width, box_height))

        if config.two_boxes_for_ar1 and \
            (scale_idx > 0 or not config.reduce_boxes_in_lowest_layer):

            # Compute one slightly larger version using the geometric mean of this scale value and the next.
            box_height = box_width = np.sqrt(this_scale * next_scale) * size
            extra_box_for_ar1 = (box_width, box_height)
            if config.add_extra_ar1_box_at_end:   # tensorflow
                wh_list.append(extra_box_for_ar1)
            else:               # caffe
                wh_list.insert(1, extra_box_for_ar1) # insert after first square box

        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        if (this_steps is None):
            step_height = config.img_height / feature_map_size[0]
            step_width = config.img_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps
            else:
                raise ValueError('Invalid steps format')
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.

        if (this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets
            else:
                raise ValueError('Invalid offsets format')

        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy_vec = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx_vec = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx_vec, cy_vec)
        cx = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down
        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = cx  # Set cx
        boxes_tensor[:, :, :, 1] = cy  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:,0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:,1]  # Set h

        if config.box_xy_order == 'yx':   # tensorflow has coordinates in yx order
            boxes_tensor = convert_coordinates_xy(
                boxes_tensor, start_index=0, coords='centroids')

        # Convert `(cx, cy, w, h)` to `(xmin, ymin, xmax, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # If `limit_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if config.limit_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 2]]
            x_coords[x_coords >= config.img_width] = config.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 2]] = x_coords
            y_coords = boxes_tensor[:,:,:,[1, 3]]
            y_coords[y_coords >= config.img_height] = config.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1, 3]] = y_coords

        # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if config.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= config.img_width
            boxes_tensor[:, :, :, [1, 3]] /= config.img_height

        # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth.
        if config.anchor_coords == 'centroids':
            # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids')
        elif config.anchor_coords == 'minmax':
            # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax')

        if diagnostics:
            return boxes_tensor, (cy, cx), wh_list, (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor





class SSDBoxEncoder:
    '''
    A class to transform ground truth labels for object detection in images
    (2D bounding box coordinates and class labels) to the format required for
    training an SSD model, and to transform predictions of the SSD model back
    to the original format of the input labels.

    In the process of encoding ground truth labels, a template of anchor boxes
    is being built, which are subsequently matched to the ground truth boxes
    via an intersection-over-union threshold criterion.

    Arguments
    pos_iou_threshold (float, optional): The intersection-over-union similarity threshold that must be
        met in order to match a given ground truth box to a given anchor box. Defaults to 0.5.
    neg_iou_threshold (float, optional): The maximum allowed intersection-over-union similarity of an
        anchor box with any ground truth box to be labeled a negative (i.e. background) box. If an
        anchor box is neither a positive, nor a negative box, it will be ignored during training.
    '''

    def __init__(self,
                 ssd_config,
                 predictor_sizes,
                 pos_iou_threshold=None,   # default: 0.5 (from ssd_config)
                 neg_iou_threshold=None,   # default: 0.3 (from ssd_config)
                 ):


        assert isinstance(ssd_config, SSDConfig)

        predictor_sizes = np.array(predictor_sizes)
        if len(predictor_sizes.shape) == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        if pos_iou_threshold is None:
            pos_iou_threshold = ssd_config.encode_pos_iou_threshold

        if neg_iou_threshold is None:
            neg_iou_threshold = ssd_config.encode_neg_iou_threshold

        if neg_iou_threshold > pos_iou_threshold:
            raise ValueError("It cannot be `neg_iou_threshold > pos_iou_threshold`.")

        if len(predictor_sizes) != ssd_config.n_predictor_layers:
            raise ValueError("Number of predictor layers (%d) must match the number in the config (%d)" % (
                len(predictor_sizes), ssd_config.n_predictor_layers) )


        self.n_classes = ssd_config.n_classes + 1
        self.predictor_sizes = predictor_sizes
        self.ssd_config = ssd_config

        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold


        # Compute the number of boxes per cell.
        if (1 in ssd_config.aspect_ratios) and ssd_config.two_boxes_for_ar1:
            self.n_boxes = len(ssd_config.aspect_ratios) + 1
        else:
            self.n_boxes = len(ssd_config.aspect_ratios)
        # Compute the anchor boxes for all the predictor layers. We only have to do this once
        # as the anchor boxes depend only on the model configuration, not on the input data.
        # For each conv predictor layer (i.e. for each scale factor)the tensors for that lauer's
        # anchor boxes will have the shape `(feature_map_height, feature_map_width, n_boxes, 4)`.
        self.boxes_list = [] # This will contain the anchor boxes for each predictor layer.

        self.wh_list_diag = [] # Box widths and heights for each predictor layer
        self.steps_diag = [] # Horizontal and vertical distances between any two boxes for each predictor layer
        self.offsets_diag = [] # Offsets for each predictor layer
        self.centers_diag = [] # Anchor box center points as `(cy, cx)` for each predictor layer

        box_generator = SSDAnchorBoxGenerator(ssd_config)

        for scale_idx, predictor_size in enumerate(self.predictor_sizes):
            boxes, center, wh, step, offset = \
                box_generator.generate_boxes_for_scale(scale_idx,
                                                       feature_map_size=predictor_size,
                                                       diagnostics=True)
            print ('anchor boxes ', boxes.shape)
            self.boxes_list.append(boxes)
            self.wh_list_diag.append(wh)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)
            self.centers_diag.append(center)


    def generate_encode_template(self, batch_size, diagnostics=False):
        '''
        Produces an encoding template for the ground truth label tensor for a given batch.

        Note that all tensor creation, reshaping and concatenation operations performed in this function
        and the sub-functions it calls are identical to those performed inside the conv net model. This, of course,
        must be the case in order to preserve the spatial meaning of each box prediction, but it's useful to make
        yourself aware of this fact and why it is necessary.

        In other words, the boxes in `y_encoded` must have a specific order in order correspond to the right spatial
        positions and scales of the boxes predicted by the model. The sequence of operations here ensures that `y_encoded`
        has this specific form.

        Arguments:
            batch_size (int): The batch size.
            diagnostics (bool, optional): See the documentation for `generate_anchor_boxes()`. The diagnostic output
                here is similar, just for all predictor conv layers.

        Returns:
            A Numpy array of shape `(batch_size, #boxes, #classes + 12)`, the template into which to encode
            the ground truth labels for training. The last axis has length `#classes + 12` because the model
            output contains not only the 4 predicted box coordinate offsets, but also the 4 coordinates for
            the anchor boxes and the 4 variance values.
        '''
        # Tile the anchor boxes for each predictor layer across all batch items.
        boxes_batch = []
        for boxes in self.boxes_list:
            # Prepend one dimension to `self.boxes_list` to account for the batch size and tile it along.
            # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
            boxes = np.expand_dims(boxes.copy(), axis=0)

            # boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1),)
            boxes = np.repeat(boxes, batch_size,axis=0)

            # Now reshape the 5D tensor above into a 3D tensor of shape
            # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. The resulting
            # order of the tensor content will be identical to the order obtained from the reshaping operation
            # in our Keras model (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
            # use the same default index order, which is C-like index ordering)
            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)

        # Concatenate the anchor tensors from the individual layers to one.
        boxes_tensor = np.concatenate(boxes_batch, axis=1)

        config = self.ssd_config
        # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        #    It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # 4: Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
        #    contains the same 4 variance values for every position in the last axis.
        if config.add_variances_to_anchors:
            variances_tensor = np.zeros_like(boxes_tensor)
            variances_tensor += config.variances # Long live broadcasting
        else:
            # make empty array for concatenating with other arrays
            variances_tensor = np.zeros((batch_size, boxes_tensor.shape[1], 0))


        # 4: Concatenate the classes, boxes and variances tensors to get our final template for y_encoded. We also need
        #    another tensor of the shape of `boxes_tensor` as a space filler so that `y_encode_template` has the same
        #    shape as the SSD model output tensor. The content of this tensor is irrelevant, we'll just use
        #    `boxes_tensor` a second time. (This is where the anchors would go)
        #blank_offsets_tensor = boxes_tensor
        y_encode_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encode_template, self.centers_diag, self.wh_list_diag, self.steps_diag, self.offsets_diag
        else:
            return y_encode_template

    def encode_y(self, ground_truth_labels, diagnostics=False):
        '''
        Convert ground truth bounding box data into a suitable format to train an SSD model.

        For each image in the batch, each ground truth bounding box belonging to that image will be compared against each
        anchor box in a template with respect to their jaccard similarity. If the jaccard similarity is greater than
        or equal to the set threshold, the boxes will be matched, meaning that the ground truth box coordinates and class
        will be written to the the specific position of the matched anchor box in the template.

        The class for all anchor boxes for which there was no match with any ground truth box will be set to the
        background class, except for those anchor boxes whose IoU similarity with any ground truth box is higher than
        the set negative threshold (see the `neg_iou_threshold` argument in `__init__()`).

        Arguments:
            ground_truth_labels (list): A python list of length `batch_size` that contains one 2D Numpy array
                for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
                to the respective image, and the data for each ground truth bounding box has the format
                `(class_id, xmin, ymin, xmax, ymax)`, and `class_id` must be an integer greater than 0 for all boxes
                as class_id 0 is reserved for the background class.
            diagnostics (bool, optional): If `True`, not only the encoded ground truth tensor will be returned,
                but also a copy of it with anchor box coordinates in place of the ground truth coordinates.
                This can be very useful if you want to visualize which anchor boxes got matched to which ground truth
                boxes.

        Returns:
            `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)` that serves as the
            ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded. The four elements after the class vecotrs in
            the last axis are the box coordinates, the next four elements after that are just dummy elements, and
            the last four elements are the variances.
        '''

        # 1: Generate the template for y_encoded
        y_encode_template = self.generate_encode_template(batch_size=len(ground_truth_labels), diagnostics=False)
        y_encoded = np.copy(y_encode_template) # We'll write the ground truth box data to this array

        # 2: Match the boxes from `ground_truth_labels` to the anchor boxes in `y_encode_template`
        #    and for each matched box record the ground truth coordinates in `y_encoded`.
        #    Every time there is no match for a anchor box, record `class_id` 0 in `y_encoded` for that anchor box.
        config = self.ssd_config
        class_vector = np.eye(self.n_classes) # An identity matrix that we'll use as one-hot class vectors
        offset = config.n_classes+1
        #idx_coords = list(range(offset, offset + 4))
        coords_slice = slice(offset, offset + 4)
        class_coords_slice = slice(0, offset + 4)

        class_id, xmin, ymin, xmax, ymax = 0, 1, 2, 3, 4
        '''
        for i in range(y_encode_template.shape[0]): # For each batch item...
            available_boxes = np.ones((y_encode_template.shape[1])) # 1 for all anchor boxes that are not yet matched to a ground truth box, 0 otherwise
            negative_boxes = np.ones((y_encode_template.shape[1])) # 1 for all negative boxes, 0 otherwise
            for true_box in ground_truth_labels[i]: # For each ground truth box belonging to the current batch item...
                # We assume the ground truth labels are in 'corners' format.
                true_box = true_box.astype(np.float)
                xmin, ymin, xmax, ymax = true_box[1:]
                if abs(xmax - xmin < 0.001) or abs(ymax - ymin < 0.001): continue # Protect ourselves against bad ground truth data: boxes with width or height equal to zero

                if config.box_xy_order != 'xy':
                    #xmin, ymin,  xmax, ymax ==> ymin,xmin,  ymax,xmax
                    true_box = convert_coordinates_xy(true_box, start_index=1, coords="corners")

                if config.normalize_coords:
                    true_box[[1,3]] /= config.img_width # Normalize xmin and xmax to be within [0,1]
                    true_box[[2,4]] /= config.img_height # Normalize ymin and ymax to be within [0,1]

                # Create copy of the GT in anchor_coords format, for comparing IOU against other anchors
                true_box_anchor = true_box.copy()
                # GT Boxes are currently in 'corners' coordinates. Convert them to the desired coordinates, if different
                if config.anchor_coords != 'corners':  # convert from corners to anchor box format
                    conversion_anchor = 'corners2' + config.anchor_coords
                    true_box_anchor = convert_coordinates(true_box, start_index=1, conversion=conversion_anchor)

                # Create copy of the GT box in pred_coords format, for inserting into the encoded coordinates
                true_box_pred = true_box.copy()
                if config.pred_coords != 'corners':  # convert to whatever 'corners' coordinates
                    conversion_pred = 'corners2' + config.pred_coords
                    true_box_pred = convert_coordinates(true_box, start_index=1, conversion=conversion_pred)

                similarities = iou(labels[:, [xmin, ymin, xmax, ymax]], y_encoded[i, :, coords_slice], coords=self.coords,
                                   mode='outer_product', border_pixels=self.border_pixels)

                # First: Do bipartite matching, i.e. match each ground truth box to the one anchor box with the highest IoU.
                #        This ensures that each ground truth box will have at least one good match.

                # For each ground truth box, get the anchor box to match with it.
                bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)

                # Write the ground truth data to the matched anchor boxes.
                y_encoded[i, bipartite_matches, :-8] = labels_one_hot

                # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
                similarities[:, bipartite_matches] = 0





                similarities = iou(y_encode_template[i,:,offset:offset+4], true_box_anchor[1:], coords=config.anchor_coords) # The iou similarities for all anchor boxes
                negative_boxes[similarities >= self.neg_iou_threshold] = 0 # If a negative box gets an IoU match >= `self.neg_iou_threshold`, it's no longer a valid negative box
                similarities *= available_boxes # Filter out anchor boxes which aren't available anymore (i.e. already matched to a different ground truth box)
                available_and_thresh_met = np.copy(similarities)
                available_and_thresh_met[available_and_thresh_met < self.pos_iou_threshold] = 0 # Filter out anchor boxes which don't meet the iou threshold
                assign_indices = np.nonzero(available_and_thresh_met)[0] # Get the indices of the left-over anchor boxes to which we want to assign this ground truth box
                if len(assign_indices) > 0: # If we have any matches
                    y_encoded[i,assign_indices,:offset+4] = np.concatenate((class_vector[int(true_box[0])], true_box_pred[1:]), axis=0) # Write the ground truth box coordinates and class to all assigned anchor box positions. Remember that the last four elements of `y_encoded` are just dummy entries.
                    available_boxes[assign_indices] = 0 # Make the assigned anchor boxes unavailable for the next ground truth box
                else: # If we don't have any matches
                    best_match_index = np.argmax(similarities) # Get the index of the best iou match out of all available boxes
                    y_encoded[i,best_match_index,:offset+4] = np.concatenate((class_vector[int(true_box[0])], true_box_pred[1:]), axis=0) # Write the ground truth box coordinates and class to the best match anchor box position
                    available_boxes[best_match_index] = 0 # Make the assigned anchor box unavailable for the next ground truth box
                    negative_boxes[best_match_index] = 0 # The assigned anchor box is no longer a negative box
            # Set the classes of all remaining available anchor boxes to class zero
            background_class_indices = np.nonzero(negative_boxes)[0]
            y_encoded[i,background_class_indices,0] = 1
        '''

        ### New code
        y_encoded[:, :, config.background_id] = 1 # All boxes are background boxes by default.
        n_boxes = y_encoded.shape[1] # The total number of boxes that the model predicts per batch item
        class_vectors = np.eye(self.n_classes) # An identity matrix that we'll use as one-hot class vectors
        batch_size = len(ground_truth_labels)

        for i in range(batch_size): # For each batch item...

            if ground_truth_labels[i].size == 0: continue # If there is no ground truth for this batch item, there is nothing to match.
            labels = ground_truth_labels[i].astype(np.float) # The labels for this batch item

            # Check for degenerate ground truth bounding boxes before attempting any computations.
            if np.any(labels[:,[xmax]] - labels[:,[xmin]] <= 0) or np.any(labels[:,[ymax]] - labels[:,[ymin]] <= 0):
                raise ValueError("SSDInputEncoder detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, labels) +
                                 "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. Degenerate ground truth " +
                                 "bounding boxes will lead to NaN errors during the training.")

            if config.box_xy_order != 'xy':
                #xmin, ymin,  xmax, ymax ==> ymin,xmin,  ymax,xmax
                labels = convert_coordinates_xy(labels, start_index=1, coords="corners")

            # Maybe normalize the box coordinates.
            if config.normalize_coords:
                labels[:,[ymin,ymax]] /= config.img_width  # Normalize xmin and xmax to be within [0,1]
                labels[:,[xmin,xmax]] /= config.img_height  # Normalize ymin and ymax to be within [0,1]

            # Create copy of the GT in anchor_coords format, for comparing IOU against other anchors
            labels_anchor = labels.copy()
            # GT Boxes are currently in 'corners' coordinates. Convert them to the desired coordinates, if different
            if config.anchor_coords != 'corners':  # convert from corners to anchor box format
                conversion_anchor = 'corners2' + config.anchor_coords
                labels_anchor = convert_coordinates(labels, start_index=1, conversion=conversion_anchor)

            # Create copy of the GT box in pred_coords format, for inserting into the encoded coordinates
            labels_pred = labels.copy()
            if config.pred_coords != 'corners':  # convert to whatever 'corners' coordinates
                conversion_pred = 'corners2' + config.pred_coords
                labels_pred = convert_coordinates(labels, start_index=1, conversion=conversion_pred)

            # Maybe convert the box coordinate format.
            #if config.anchor_coords == 'centroids':
            #    labels = convert_coordinates(labels, start_index=xmin, conversion='corners2centroids', border_pixels=config.border_pixels)
            #elif config.anchor_coords == 'minmax':
            #    labels = convert_coordinates(labels, start_index=xmin, conversion='corners2minmax')

            classes_one_hot = class_vectors[labels_anchor[:, class_id].astype(np.int)] # The one-hot class IDs for the ground truth boxes of this batch item
            labels_one_hot = np.concatenate([classes_one_hot, labels_pred[:, [xmin,ymin,xmax,ymax]]], axis=-1) # The one-hot version of the labels for this batch item

            # Compute the IoU similarities between all anchor boxes and all ground truth boxes for this batch item.
            # This is a matrix of shape `(num_ground_truth_boxes, num_anchor_boxes)`.
            similarities = iou(labels_anchor[:,[xmin,ymin,xmax,ymax]], y_encoded[i,:,coords_slice], coords=config.pred_coords,
                               mode='outer_product', border_pixels=config.border_pixels)

            # First: Do bipartite matching, i.e. match each ground truth box to the one anchor box with the highest IoU.
            #        This ensures that each ground truth box will have at least one good match.

            # For each ground truth box, get the anchor box to match with it.
            bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)

            # Write the ground truth data to the matched anchor boxes.
            y_encoded[i, bipartite_matches, class_coords_slice] = labels_one_hot

            # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
            similarities[:, bipartite_matches] = 0

            # Second: Maybe do 'multi' matching, where each remaining anchor box will be matched to its most similar
            #         ground truth box with an IoU of at least `pos_iou_threshold`, or not matched if there is no
            #         such ground truth box.

            if config.matching_type == 'multi':

                # Get all matches that satisfy the IoU threshold.
                matches = match_multi(weight_matrix=similarities, threshold=self.pos_iou_threshold)

                # Write the ground truth data to the matched anchor boxes.
                y_encoded[i, matches[1], :-8] = labels_one_hot[matches[0]]

                # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
                similarities[:, matches[1]] = 0

            # Third: Now after the matching is done, all negative (background) anchor boxes that have
            #        an IoU of `neg_iou_limit` or more with any ground truth box will be set to netral,
            #        i.e. they will no longer be background boxes. These anchors are "too close" to a
            #        ground truth box to be valid background boxes.

            max_background_similarities = np.amax(similarities, axis=0)
            neutral_boxes = np.nonzero(max_background_similarities >= config.encode_neg_iou_threshold)[0]
            y_encoded[i, neutral_boxes, config.background_id] = 0



        # 3: Convert absolute box coordinates to offsets from the anchor boxes and normalize them
        check_with_old_code = False and config.add_variances_to_anchors
        y_raw = y_encoded.copy()
        y_encoded = encode_y_coordinates(y_raw, config)

        if check_with_old_code:
            y_encoded2 = y_raw.copy()
            assert config.pred_coords == config.anchor_coords
            if config.pred_coords == 'centroids':
                y_encoded2[:, :, [-12, -11]] -= y_encode_template[:, :,[-12, -11]]  # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
                y_encoded2[:, :, [-12, -11]] /= y_encode_template[:, :, [-10, -9]] * y_encode_template[:, :, [-4,-3]]  # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
                y_encoded2[:, :, [-10, -9]] /= y_encode_template[:, :, [-10, -9]]  # w(gt) / w(anchor), h(gt) / h(anchor)
                y_encoded2[:, :, [-10, -9]] = np.log(y_encoded2[:, :, [-10, -9]]) / y_encode_template[:, :, [-2,-1]]  # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
            elif  config.pred_coords == 'corners':
                y_encoded2[:, :, -12:-8] -= y_encode_template[:, :, -12:-8]  # (gt - anchor) for all four coordinates
                y_encoded2[:, :, [-12, -10]] /= np.expand_dims(y_encode_template[:, :, -10] - y_encode_template[:, :, -12],axis=-1)  # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
                y_encoded2[:, :, [-11, -9]] /= np.expand_dims(y_encode_template[:, :, -9] - y_encode_template[:, :, -11],axis=-1)  # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
                y_encoded2[:, :, -12:-8] /= y_encode_template[:, :,-4:]  # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively
            else:
                y_encoded2[:, :, -12:-8] -= y_encode_template[:, :, -12:-8]  # (gt - anchor) for all four coordinates
                y_encoded2[:, :, [-12, -11]] /= np.expand_dims(y_encode_template[:, :, -11] - y_encode_template[:, :, -12],axis=-1)  # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
                y_encoded2[:, :, [-10, -9]] /= np.expand_dims(y_encode_template[:, :, -9] - y_encode_template[:, :, -10],axis=-1)  # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
                y_encoded2[:, :, -12:-8] /= y_encode_template[:, :,-4:]  # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively

            #y_encoded = encode_y_coordinates(y_encoded_raw, y_encode_template, config)
            assert np.allclose(y_encoded, y_encoded2)

        do_check_encode_decode_consistency = True
        if do_check_encode_decode_consistency:
            y_encoded_1 = encode_y_coordinates(y_raw, config=config)
            y_encoded_1_d = decode_y_coordinates(y_encoded_1, config=config, remove_anchors=False)

            assert np.allclose(y_encoded_1_d, y_raw)


        if diagnostics:
            # Here we'll save the matched anchor boxes (i.e. anchor boxes that were matched to a ground truth box, but keeping the anchor box coordinates).
            y_matched_anchors = np.copy(y_encoded)
            y_matched_anchors[:,:,coords_slice] = 0 # Keeping the anchor box coordinates means setting the offsets to zero.
            return y_encoded, y_matched_anchors
        else:
            return y_encoded






def intersection_area(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    '''
    Computes the intersection areas of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the intersection areas for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation
    of the `mode` argument for details.

    Arguments:
        boxes1 (np.ndarray): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(m, 4)` containing the coordinates for `m` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes2`.
        boxes2 (np.ndarray): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes1`.
        coords (str, optional): The coordinate format in the input arrays. Can be either 'centroids' for the format
            `(cx, cy, w, h)`, 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format
            `(xmin, ymin, xmax, ymax)`.
        mode (str, optional): Can be one of 'outer_product' and 'element-wise'. In 'outer_product' mode, returns an
            `(m,n)` matrix with the intersection areas for all possible combinations of the `m` boxes in `boxes1` with the
            `n` boxes in `boxes2`. In 'element-wise' mode, returns a 1D array and the shapes of `boxes1` and `boxes2`
            must be boadcast-compatible. If both `boxes1` and `boxes2` have `m` boxes, then this returns an array of
            length `m` where the i-th position contains the intersection area of `boxes1[i]` with `boxes2[i]`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float containing values with
        the intersection areas of the boxes in `boxes1` and `boxes2`.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("All boxes must consist of 4 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(boxes1.shape[1], boxes2.shape[1]))
    if not mode in {'outer_product', 'element-wise'}:
        raise ValueError("`mode` must be one of 'outer_product' and 'element-wise', but got '{}'.",format(mode))

    # Convert the coordinates if necessary.
    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
    elif not (coords in {'minmax', 'corners'}):
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    m = boxes1.shape[0] # The number of boxes in `boxes1`
    n = boxes2.shape[0] # The number of boxes in `boxes2`

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin, ymin, xmax, ymax = 0, 1, 2, 3
    elif coords == 'minmax':
        xmin, xmax, ymin, ymax = 0, 1, 2, 3
    else:
        raise ValueError('Invalid coords: %s' % coords)

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1 # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1 # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.
    else:
        raise ValueError('Invalid border_pixels option: %s ' % border_pixels)

    # Compute the intersection areas.

    if mode == 'outer_product':

        # For all possible box combinations, get the greater xmin and ymin values.
        # This is a tensor of shape (m,n,2).
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:,[xmin,ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmin,ymin]], axis=0), reps=(m, 1, 1)))

        # For all possible box combinations, get the smaller xmax and ymax values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:,[xmax,ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmax,ymax]], axis=0), reps=(m, 1, 1)))

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,:,0] * side_lengths[:,:,1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:,[xmin,ymin]], boxes2[:,[xmin,ymin]])
        max_xy = np.minimum(boxes1[:,[xmax,ymax]], boxes2[:,[xmax,ymax]])

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,0] * side_lengths[:,1]

def intersection_area_(boxes1, boxes2, coords='corners', mode='outer_product', border_pixels='half'):
    '''
    The same as 'intersection_area()' but for internal use, i.e. without all the safety checks.
    '''

    m = boxes1.shape[0] # The number of boxes in `boxes1`
    n = boxes2.shape[0] # The number of boxes in `boxes2`

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin, ymin, xmax, ymax = 0, 1, 2, 3
    elif coords == 'minmax':
        xmin, xmax, ymin, ymax = 0, 1, 2, 3
    else:
        raise ValueError('Invalid coords option : %s' % coords)

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1 # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1 # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.
    else:
        raise ValueError('Invalid border pixels option : %s' % border_pixels)
    # Compute the intersection areas.

    if mode == 'outer_product':

        # For all possible box combinations, get the greater xmin and ymin values.
        # This is a tensor of shape (m,n,2).
        min_xy = np.maximum(np.expand_dims(boxes1[:,[xmin,ymin]], axis=1),
                            np.expand_dims(boxes2[:,[xmin,ymin]], axis=0))

        # For all possible box combinations, get the smaller xmax and ymax values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(np.expand_dims(boxes1[:,[xmax,ymax]], axis=1),
                            np.expand_dims(boxes2[:,[xmax,ymax]], axis=0))

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,:,0] * side_lengths[:,:,1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:,[xmin,ymin]], boxes2[:,[xmin,ymin]])
        max_xy = np.minimum(boxes1[:,[xmax,ymax]], boxes2[:,[xmax,ymax]])

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,0] * side_lengths[:,1]


def intersection_area_orig_(boxes1, boxes2, coords='corners', mode='outer_product', border_pixels='half'):
    """
    The same as 'intersection_area()' but for internal use, i.e. without all the safety checks.
    """

    m = boxes1.shape[0] # The number of boxes in `boxes1`
    n = boxes2.shape[0] # The number of boxes in `boxes2`

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin, ymin, xmax, ymax = 0, 1, 2, 3
    elif coords == 'minmax':
        xmin, xmax, ymin, ymax = 0, 1, 2, 3
    else:
        raise ValueError('Invalid coords: %s' % coords)

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1 # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1 # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.
    else:
        raise ValueError('invalid value for border pixels: %s' % border_pixels)
    # Compute the intersection areas.

    if mode == 'outer_product':

        # For all possible box combinations, get the greater xmin and ymin values.
        # This is a tensor of shape (m,n,2).
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:,[xmin,ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmin,ymin]], axis=0), reps=(m, 1, 1)))

        # For all possible box combinations, get the smaller xmax and ymax values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:,[xmax,ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmax,ymax]], axis=0), reps=(m, 1, 1)))

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,:,0] * side_lengths[:,:,1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:,[xmin,ymin]], boxes2[:,[xmin,ymin]])
        max_xy = np.minimum(boxes1[:,[xmax,ymax]], boxes2[:,[xmax,ymax]])

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,0] * side_lengths[:,1]


def iou(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    """
    Computes the intersection-over-union similarity (also known as Jaccard similarity)
    of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the IoUs for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation
    of the `mode` argument for details.

    Arguments:
        boxes1 (np.ndarray): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(m, 4)` containing the coordinates for `m` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes2`.
        boxes2 (np.ndarray): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes1`.
        coords (str, optional): The coordinate format in the input arrays. Can be either 'centroids' for the format
            `(cx, cy, w, h)`, 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format
            `(xmin, ymin, xmax, ymax)`.
        mode (str, optional): Can be one of 'outer_product' and 'element-wise'. In 'outer_product' mode, returns an
            `(m,n)` matrix with the IoU overlaps for all possible combinations of the `m` boxes in `boxes1` with the
            `n` boxes in `boxes2`. In 'element-wise' mode, returns a 1D array and the shapes of `boxes1` and `boxes2`
            must be boadcast-compatible. If both `boxes1` and `boxes2` have `m` boxes, then this returns an array of
            length `m` where the i-th position contains the IoU overlap of `boxes1[i]` with `boxes2[i]`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float containing values in [0,1],
        the Jaccard similarity of the boxes in `boxes1` and `boxes2`. 0 means there is no overlap between two given
        boxes, 1 means their coordinates are identical.
    """

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2:
        raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2:
        raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("All boxes must consist of 4 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(boxes1.shape[1], boxes2.shape[1]))
    if not mode in {'outer_product', 'element-wise'}:
        raise ValueError("`mode` must be one of 'outer_product' and 'element-wise', but got '{}'.".format(mode))

    # Convert the coordinates if necessary.
    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
    elif not (coords in {'minmax', 'corners'}):
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    # Compute the IoU.

    # Compute the interesection areas.

    intersection_areas = intersection_area_(boxes1, boxes2, coords=coords, mode=mode)

    m = boxes1.shape[0] # The number of boxes in `boxes1`
    n = boxes2.shape[0] # The number of boxes in `boxes2`

    # Compute the union areas.

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin, ymin, xmax, ymax = 0, 1, 2, 3
    elif coords == 'minmax':
        xmin, xmax, ymin, ymax = 0, 1, 2, 3
    else:
        raise ValueError('Invalid coords argument')


    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1 # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1 # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.
    else:
        raise ValueError('Invalid border_pixels argument')

    if mode == 'outer_product':
        boxes1_areas = np.expand_dims((boxes1[:,xmax] - boxes1[:,xmin] + d) * (boxes1[:,ymax] - boxes1[:,ymin] + d), axis=1)
        boxes2_areas = np.expand_dims((boxes2[:,xmax] - boxes2[:,xmin] + d) * (boxes2[:,ymax] - boxes2[:,ymin] + d), axis=0)

    elif mode == 'element-wise':
        boxes1_areas = (boxes1[:,xmax] - boxes1[:,xmin] + d) * (boxes1[:,ymax] - boxes1[:,ymin] + d)
        boxes2_areas = (boxes2[:,xmax] - boxes2[:,xmin] + d) * (boxes2[:,ymax] - boxes2[:,ymin] + d)

    else:
        raise ValueError('invalid mode')

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas



def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    """
    Convert coordinates for axis-aligned 2D boxes between two coordinate formats.

    Creates a copy of `tensor`, i.e. does not operate in place. Currently there are
    three supported coordinate formats that can be converted from and to each other:
        1) (xmin, xmax, ymin, ymax) - the 'minmax' format
        2) (xmin, ymin, xmax, ymax) - the 'corners' format
        2) (cx, cy, w, h) - the 'centroids' format

    Note that converting from one of the supported formats to another and back is
    an identity operation up to possible rounding errors for integer tensors.

    Arguments:
        tensor (np.ndarray): A Numpy nD array containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        conversion (str, optional): The conversion direction. Can be 'minmax2centroids',
            'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners',
            or 'corners2minmax'.

    Returns:
        A Numpy nD array, a copy of the input tensor with the converted coordinates
        in place of the original coordinates and the unaltered elements of the original
        tensor elsewhere.
    """
    if isinstance(tensor, tf.Tensor):  # tensorflow/keras tensors
        copy, concat= tf.identity, tf.concat

    else:                              # numpy arrays
        copy, concat = np.copy, np.concatenate
        #assert isinstance(tensor, np.float)
        #assert np.issubdtype(tensor, np.floating)

    size_dim = tensor.shape[-1]

    ind = start_index
    inds = [ slice(ind+i, ind+i+1) for i in range(4) ]
    pre_slice  = slice(0, start_index)
    post_slice = slice(   start_index + 4, None)  # entries after coords
    if np.mod(start_index, size_dim) == size_dim - 4:  # last 4 elements:
        inds[-1] = slice(ind+3,None) # fix last slice so is not empty
        post_slice = slice(-1,-1)

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    #tensor1 = cast( copy(tensor), np.float)
    tensor_pre = tensor[..., pre_slice]
    tensor_post = tensor[..., post_slice]

    if conversion == 'minmax2centroids':
        s_xmin, s_xmax, s_ymin, s_ymax = inds
        cx = (tensor[..., s_xmin] + tensor[..., s_xmax]) / 2.0  # Set cx
        cy = (tensor[..., s_ymin] + tensor[..., s_ymax]) / 2.0  # Set cy
        w = tensor[..., s_xmax] - tensor[..., s_xmin] + d # Set w
        h = tensor[..., s_ymax] - tensor[..., s_ymin] + d  # Set h
        tensor_coords = concat((cx, cy, w, h), axis=-1)
    elif conversion == 'centroids2minmax':
        s_cx, s_cy, s_w, s_h = inds
        xmin = tensor[..., s_cx] - tensor[..., s_w] / 2.0  # Set xmin
        xmax = tensor[..., s_cx] + tensor[..., s_w] / 2.0  # Set xmax
        ymin = tensor[..., s_cy] - tensor[..., s_h] / 2.0  # Set ymin
        ymax = tensor[..., s_cy] + tensor[..., s_h] / 2.0  # Set ymax
        tensor_coords = concat((xmin, xmax, ymin, ymax), axis=-1)
    elif conversion == 'corners2centroids':
        s_xmin, s_ymin, s_xmax, s_ymax = inds
        cx = (tensor[..., s_xmin] + tensor[..., s_xmax]) / 2.0  # Set cx
        cy = (tensor[..., s_ymin] + tensor[..., s_ymax]) / 2.0  # Set cy
        w = tensor[..., s_xmax] - tensor[..., s_xmin] + d  # Set w
        h = tensor[..., s_ymax] - tensor[..., s_ymin] + d  # Set h
        tensor_coords = concat((cx, cy, w, h), axis=-1)
    elif conversion == 'centroids2corners':
        s_cx, s_cy, s_w, s_h = inds
        xmin = tensor[..., s_cx] - tensor[..., s_w] / 2.0  # Set xmin
        ymin = tensor[..., s_cy] - tensor[..., s_h] / 2.0  # Set ymin
        xmax = tensor[..., s_cx] + tensor[..., s_w] / 2.0  # Set xmax
        ymax = tensor[..., s_cy] + tensor[..., s_h] / 2.0  # Set ymax
        tensor_coords = concat((xmin, ymin, xmax, ymax), axis=-1)
    elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
        # corners2minmax:  xmin, ymin, xmax, ymax --> xmin, xmax, ymin, ymax
        # minmax2corners:  xmin, xmax, ymin, ymax --> xmin, ymin, xmax, ymax
        # switch around ind+1 and ind+2  (ymin & xmax)
        ind0, ind1, ind2, ind3 = inds
        tensor_coords = concat(
            (tensor[..., ind0], tensor[..., ind2 ],
             tensor[..., ind1], tensor[..., ind3 ]), axis=-1)

    else:
        raise ValueError(
            "Unexpected conversion value. Supported values are 'minmax2centroids', 'centroids2minmax',"
            " 'corners2centroids', 'centroids2corners', 'minmax2corners', and 'corners2minmax'.")

    tensor1 = concat((tensor_pre, tensor_coords, tensor_post), axis=-1)

    return tensor1



def convert_coordinates_xy(tensor, start_index, coords):
    """
    Reverses order of coordinates (e.g. x-y ==> y-x, or y-x ==> x-y)

    Creates a copy of `tensor`, i.e. does not operate in place. Currently there are
    three supported coordinate formats that can be converted between xy and yx:
        1) (xmin, xmax, ymin, ymax) - the 'minmax' format
        2) (xmin, ymin, xmax, ymax) - the 'corners' format
        2) (cx, cy, w, h) - the 'centroids' format

    Arguments:
        tensor (np.ndarray): A Numpy nD array containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        coords (str, optional): The coordinates format. Can be 'centroids',
            'minmax', or  'corners',

    Returns:
        A Numpy nD array, a copy of the input tensor with the converted coordinates
        in place of the original coordinates and the unaltered elements of the original
        tensor elsewhere.
    """
    if isinstance(tensor, tf.Tensor):  # tensorflow/keras tensors
        concat= tf.concat
    else:                              # numpy arrays
        concat = np.concatenate

    ind = start_index
    inds = [slice(ind+i, ind+i+1) for i in range(4) ]
    tensor_pre = tensor[..., :start_index]
    tensor_post = tensor[..., start_index + 4:]

    if coords in  ['corners', 'centroids']:
        # Convert x1,y1, x2,y2  ==> y1,x1, y2,x2
        # or:     y1,x1, y2,x2  ==> x1,y1, x2,y2
        i_x1, i_y1, i_x2, i_y2 = inds
        x1 = tensor[..., i_x1]
        y1 = tensor[..., i_y1]
        x2 = tensor[..., i_x2]
        y2 = tensor[..., i_y2]
        tensor_coords = concat((y1, x1, y2, x2), axis=-1)
    elif coords == 'minmax':
        # Convert x1,x2, y1,y2  ==> y1,y2, x1,x2
        # or:     y1,y2, x1,x2  ==> x1,x2, y1,y2
        i_x1, i_x2, i_y1, i_y2 = inds
        x1 = tensor[..., i_x1]
        y1 = tensor[..., i_y1]
        x2 = tensor[..., i_x2]
        y2 = tensor[..., i_y2]
        tensor_coords = concat((y1, y2, x1, x2), axis=-1)
    else:
        raise ValueError(
            "Unexpected coords value. Supported values are 'centroids', 'minmax', 'corners'.")

    tensor1 = concat((tensor_pre, tensor_coords, tensor_post), axis=-1)

    return tensor1



def convert_coordinates2(tensor, start_index, conversion):
    """
    A matrix multiplication implementation of `convert_coordinates()`.
    Supports only conversion between the 'centroids' and 'minmax' formats.

    This function is marginally slower on average than `convert_coordinates()`,
    probably because it involves more (unnecessary) arithmetic operations (unnecessary
    because the two matrices are sparse).

    For details please refer to the documentation of `convert_coordinates()`.
    """
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        M = np.array([[0.5, 0. , -1.,  0.],
                      [0.5, 0. ,  1.,  0.],
                      [0. , 0.5,  0., -1.],
                      [0. , 0.5,  0.,  1.]])
        tensor1[..., ind:ind+4] = np.dot(tensor1[..., ind:ind+4], M)
    elif conversion == 'centroids2minmax':
        M = np.array([[ 1. , 1. ,  0. , 0. ],
                      [ 0. , 0. ,  1. , 1. ],
                      [-0.5, 0.5,  0. , 0. ],
                      [ 0. , 0. , -0.5, 0.5]]) # The multiplicative inverse of the matrix above
        tensor1[..., ind:ind+4] = np.dot(tensor1[..., ind:ind+4], M)
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")

    return tensor1

def greedy_nms(y_pred_decoded, iou_threshold=0.45, coords='corners'):
    """
    Perform greedy non-maximum suppression on the input boxes.

    Greedy NMS works by selecting the box with the highest score and
    removing all boxes around it that are too close to it measured by IoU-similarity.
    Out of the boxes that are left over, once again the one with the highest
    score is selected and so on, until no boxes with too much overlap are left.

    This is a basic, straight-forward NMS algorithm that is relatively efficient,
    but it has a number of downsides. One of those downsides is that the box with
    the highest score might not always be the box with the best fit to the object.
    There are more sophisticated NMS techniques like [this one](https://lirias.kuleuven.be/bitstream/123456789/506283/1/3924_postprint.pdf)
    that use a combination of nearby boxes, but in general there will probably
    always be a trade-off between speed and quality for any given NMS technique.

    Arguments:
        y_pred_decoded (list): A batch of decoded predictions. For a given batch size `n` this
            is a list of length `n` where each list element is a 2D Numpy array.
            For a batch item with `k` predicted boxes this 2D Numpy array has
            shape `(k, 6)`, where each row contains the coordinates of the respective
            box in the format `[class_id, score, xmin, xmax, ymin, ymax]`.
            Technically, the number of columns doesn't have to be 6, it can be
            arbitrary as long as the first four elements of each row are
            `xmin`, `xmax`, `ymin`, `ymax` (in this order) and the last element
            is the score assigned to the prediction. Note that this function is
            agnostic to the scale of the score or what it represents.
        iou_threshold (float, optional): All boxes with a Jaccard similarity of
            greater than `iou_threshold` with a locally maximal box will be removed
            from the set of predictions, where 'maximal' refers to the box score.
            Defaults to 0.45 following the paper.
        coords (str, optional): The coordinate format of `y_pred_decoded`.
            Can be one of the formats supported by `iou()`. Defaults to 'corners'.

    Returns:
        The predictions after removing non-maxima. The format is the same as the input format.
    """
    y_pred_decoded_nms = []
    for batch_item in y_pred_decoded: # For the labels of each batch item...
        boxes_left = np.copy(batch_item)
        maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
        while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
            maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
            maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
            maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
            boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
            if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
            similarities = iou(boxes_left[:,2:], maximum_box[2:], coords=coords) # ...compare (IoU) the other left over boxes to the maximum box...
            boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
        y_pred_decoded_nms.append(np.array(maxima))

    return y_pred_decoded_nms

def _greedy_nms(predictions, iou_threshold=0.45, coords='corners', border_pixels='half'):
    """
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function for per-class NMS in `decode_y()`.
    """
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,0]) # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,1:], maximum_box[1:], coords=coords, mode='element-wise', border_pixels=border_pixels) # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
    return np.array(maxima)



def _greedy_nms_v2(predictions, iou_threshold=0.45, coords='corners', return_idxs=False):
    """
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function for per-class NMS in `decode_y()`.
    """
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    maxima_idxs = []
    n_tot = boxes_left.shape[0]
    n_remain = n_tot
    tf_remain = np.ones(n_tot, dtype=np.bool)
    while n_remain > 0: # While there are still boxes left to compare...

        maximum_index_remain = np.argmax(boxes_left[tf_remain,0]) # ...get the index of the next box with the highest confidence...
        maximum_index = tf_remain.nonzero()[0][maximum_index_remain]
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        maxima_idxs.append(maximum_index)
        #boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        tf_remain[maximum_index] = False

        if np.count_nonzero(tf_remain) == 0:
            break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[tf_remain,1:], maximum_box[1:], coords=coords)[:,0] # ...compare (IoU) the other left over boxes to the maximum box...
        idx_too_much_overlap = tf_remain.nonzero()[0][similarities > iou_threshold]
        tf_remain[idx_too_much_overlap] = False
        n_remain = np.count_nonzero(tf_remain)
        #boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
    return np.array(maxima), np.array(maxima_idxs)

def _greedy_nms2(predictions, iou_threshold=0.45, coords='corners'):
    """
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function in `decode_y2()`.
    """
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,2:], maximum_box[2:], coords=coords) # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
    return np.array(maxima)




def encode_y_coordinates(y_raw, config):
    # Encodes the coordinates in y_raw to the encoded format used for training
    # (e.g. makes size relative to anchor box size, divide by variances, etc)
    # an SSDConfig object (config) is needed so that we know details about
    # y_raw (e.g. number of classes, coordinate system, whether the anchors
    #  contain the variances or not)

    n_classes = config.n_classes + 1
    n_variances = len(config.variances) if config.add_variances_to_anchors else 0
    n_box_coords = 4
    n_anchor_coords = 4
    box_size = n_classes + n_variances + n_box_coords + n_anchor_coords

    assert box_size == y_raw.shape[2], "Mismatch in expected size (%d expected vs %d actual)" % (
        box_size, y_raw.shape[2])

    offset = config.n_classes + 1
    if isinstance(y_raw, tf.Tensor):
        log, expand_dims, copy, concat, array = tf.log, tf.expand_dims, tf.identity, tf.concat, tf.constant
    else:
        log, expand_dims, copy, concat, array = np.log, np.expand_dims, np.copy, np.concatenate, np.array

    n_classes = config.n_classes + 1

    y_raw_classes =     y_raw[:, :, 0 : n_classes]
    y_raw_coords = copy(y_raw[:, :, n_classes : n_classes + n_box_coords])
    y_raw_anchors =     y_raw[:, :, n_classes + n_box_coords :]

    # Convert anchor coordinates format to pred format if necessary
    if config.anchor_coords == config.pred_coords:
        y_anchors = y_raw_anchors
    else:
        conversion = config.anchor_coords + '2' + config.pred_coords
        y_anchors = convert_coordinates(
            y_raw_anchors, start_index=0, conversion=conversion)

    if config.add_variances_to_anchors:
        idx_variances = slice(box_size-4, box_size)
        variances = y_raw[:, :, idx_variances]
    else:
        variances = array(config.variances, dtype=np.float32)


    if config.pred_coords == 'centroids':  # `(cx, cy, w, h)`
        idx_cxy = slice(0, 2)
        idx_wh = slice(2, 4)

        y_coords_cxy = y_raw_coords[:, :, idx_cxy]
        y_coords_wh = y_raw_coords[:, :, idx_wh]

        # make target centroid relative to anchor centroid:
        # cx(gt) - cx(anchor),          cy(gt) - cy(anchor)
        y_coords_cxy -= y_anchors[:, :, idx_cxy]

        # make centroid relative to size of box
        # ((cx(gt) - cx(anchor)) / w(anchor)) / cx_variance,          (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
        y_coords_cxy /= y_anchors[:, :, idx_wh]

        # convert size (width/height) of box to the log ratio of its size to anchor size
        y_coords_wh /= y_anchors[:, :, idx_wh]  # w(gt) / w(anchor), h(gt) / h(anchor)
        y_coords_wh = log(y_coords_wh)  # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)

        y_encoded_coords = concat((y_coords_cxy, y_coords_wh), axis=-1)

        # Scale coordinates up (dividing by variances)
        y_encoded_coords /= variances

    else:
        idxs = (0, 1, 2, 3)
        if config.pred_coords == 'corners':  # (xmin, ymin, xmax, ymax)
            idx_xmin, idx_ymin, idx_xmax, idx_ymax = idxs
        elif config.pred_coords == 'minmax':  # minmax:  `(xmin, xmax, ymin, ymax)`
            idx_xmin, idx_xmax, idx_ymin, idx_ymax = idxs
        else:
            raise ValueError('unrecognized coordinates: %s' % config.pred_coords)

        # idx_xmin_anchor, idx_ymin_anchor, idx_xmax_anchor, idx_ymax_anchor = (
        #    idx_xmin+4, idx_ymin+4, idx_xmax+4, idx_ymax+4)

        # Make size coordinates relative to anchor coordinates
        # (gt - anchor) for all four coordinates
        y_encoded_coords = y_raw_coords - y_anchors #[:, :, idx_coords_anchors]

        # Divide by height/width to make box dimensions relative to anchor box width:
        #  (gt - anchor) / anchor for all four coordinates
        # Note, w and h are assuming (without loss of generality) box_xy_order = 'xy'
        # If not, then the meanings of w and h are reversed, but the computation is the same
        w = expand_dims( y_anchors[:, :, idx_xmax] - y_anchors[:, :, idx_xmin], axis=-1)
        h = expand_dims( y_anchors[:, :, idx_ymax] - y_anchors[:, :, idx_ymin], axis=-1)
        if config.pred_coords == 'corners':  # (xmin, ymin, xmax, ymax)
            box_fullsize = concat((w, h, w, h), axis=-1)
        else:  # minmax:  `(xmin, xmax, ymin, ymax)`
            box_fullsize = concat((w, w, h, h), axis=-1)
        y_encoded_coords /= box_fullsize

        # Scale coordinates up (dividing by variances)
        y_encoded_coords /= variances


    y_encoded = concat( (y_raw_classes, y_encoded_coords, y_raw_anchors), axis=-1)


    return y_encoded



def decode_y_coordinates(y_pred, config, remove_anchors=True, final_coords=None):
    # Decodes the coordinates in ypred from the encoded format used for training
    # to the bounding box coordinates useful for evaluation
    # remove_anchors: as a convenience, the anchors (+variances) which are
    #   appended to each box can be discarded
    # final_coords: optional conversion from the native coordinates of y_pred
    #   (specified in the config) to a different coordinate system
    #   (e.g. centroids, corners, minmax)


    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    n_classes = config.n_classes + 1
    n_variances = len(config.variances) if config.add_variances_to_anchors else 0
    n_box_coords = 4
    n_anchor_coords = 4
    box_size = n_classes + n_variances + n_box_coords + n_anchor_coords

    if isinstance(y_pred, tf.Tensor):
        exp, expand_dims, copy, concat, array = tf.exp, tf.expand_dims, tf.identity, tf.concat, tf.constant
    else:
        assert y_pred.shape[2] == box_size, "Mismatch in expected size (%d expected vs %d actual)" % (
            box_size, y_pred.shape[2])
        exp, expand_dims, copy, concat, array = np.exp, np.expand_dims, np.copy, np.concatenate, np.array

    y_pred_classes = y_pred[:,:, :n_classes]
    y_pred_coords = copy(y_pred[:,:,n_classes : n_classes + n_box_coords])
    y_pred_anchors = y_pred[:, :, n_classes + n_box_coords :]

    # Convert anchor coordinates format to pred format if necessary
    if config.anchor_coords == config.pred_coords:
        y_anchors = y_pred_anchors
    else:
        conversion = config.anchor_coords + '2' + config.pred_coords
        y_anchors = convert_coordinates(
            y_pred_anchors, start_index=0, conversion=conversion)

    if config.add_variances_to_anchors:
        idx_variances = slice(box_size-4, box_size)
        variances = y_pred[:, :, idx_variances]
    else:
        variances = array( config.variances, dtype=np.float32)

    #are_equal = lambda x, y: np.allclose(np.mod(x, box_size), np.mod(y, box_size))
    if config.pred_coords == 'centroids':
        idx_cxy = slice(0, 2)  # relative to start of box coords
        idx_wh = slice(2, 4)

        # Undo division by variance.
        # divide all coordinates by variance (ie. scale up by 1/var)
        y_pred_coords *= variances

        # Split up coordinates into centroids (cxy) & size (wh)
        y_pred_coords_cxy = y_pred_coords[:, :, idx_cxy]
        y_pred_coords_wh = y_pred_coords[:, :, idx_wh]

        # Undo scaling cx/cy relative to box height/width.
        # x:: (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred),  y::  (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_coords_cxy *= y_anchors[:,:,idx_wh]

        # Undo offset cx/cy relative to anchor cx/cy
        # x:: delta_cx(pred) + cx(anchor) == cx(pred),   y:: delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_coords_cxy += y_anchors[:,:,idx_cxy] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)


        # Undo conversion of size (width/height) of box to the log ratio of itself to anchor
        # x:: exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor),    y:: exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_coords_wh = exp(y_pred_coords_wh)
        # x:: (w(pred) / w(anchor)) * w(anchor) == w(pred), y:: (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_coords_wh *= y_anchors[:,:,idx_wh]

        y_pred_coords = concat((y_pred_coords_cxy, y_pred_coords_wh), axis=-1)


    else:
        idx_anchors = slice(0, 4)
        if config.pred_coords == 'corners':  # (xmin, ymin, xmax, ymax)
            idx_xmin, idx_ymin, idx_xmax, idx_ymax = (0, 1, 2, 3)
        elif config.pred_coords == 'minmax':  # minmax:  `(xmin, xmax, ymin, ymax)`
            idx_xmin, idx_xmax, idx_ymin, idx_ymax = (0, 1, 2, 3)
        else:
            raise ValueError("Unexpected value for `coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

        # Undo division by variance.
        # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_coords *= variances

        # Undo making box width and box height relative to anchor box width:
        # x:: delta_xmin[max](pred) / w(anchor) * w(anchor)  == delta_xmin[max](pred),
        # y:: delta_ymin[max](pred) / h(anchor) * h(anchor) == delta_ymin[max](pred),
        w = expand_dims(y_pred_anchors[:, :, idx_xmax] - y_pred_anchors[:, :, idx_xmin], axis=-1)
        h = expand_dims(y_pred_anchors[:, :, idx_ymax] - y_pred_anchors[:, :, idx_ymin], axis=-1)
        if config.pred_coords == 'corners':  # (xmin, ymin, xmax, ymax)
            box_fullsize = concat((w, h, w, h), axis=-1)
        else:  # minmax:  `(xmin, xmax, ymin, ymax)`
            box_fullsize = concat((w, w, h, h), axis=-1)
        y_pred_coords *= box_fullsize

        # Undo making size coordinates relative to anchor coordinates
        #  delta_x[y] + x[y]_anchor = x[y]  for all four coordinates
        y_pred_coords += y_pred_anchors[:,:,idx_anchors] # delta(pred) + anchor == pred for all four coordinates

    # After decoding coordinates, can do one final coordinate format conversion:
    if final_coords is not None and config.pred_coords != final_coords:
        conversion = config.pred_coords + '2' + final_coords
        y_pred_coords = convert_coordinates(y_pred_coords, start_index=0, conversion=conversion)

    if remove_anchors:
        # Slice out the classes and the four offsets, (throw away the anchor coordinates and variances),
        # resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`
        y_pred_decoded = concat((y_pred_classes, y_pred_coords), axis=-1)
    else:
        y_pred_decoded = concat( (y_pred_classes, y_pred_coords, y_pred_anchors), axis=-1)


    return y_pred_decoded


def decode_y_orig_repo(y_pred,
             confidence_thresh=0.01,
             iou_threshold=0.45,
             top_k=200,
             input_coords='centroids',
             normalize_coords=True,
             img_height=None,
             img_width=None,
                       to_dict=False):
    """
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `enconde_y()` takes as input).

    After the decoding, two stages of prediction filtering are performed for each class individually:
    First confidence thresholding, then greedy non-maximum suppression. The filtering results for all
    classes are concatenated and the `top_k` overall highest confidence results constitute the final
    predictions for a given batch item. This procedure follows the original Caffe implementation.
    For a slightly different and more efficient alternative to decode raw model output that performs
    non-maximum suppresion globally instead of per class, see `decode_y2()` below.

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage. Defaults to 0.01, following the paper.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score. Defaults to 0.45 following the paper.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage. Defaults to 200, following the paper.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`. Defaults to 'centroids'.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`. Defaults to `False`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, ymin, xmax, ymax]`.
    """
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError(
            "If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(
                img_height, img_width))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    y_pred_decoded_raw = np.copy(y_pred[:, :,
                                 :-8])  # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

    if input_coords == 'centroids':
        y_pred_decoded_raw[:, :, [-2, -1]] = np.exp(y_pred_decoded_raw[:, :, [-2, -1]] * y_pred[:, :, [-2,
                                                                                                       -1]])  # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_decoded_raw[:, :, [-2, -1]] *= y_pred[:, :, [-6,
                                                            -5]]  # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_decoded_raw[:, :, [-4, -3]] *= y_pred[:, :, [-4, -3]] * y_pred[:, :, [-6,
                                                                                     -5]]  # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_decoded_raw[:, :, [-4, -3]] += y_pred[:, :, [-8,
                                                            -7]]  # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:, :, -4:] *= y_pred[:, :,
                                         -4:]  # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:, :, [-4, -3]] *= np.expand_dims(y_pred[:, :, -7] - y_pred[:, :, -8],
                                                             axis=-1)  # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:, :, [-2, -1]] *= np.expand_dims(y_pred[:, :, -5] - y_pred[:, :, -6],
                                                             axis=-1)  # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:, :, -4:] += y_pred[:, :, -8:-4]  # delta(pred) + anchor == pred for all four coordinates
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        y_pred_decoded_raw[:, :, -4:] *= y_pred[:, :,
                                         -4:]  # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:, :, [-4, -2]] *= np.expand_dims(y_pred[:, :, -6] - y_pred[:, :, -8],
                                                             axis=-1)  # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:, :, [-3, -1]] *= np.expand_dims(y_pred[:, :, -5] - y_pred[:, :, -7],
                                                             axis=-1)  # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:, :, -4:] += y_pred[:, :, -8:-4]  # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError(
            "Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

    if normalize_coords:
        y_pred_decoded_raw[:, :, [-4, -2]] *= img_width  # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:, :, [-3, -1]] *= img_height  # Convert ymin, ymax back to absolute coordinates

    # 3: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[
                    -1] - 4  # The number of classes is the length of the last axis minus the four box coordinates

    y_pred_decoded = []  # Store the final predictions in this list

    for batch_item in y_pred_decoded_raw:  # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = []  # Store the final predictions for this batch item here
        for class_id in range(1, n_classes):  # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:, [class_id, -4, -3, -2,
                                          -1]]  # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
            threshold_met = single_class[single_class[:,
                                         0] > confidence_thresh]  # ...keep only those boxes with a confidence above the set threshold.
            if threshold_met.shape[0] > 0:  # If any boxes made the threshold...
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold,
                                     coords='corners')  # ...perform NMS on them.
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[
                    1] + 1))  # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:, 0] = class_id  # Write the class ID to the first column...
                maxima_output[:, 1:] = maxima  # ...and write the maxima to the other columns...
                pred.append(
                    maxima_output)  # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        if pred:  # If there are any predictions left after confidence-thresholding...
            pred = np.concatenate(pred, axis=0)
            if pred.shape[
                0] > top_k:  # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                top_k_indices = np.argpartition(pred[:, 1], kth=pred.shape[0] - top_k, axis=0)[
                                pred.shape[0] - top_k:]  # ...get the indices of the `top_k` highest-score maxima...
                pred = pred[top_k_indices]  # ...and keep only those entries of `pred`...
        y_pred_decoded.append(
            pred)  # ...and now that we're done, append the array of final predictions for this batch item to the output list

    if to_dict:
        y_pred_decoded_dicts = [boxarray2dict(yp) for yp in y_pred_decoded]
        return y_pred_decoded_dicts
    else:
        return y_pred_decoded


def decode_detections_ssd_repo(y_pred,
                      confidence_thresh=0.01,
                      iou_threshold=0.45,
                      top_k=200,
                      input_coords='centroids',
                      normalize_coords=True,
                      img_height=None,
                      img_width=None,
                      border_pixels='half'):
    """
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `SSDInputEncoder` takes as input).

    After the decoding, two stages of prediction filtering are performed for each class individually:
    First confidence thresholding, then greedy non-maximum suppression. The filtering results for all
    classes are concatenated and the `top_k` overall highest confidence results constitute the final
    predictions for a given batch item. This procedure follows the original Caffe implementation.
    For a slightly different and more efficient alternative to decode raw model output that performs
    non-maximum suppresion globally instead of per class, see `decode_detections_fast()` below.

    Arguments:
        y_pred (np.ndarray): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, ymin, xmax, ymax]`.
    """
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    y_pred_decoded_raw = np.copy(y_pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

    if input_coords == 'centroids':
        y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_decoded_raw[:,:,[-2,-1]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-2]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-3,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

    if normalize_coords:
        y_pred_decoded_raw[:,:,[-4,-2]] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:,:,[-3,-1]] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 3: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[-1] - 4 # The number of classes is the length of the last axis minus the four box coordinates

    y_pred_decoded = [] # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[class_id, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
            threshold_met = single_class[single_class[:,0] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold, coords='corners', border_pixels=border_pixels) # ...perform NMS on them.
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:,0] = class_id # Write the class ID to the first column...
                maxima_output[:,1:] = maxima # ...and write the maxima to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        if pred: # If there are any predictions left after confidence-thresholding...
            pred = np.concatenate(pred, axis=0)
            if top_k != 'all' and pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        else:
            pred = np.array(pred) # Even if empty, `pred` must become a Numpy array.
        y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded



def decode_y(y_pred,
             confidence_thresh=None, # default: 0.01,
             iou_threshold=None, # default: 0.45,
             top_k=None,  #default: 200,
             input_coords=None, # default: 'centroids',
             normalize_coords=None, # default:  True,
             ssd_config=None,
             do_nms=True,  # do non-max suppression. otherwise returns decoded_y boxes without converting to boxes.
             to_dict=False,
             imsize=None):
    """
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `enconde_y()` takes as input).

    After the decoding, two stages of prediction filtering are performed for each class individually:
    First confidence thresholding, then greedy non-maximum suppression. The filtering results for all
    classes are concatenated and the `top_k` overall highest confidence results constitute the final
    predictions for a given batch item. This procedure follows the original Caffe implementation.
    For a slightly different and more efficient alternative to decode raw model output that performs
    non-maximum suppresion globally instead of per class, see `decode_y2()` below.

    Arguments:
        y_pred (np.ndarray): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage. Defaults to 0.01, following the paper.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score. Defaults to 0.45 following the paper.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage. Defaults to 200, following the paper.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`. Defaults to 'centroids'.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`. Defaults to `False`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, ymin, xmax, ymax]`.
    """

    get_val = lambda x, default: x if x is not None else default

    confidence_thresh = get_val(confidence_thresh, ssd_config.score_thresh)
    iou_threshold = get_val(iou_threshold, ssd_config.nms_iou_thresh)
    top_k = get_val(top_k, ssd_config.max_total_size)
    normalize_coords = get_val(normalize_coords, ssd_config.normalize_coords)
    img_height = ssd_config.img_height
    img_width = ssd_config.img_width
    if imsize is not None:
        img_height, img_width = imsize

    if y_pred.ndim == 2:
        y_pred = np.expand_dims(y_pred, axis=0)

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    n_classes = ssd_config.n_classes + 1
    n_variances = len(ssd_config.variances) if ssd_config.add_variances_to_anchors else 0
    n_box_coords = 4
    n_anchor_coords = 4
    expected_size = n_classes + n_variances + n_box_coords + n_anchor_coords

    assert y_pred.shape[2] == expected_size, "Mismatch in expected size (%d expected vs %d actual)" % (expected_size, y_pred.shape[2])

    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    if not ssd_config.class_activation_in_network: # apply softmax (or other score conversion) now if not done before.
        if ssd_config.class_activation == 'softmax': # keras
            y_pred[:,:, :n_classes] = softmax(y_pred[:, :, :n_classes], axis=2)
        elif ssd_config.class_activation == 'sigmoid':  #tensorflow
            y_pred[:, :, :n_classes] = sigmoid(y_pred[:, :, :n_classes])
        else:
            raise ValueError('Unrecognized score converting function : %s' %
                             ssd_config.class_activation)


    y_pred_decoded_raw = decode_y_coordinates(y_pred, config=ssd_config,
                                                 remove_anchors=True,
                                                 final_coords='corners')

    if np.any(np.isinf(y_pred_decoded_raw) | np.isnan(y_pred_decoded_raw)):
        blank_preds = [[]] * y_pred.shape[0]
        return blank_preds
        # return []


    do_check_with_old_code = True and ssd_config.add_variances_to_anchors
    if do_check_with_old_code:

        y_pred_decoded_raw2 = np.copy(y_pred[:,:,:-(n_anchor_coords+n_variances)]) # Slice out the classes and the four offsets, (throw away the anchor coordinates and variances), resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

        if ssd_config.pred_coords == 'centroids':
            y_pred_decoded_raw2[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw2[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]])
            # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
            y_pred_decoded_raw2[:,:,[-2,-1]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
            y_pred_decoded_raw2[:,:,[-4,-3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
            y_pred_decoded_raw2[:,:,[-4,-3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
            y_pred_decoded_raw2 = convert_coordinates(y_pred_decoded_raw2, start_index=-4, conversion='centroids2corners')
        elif ssd_config.pred_coords == 'minmax':
            y_pred_decoded_raw2[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
            y_pred_decoded_raw2[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
            y_pred_decoded_raw2[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
            y_pred_decoded_raw2[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
            y_pred_decoded_raw2 = convert_coordinates(y_pred_decoded_raw2, start_index=-4, conversion='minmax2corners')
        elif ssd_config.pred_coords == 'corners':
            y_pred_decoded_raw2[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
            y_pred_decoded_raw2[:,:,[-4,-2]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
            y_pred_decoded_raw2[:,:,[-3,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
            y_pred_decoded_raw2[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        else:
            raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")


        assert np.allclose(y_pred_decoded_raw2, y_pred_decoded_raw, atol=1e-6)

    idx_xminmax = [-4, -2]
    idx_yminmax = [-3, -1]

    # Convert yx format to xy format:
    final_xy_order = 'xy'
    if ssd_config.box_xy_order != final_xy_order:
        y_pred_decoded_raw = convert_coordinates_xy(
            y_pred_decoded_raw, start_index=n_classes, coords=ssd_config.pred_coords)

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that
    if normalize_coords:
        y_pred_decoded_raw[:,:,idx_xminmax] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:,:,idx_yminmax] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 3: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[-1] - 4 # The number of classes is the length of the last axis minus the four box coordinates

    y_preds_arrays = []
    y_preds_idxs = []
    for ii, batch_item in enumerate(y_pred_decoded_raw): # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here
        pred_idxs = []
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[class_id, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
            single_class_conf = single_class[:,0]
            idxs_above_th = single_class_conf > confidence_thresh
            threshold_met = single_class[idxs_above_th]  # ...keep only those boxes with a confidence above the set threshold.
            num_above_conf_th = threshold_met.shape[0]
            if num_above_conf_th > 1: # have at least one box for this class
                #if single_class_conf.min() < 0:
                #    single_class_conf = scipy.special.softmax(single_class_conf)
                if iou_threshold > 1 or not do_nms: # include all boxes
                    maxima_output = np.concatenate( (np.ones( (num_above_conf_th, 1))*class_id, threshold_met), axis=1)
                    pred.append(maxima_output)
                    pred_idxs.extend( np.nonzero(idxs_above_th)[0 ])
                else: # use non-max suppression to filter out classes
                    maxima, maxima_idxs = _greedy_nms_v2(threshold_met, iou_threshold=iou_threshold, coords='corners', return_idxs=True)  # ...perform NMS on them.
                    maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                    maxima_output[:,0] = class_id # Write the class ID to the first column...
                    maxima_output[:,1:] = maxima # ...and write the maxima to the other columns...
                    pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
                    pred_idxs.extend(maxima_idxs)

        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        if pred: # If there are any predictions left after confidence-thresholding...
            pred = np.concatenate(pred, axis=0)
            if pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        else:
            pred = np.array([])
        y_preds_arrays.append(pred)
        y_preds_idxs.append( np.concatenate( (np.array(pred_idxs),) ) )


    # Convert to dictionaries for easier access
    if to_dict:
        y_preds_dicts = [boxarray2dict(pred, pred_idxs, ssd_config=ssd_config, imsize=imsize)
                         for pred,pred_idxs in zip(y_preds_arrays, y_preds_idxs) ]
        return y_preds_dicts
    else:
        return y_preds_arrays



def boxarray2dict(boxes, box_idxs=None, sort_by_score=True, ssd_config=None, imsize=None):
    # assumes boxes are a 2d array with format
    #   [[class_id1, score1, xmin1, ymin1, xmax1, ymax1],
    #    [class_id2, score2, xmin2, ymin2, xmax2, ymax2]]
    boxes_dict = OrderedDict()

    box_size = None
    if imsize is None and ssd_config is not None:
        imsize = ssd_config.imsize[:2]
    if imsize is not None:
        img_height, img_width = imsize
        box_size = np.array([img_width, img_height, img_width, img_height])


    num_detections = boxes.shape[0]
    boxes_dict['num_detections'] = num_detections
    if num_detections > 0:
        gt_labels = len(boxes[0]) == 5
        if gt_labels:
            classes = boxes[:, 0].astype(np.int32)
            scores = np.ones(num_detections)
            box_corners = boxes[:, 1:]
            idxs = slice(num_detections)

        else:
            classes = boxes[:, 0].astype(np.int32)
            scores = boxes[:, 1]
            box_corners = boxes[:, 2:]

            if sort_by_score:
                idxs = np.argsort(-boxes[:, 1])  # add negative to sort in ascending order
            else:
                idxs = slice(num_detections)

        boxes_dict['num_detections'] = num_detections
        boxes_dict['classes'] = classes[idxs]
        boxes_dict['scores'] = scores[idxs]
        if box_idxs is not None:
            box_idxs = box_idxs[idxs]
        boxes_dict['box_idxs'] = box_idxs
        box_corners = box_corners[idxs]
        if box_size is not None:
            if np.any(box_corners > 1):
                box_corners_norm = box_corners / box_size
                box_corners_raw = box_corners
            else:
                box_corners_norm = box_corners
                box_corners_raw = box_corners * box_size

            boxes_dict['boxes'] = box_corners_raw
            boxes_dict['boxes_norm'] = box_corners_norm
        else:
            boxes_dict['boxes_norm'] = box_corners


    return boxes_dict


def drawLabelsOnImage(im, labels_dict, orig_shape=None, class_names=None):
    assert isinstance(labels_dict, dict)

    boxes, classes, scores, num_det = \
        labels_dict['boxes'], labels_dict['classes'], \
        labels_dict['scores'], labels_dict['num_detections']

    imsize = im.shape[:2]
    im = im.copy()
    add_scores = not np.all(scores==1)
    for pred_idx in range(num_det):
        box_coords, class_id, score = boxes[pred_idx], classes[pred_idx], scores[pred_idx]
        # box_coords, class_id, score = pred[:-4], int(pred[0]), pred[1]
        if orig_shape is not None:
            xmin, ymin, xmax, ymax = scaleBoxPredToOrigImage(box_coords, orig_shape, imsize)
        else:
            xmin, ymin, xmax, ymax = box_coords

        # print int(box[-4]), int(box[-2]) , int(box[-3]) , int(box[-1])
        print(xmin, xmax, ymin, ymax)
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

        if class_names is not None:
            label = '%s' % (class_names[class_id])
        else:
            label = '%d' % (class_id)
        if add_scores:
            label += '[%.3f]' % score


        text_thickness = 1
        cv2.putText(im, label, (xmin, max(ymin - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255),
                    thickness=text_thickness)

    return im
        # putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None): # real signature unknown; restored from __doc__




def decode_y2(y_pred,
              confidence_thresh=0.5,
              iou_threshold=0.45,
              top_k='all',
              input_coords='centroids',
              normalize_coords=True,
              img_height=None,
              img_width=None):
    """
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `encode_y()` takes as input).

    Optionally performs confidence thresholding and greedy non-maximum suppression after the decoding stage.

    Note that the decoding procedure used here is not the same as the procedure used in the original Caffe implementation.
    For each box, the procedure used here assigns the box's highest confidence as its predicted class. Then it removes
    all boxes for which the highest confidence is the background class. This results in less work for the subsequent
    non-maximum suppression, because the vast majority of the predictions will be filtered out just by the fact that
    their highest confidence is for the background class. It is much more efficient than the procedure of the original
    implementation, but the results may also differ.

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in any positive
            class required for a given box to be considered a positive prediction. A lower value will result
            in better recall, while a higher value will result in better precision. Do not use this parameter with the
            goal to combat the inevitably many duplicates that an SSD will produce, the subsequent non-maximum suppression
            stage will take care of those. Defaults to 0.5.
        iou_threshold (float, optional): `None` or a float in [0,1]. If `None`, no non-maximum suppression will be
            performed. If not `None`, greedy NMS will be performed after the confidence thresholding stage, meaning
            all boxes with a Jaccard similarity of greater than `iou_threshold` with a locally maximal box will be removed
            from the set of predictions, where 'maximal' refers to the box score. Defaults to 0.45.
        top_k (int, optional): 'all' or an integer with number of highest scoring predictions to be kept for each batch item
            after the non-maximum suppression stage. Defaults to 'all', in which case all predictions left after the NMS stage
            will be kept.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`. Defaults to 'centroids'.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`. Defaults to `False`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, xmax, ymin, ymax]`.
    """
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    # 1: Convert the classes from one-hot encoding to their class ID
    y_pred_converted = np.copy(y_pred[:,:,-14:-8]) # Slice out the four offset predictions plus two elements whereto we'll write the class IDs and confidences in the next step
    y_pred_converted[:,:,0] = np.argmax(y_pred[:,:,:-12], axis=-1) # The indices of the highest confidence values in the one-hot class vectors are the class ID
    y_pred_converted[:,:,1] = np.amax(y_pred[:,:,:-12], axis=-1) # Store the confidence values themselves, too

    # 2: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    if input_coords == 'centroids':
        y_pred_converted[:,:,[4,5]] = np.exp(y_pred_converted[:,:,[4,5]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_converted[:,:,[4,5]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_converted[:,:,[2,3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_converted[:,:,[2,3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_converted = convert_coordinates(y_pred_converted, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        y_pred_converted[:,:,2:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_converted[:,:,[2,3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_converted[:,:,[4,5]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_converted[:,:,2:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        y_pred_converted = convert_coordinates(y_pred_converted, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        y_pred_converted[:,:,2:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_converted[:,:,[2,4]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_converted[:,:,[3,5]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_converted[:,:,2:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    # 3: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that
    if normalize_coords:
        y_pred_converted[:,:,[2,4]] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_converted[:,:,[3,5]] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 4: Decode our huge `(batch, #boxes, 6)` tensor into a list of length `batch` where each list entry is an array containing only the positive predictions
    y_pred_decoded = []
    for batch_item in y_pred_converted: # For each image in the batch...
        boxes = batch_item[np.nonzero(batch_item[:,0])] # ...get all boxes that don't belong to the background class,...
        boxes = boxes[boxes[:,1] >= confidence_thresh] # ...then filter out those positive boxes for which the prediction confidence is too low and after that...
        if iou_threshold: # ...if an IoU threshold is set...
            boxes = _greedy_nms2(boxes, iou_threshold=iou_threshold, coords='corners') # ...perform NMS on the remaining boxes.
        if top_k != 'all' and boxes.shape[0] > top_k: # If we have more than `top_k` results left at this point...
            top_k_indices = np.argpartition(boxes[:,1], kth=boxes.shape[0]-top_k, axis=0)[boxes.shape[0]-top_k:] # ...get the indices of the `top_k` highest-scoring boxes...
            boxes = boxes[top_k_indices] # ...and keep only those boxes...
        y_pred_decoded.append(boxes) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded



"""
ssd_keras/ssd_encoder_decoder/matching_utils.py
Utilities to match ground truth boxes to anchor boxes.
"""

def match_bipartite_greedy(weight_matrix):
    """
    Returns a bipartite matching according to the given weight matrix.

    The algorithm works as follows:

    Let the first axis of `weight_matrix` represent ground truth boxes
    and the second axis anchor boxes.
    The ground truth box that has the greatest similarity with any
    anchor box will be matched first, then out of the remaining ground
    truth boxes, the ground truth box that has the greatest similarity
    with any of the remaining anchor boxes will be matched second, and
    so on. That is, the ground truth boxes will be matched in descending
    order by maximum similarity with any of the respectively remaining
    anchor boxes.
    The runtime complexity is O(m^2 * n), where `m` is the number of
    ground truth boxes and `n` is the number of anchor boxes.

    Arguments:
        weight_matrix (np.ndarray): A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.

    Returns:
        A 1D Numpy array of length `weight_matrix.shape[0]` that represents
        the matched index along the second axis of `weight_matrix` for each index
        along the first axis.
    """

    weight_matrix = np.copy(weight_matrix) # We'll modify this array.
    num_ground_truth_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_ground_truth_boxes)) # Only relevant for fancy-indexing below.

    # This 1D array will contain for each ground truth box the index of
    # the matched anchor box.
    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)

    # In each iteration of the loop below, exactly one ground truth box
    # will be matched to one anchor box.
    for _ in range(num_ground_truth_boxes):

        # Find the maximal anchor-ground truth pair in two steps: First, reduce
        # over the anchor boxes and then reduce over the ground truth boxes.
        anchor_indices = np.argmax(weight_matrix, axis=1) # Reduce along the anchor box axis.
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        ground_truth_index = np.argmax(overlaps) # Reduce along the ground truth box axis.
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index # Set the match.

        # Set the row of the matched ground truth box and the column of the matched
        # anchor box to all zeros. This ensures that those boxes will not be matched again,
        # because they will never be the best matches for any other boxes.
        weight_matrix[ground_truth_index] = 0
        weight_matrix[:,anchor_index] = 0

    return matches

def match_multi(weight_matrix, threshold):
    """
    Matches all elements along the second axis of `weight_matrix` to their best
    matches along the first axis subject to the constraint that the weight of a match
    must be greater than or equal to `threshold` in order to produce a match.

    If the weight matrix contains elements that should be ignored, the row or column
    representing the respective elemet should be set to a value below `threshold`.

    Arguments:
        weight_matrix (np.ndarray): A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.
        threshold (float): A float that represents the threshold (i.e. lower bound)
            that must be met by a pair of elements to produce a match.

    Returns:
        Two 1D Numpy arrays of equal length that represent the matched indices. The first
        array contains the indices along the first axis of `weight_matrix`, the second array
        contains the indices along the second axis.
    """

    num_anchor_boxes = weight_matrix.shape[1]
    all_anchor_indices = list(range(num_anchor_boxes)) # Only relevant for fancy-indexing below.

    # Find the best ground truth match for every anchor box.
    ground_truth_indices = np.argmax(weight_matrix, axis=0) # Array of shape (weight_matrix.shape[1],)
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices] # Array of shape (weight_matrix.shape[1],)

    # Filter out the matches with a weight below the threshold.
    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]
    gt_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]

    return gt_indices_thresh_met, anchor_indices_thresh_met




################################################################################
# Debugging tools, not relevant for normal use
################################################################################

# The functions below are for debugging, so you won't normally need them. That is,
# unless you need to debug your model, of course.

def decode_y_debug(y_pred,
                   confidence_thresh=0.01,
                   iou_threshold=0.45,
                   top_k=200,
                   input_coords='centroids',
                   normalize_coords=True,
                   img_height=None,
                   img_width=None,
                   variance_encoded_in_target=False):
    """
    This decoder performs the same processing as `decode_y()`, but the output format for each left-over
    predicted box is `[box_id, class_id, confidence, xmin, ymin, xmax, ymax]`.

    That is, in addition to the usual data, each predicted box has the internal index of that box within
    the model (`box_id`) prepended to it. This allows you to know exactly which part of the model made a given
    box prediction; in particular, it allows you to know which predictor layer made a given prediction.
    This can be useful for debugging.

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage. Defaults to 0.01, following the paper.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score. Defaults to 0.45 following the paper.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage. Defaults to 200, following the paper.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`. Defaults to 'centroids'.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`. Defaults to `False`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 7)` where each row is a box prediction for
        a non-background class for the respective image in the format `[box_id, class_id, confidence, xmin, ymin, xmax, ymax]`.
    """
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    y_pred_decoded_raw = np.copy(y_pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

    if input_coords == 'centroids':
        if variance_encoded_in_target:
            # Decode the predicted box center x and y coordinates.
            y_pred_decoded_raw[:,:,[-4,-3]] = y_pred_decoded_raw[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] + y_pred[:,:,[-8,-7]]
            # Decode the predicted box width and heigt.
            y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]]) * y_pred[:,:,[-6,-5]]
        else:
            # Decode the predicted box center x and y coordinates.
            y_pred_decoded_raw[:,:,[-4,-3]] = y_pred_decoded_raw[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] * y_pred[:,:,[-4,-3]] + y_pred[:,:,[-8,-7]]
            # Decode the predicted box width and heigt.
            y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) * y_pred[:,:,[-6,-5]]
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-2]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-3,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

    if normalize_coords:
        y_pred_decoded_raw[:,:,[-4,-2]] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:,:,[-3,-1]] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 3: For each batch item, prepend each box's internal index to its coordinates.

    y_pred_decoded_raw2 = np.zeros((y_pred_decoded_raw.shape[0], y_pred_decoded_raw.shape[1], y_pred_decoded_raw.shape[2] + 1)) # Expand the last axis by one.
    y_pred_decoded_raw2[:,:,1:] = y_pred_decoded_raw
    y_pred_decoded_raw2[:,:,0] = np.arange(y_pred_decoded_raw.shape[1]) # Put the box indices as the first element for each box via broadcasting.
    y_pred_decoded_raw = y_pred_decoded_raw2

    # 4: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[-1] - 5 # The number of classes is the length of the last axis minus the four box coordinates and minus the index

    y_pred_decoded = [] # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[0, class_id + 1, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 6]` and...
            threshold_met = single_class[single_class[:,1] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                maxima = _greedy_nms_debug(threshold_met, iou_threshold=iou_threshold, coords='corners') # ...perform NMS on them.
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:,0] = maxima[:,0] # Write the box index to the first column...
                maxima_output[:,1] = class_id # ...and write the class ID to the second column...
                maxima_output[:,2:] = maxima[:,1:] # ...and write the rest of the maxima data to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        pred = np.concatenate(pred, axis=0)
        if pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
            top_k_indices = np.argpartition(pred[:,2], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
            pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded

def _greedy_nms_debug(predictions, iou_threshold=0.45, coords='corners'):
    """
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function for per-class NMS in `decode_y_debug()`. The difference is that it keeps the indices of all left-over boxes
    for each batch item, which allows you to know which predictor layer predicted a given output box and is thus
    useful for debugging.
    """
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,2:], maximum_box[2:], coords=coords) # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
    return np.array(maxima)

def get_num_boxes_per_pred_layer(predictor_sizes, aspect_ratios, two_boxes_for_ar1):
    """
    Returns a list of the number of boxes that each predictor layer predicts.

    `aspect_ratios` must be a nested list, containing a list of aspect ratios
    for each predictor layer.
    """
    num_boxes_per_pred_layer = []
    for i in range(len(predictor_sizes)):
        if two_boxes_for_ar1:
            num_boxes_per_pred_layer.append(predictor_sizes[i][0] * predictor_sizes[i][1] * (len(aspect_ratios[i]) + 1))
        else:
            num_boxes_per_pred_layer.append(predictor_sizes[i][0] * predictor_sizes[i][1] * len(aspect_ratios[i]))
    return num_boxes_per_pred_layer

def get_pred_layers(y_pred_decoded, num_boxes_per_pred_layer):
    """
    For a given prediction tensor decoded with `decode_y_debug()`, returns a list
    with the indices of the predictor layers that made each predictions.

    That is, this function lets you know which predictor layer is responsible
    for a given prediction.

    Arguments:
        y_pred_decoded (array): The decoded model output tensor. Must have been
            decoded with `decode_y_debug()` so that it contains the internal box index
            for each predicted box.
        num_boxes_per_pred_layer (list): A list that contains the total number
            of boxes that each predictor layer predicts.
    """
    pred_layers_all = []
    cum_boxes_per_pred_layer = np.cumsum(num_boxes_per_pred_layer)
    for batch_item in y_pred_decoded:
        pred_layers = []
        for prediction in batch_item:
            if (prediction[0] < 0) or (prediction[0] >= cum_boxes_per_pred_layer[-1]):
                raise ValueError("Box index is out of bounds of the possible indices as given by the values in `num_boxes_per_pred_layer`.")
            for i in range(len(cum_boxes_per_pred_layer)):
                if prediction[0] < cum_boxes_per_pred_layer[i]:
                    pred_layers.append(i)
                    break
        pred_layers_all.append(pred_layers)
    return pred_layers_all


"""
def get_pascal_class_names(include_bckg_class=True):
    pascal_class_names = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
                          'Chair', 'Cow', 'Dining table', 'Dog', 'Horse', 'Motorbike', 'Person',
                          'Potted plant', 'Sheep', 'Sofa', 'Train', 'TV monitor']
    if include_bckg_class:
        pascal_class_names.insert(0, '[Background]')

    return pascal_class_names
"""

def scaleBoxPredToOrigImage(box, orig_shape, pred_shape):
    img_width, img_height = pred_shape

    xmin = int(box[-4] * orig_shape[1] / img_width)
    ymin = int(box[-3] * orig_shape[0] / img_height)
    xmax = int(box[-2] * orig_shape[1] / img_width)
    ymax = int(box[-1] * orig_shape[0] / img_height)

    return xmin, ymin, xmax, ymax




def normalize_to_target(inputs,
                        target_norm_value,
                        dim,
                        epsilon=1e-7,
                        trainable=True,
                        scope='NormalizeToTarget',
                        summarize=True):
  """L2 normalizes the inputs across the specified dimension to a target norm.

  This op implements the L2 Normalization layer introduced in
  Liu, Wei, et al. "SSD: Single Shot MultiBox Detector."
  and Liu, Wei, Andrew Rabinovich, and Alexander C. Berg.
  "Parsenet: Looking wider to see better." and is useful for bringing
  activations from multiple layers in a convnet to a standard scale.

  Note that the rank of `inputs` must be known and the dimension to which
  normalization is to be applied should be statically defined.

  TODO(jonathanhuang): Add option to scale by L2 norm of the entire input.

  Args:
    inputs: A `Tensor` of arbitrary size.
    target_norm_value: A float value that specifies an initial target norm or
      a list of floats (whose length must be equal to the depth along the
      dimension to be normalized) specifying a per-dimension multiplier
      after normalization.
    dim: The dimension along which the input is normalized.
    epsilon: A small value to add to the inputs to avoid dividing by zero.
    trainable: Whether the norm is trainable or not
    scope: Optional scope for variable_scope.
    summarize: Whether or not to add a tensorflow summary for the op.

  Returns:
    The input tensor normalized to the specified target norm.

  Raises:
    ValueError: If dim is smaller than the number of dimensions in 'inputs'.
    ValueError: If target_norm_value is not a float or a list of floats with
      length equal to the depth along the dimension to be normalized.
  """
  with tf.variable_scope(scope, 'NormalizeToTarget', [inputs]):
    if not inputs.get_shape():
      raise ValueError('The input rank must be known.')
    input_shape = inputs.get_shape().as_list()
    input_rank = len(input_shape)
    if dim < 0 or dim >= input_rank:
      raise ValueError(
          'dim must be non-negative but smaller than the input rank.')
    if not input_shape[dim]:
      raise ValueError('input shape should be statically defined along '
                       'the specified dimension.')
    depth = input_shape[dim]
    if not (isinstance(target_norm_value, float) or
            (isinstance(target_norm_value, list) and
             len(target_norm_value) == depth) and
            all([isinstance(val, float) for val in target_norm_value])):
      raise ValueError('target_norm_value must be a float or a list of floats '
                       'with length equal to the depth along the dimension to '
                       'be normalized.')
    if isinstance(target_norm_value, float):
      initial_norm = depth * [target_norm_value]
    else:
      initial_norm = target_norm_value
    target_norm = tf.contrib.framework.model_variable(
        name='weights', dtype=tf.float32,
        initializer=tf.constant(initial_norm, dtype=tf.float32),
        trainable=trainable)
    if summarize:
      mean = tf.reduce_mean(target_norm)
      tf.summary.scalar(tf.get_variable_scope().name, mean)
    lengths = epsilon + tf.sqrt(tf.reduce_sum(tf.square(inputs), dim, True))
    mult_shape = input_rank*[1]
    mult_shape[dim] = depth
    return tf.reshape(target_norm, mult_shape) * tf.truediv(inputs, lengths)






def get_post_processor(y_pred=None, # mbox_class, mbox_encoding=None, mbox_anchors=None,
                   sess=None, ssd_config=None, label_mapping=None):

    if sess is None:
        sess = tf.get_default_session()

    assert ssd_config is not None, "Please provide SSDConfig"
    #if config is None:
    #    config = SSDConfig(net_type=net.SSDNetType(2,False, 'tf'))

    #if mbox_encoding is None and mbox_anchors is None: # all concatenated in mbox_class
    num_classes = ssd_config.n_classes + 1
    num_box_params = 4
    num_anchor_params = 4
    num_variances = 4 if ssd_config.add_variances_to_anchors else 0
    n_tot = num_classes + num_box_params + num_anchor_params + num_variances

    box_class_idxs = slice(0, num_classes)  # 0:91
    box_encoding_idxs = slice(num_classes, num_classes+num_box_params) # 91:95
    box_anchor_idxs = slice(num_classes+num_box_params, num_classes+num_box_params+num_anchor_params)  # 95:99
    if ssd_config.imsize == (300,300):
        num_boxes = 1917
    elif ssd_config.imsize == (224,224):
        num_boxes = 1014
    else:
        num_boxes = None

    y_pred_placeholder = tf.placeholder(tf.float32, [None, num_boxes, n_tot])

    mbox_class = y_pred_placeholder[:, :, box_class_idxs]
    mbox_encoding = y_pred_placeholder[:, :, box_encoding_idxs]
    mbox_anchors = y_pred_placeholder[0, :, box_anchor_idxs]


    # mbox_class =  tf.constant(expected_activations['concat_1'])
    # mbox_encoding = tf.constant( expected_activations['concat'] )
    # mbox_anchors = tf.constant( expected_activations['MultipleGridAnchorGenerator/Concatenate/concat'] )

    #mbox_class = expected_activations['concat_1']
    mbox_class_no_background_class = mbox_class[:, :, 1:]
    mbox_class_convert_scores = tf.math.sigmoid(mbox_class_no_background_class)

    # mbox_encoding = expected_activations['concat']
    # mbox_anchors = expected_activations['MultipleGridAnchorGenerator/Concatenate/concat']

    #mbox_class_tf = tf.constant(expected_activations['concat_1'])
    #mbox_encoding_tf = tf.constant(expected_activations['concat'])
    #mbox_anchors_tf = tf.constant(expected_activations['MultipleGridAnchorGenerator/Concatenate/concat'])

    box_coder_scales = list(1.0 / v for v in ssd_config.variances)
    box_coder_faster_rcnn = FasterRcnnBoxCoder(box_coder_scales)

    decoded_boxes_tf, decoded_keypoints = post_processing.batch_decode(
        box_coder_faster_rcnn, mbox_encoding, mbox_anchors)

    decoded_boxes_tf_exp = tf.expand_dims(decoded_boxes_tf, axis=2)

    score_thresh = ssd_config.score_thresh # 0.2
    iou_thresh = ssd_config.nms_iou_thresh  # 0.6
    max_total_size = ssd_config.max_total_size #  100
    max_size_per_class = ssd_config.max_size_per_class # 10

    nms_boxes_tf, nms_scores_tf, nms_classes_tf, nms_masks_tf, nms_extra_tf, num_detections_tf = \
        post_processing.batch_multiclass_non_max_suppression(
        decoded_boxes_tf_exp, mbox_class_convert_scores, score_thresh=score_thresh,
            iou_thresh=iou_thresh, max_size_per_class=max_size_per_class,
            max_total_size=max_total_size)


    def post_processer_func(y_pred, do_nms=True):
        if not do_nms:
            decoded_boxes = sess.run([decoded_boxes_tf],
                         feed_dict={y_pred_placeholder: y_pred})
            return decoded_boxes

        nms_boxes, nms_scores, nms_classes, num_detections, decoded_boxes = \
            sess.run([nms_boxes_tf, nms_scores_tf, nms_classes_tf, num_detections_tf, decoded_boxes_tf], feed_dict={y_pred_placeholder:y_pred})

        batch_size = y_pred.shape[0]
        output_dicts = []
        for batch_i in range(batch_size):

            num_detections_i = int(num_detections[batch_i])

            detection_boxes = nms_boxes[batch_i, :num_detections_i]
            detection_classes = 1 + nms_classes[batch_i, :num_detections_i].astype(np.int64)
            detection_scores = nms_scores[batch_i, :num_detections_i]

            if ssd_config.box_xy_order=='yx':
                # switch from y,x to x,y:
                detection_boxes = detection_boxes[:,[1,0,3,2]]  # ymin, xmin, ymax,xmax ==> xmin, ymin, xmax, ymax

            img_width, img_height = ssd_config.img_width, ssd_config.img_height
            image_size_vec = np.array([img_width, img_height, img_width, img_height])
            if ssd_config.normalize_coords:
                detection_boxes_norm = detection_boxes
                detection_boxes_img_coords = detection_boxes * image_size_vec
            else:
                detection_boxes_img_coords = detection_boxes
                detection_boxes_norm = detection_boxes / image_size_vec

            if label_mapping is not None:
                detection_classes = label_mapping[detection_classes]

            output_dict = OrderedDict()
            output_dict['num_detections'] = num_detections_i
            if num_detections_i > 0:
                output_dict['classes']= detection_classes
                output_dict['scores'] = detection_scores
                output_dict['boxes'] = detection_boxes_img_coords
                output_dict['boxes_norm'] = detection_boxes_norm

            output_dicts.append(output_dict)

        return output_dicts

    return post_processer_func

bbox_idxs_default = dict(xmin=1, ymin=2, xmax=3, ymax=4)   # xmin, ymin, xmax, ymax


def crop_image_and_annot(X, y, crop, include_thresh, bbox_idxs=None):
    if bbox_idxs is None:
        bbox_idxs = bbox_idxs_default
    xmin, ymin, xmax, ymax = bbox_idxs['xmin'], bbox_idxs['ymin'], \
                             bbox_idxs['xmax'], bbox_idxs['ymax']

    limit_boxes=True
    crop_top, crop_bottom, crop_left, crop_right = crop
    img_height, img_width = X.shape[:2]
    # Crop the image
    X = np.copy(X[crop_top:img_height - crop_bottom, crop_left:img_width - crop_right])
    # Update the image size so that subsequent transformations can work correctly
    img_height -= crop_top + crop_bottom
    img_width -= crop_left + crop_right
    if y is not None and len(y) > 0:
        # Translate the box coordinates into the new coordinate system if necessary: The origin is shifted by `(crop[0], crop[2])` (i.e. by the top and left crop values)
        # If nothing was cropped off from the top or left of the image, the coordinate system stays the same as before
        if crop_top > 0:
            y[:, [ymin, ymax]] -= crop_top
        if crop_left > 0:
            y[:, [xmin, xmax]] -= crop_left
        # Limit the box coordinates to lie within the new image boundaries
        if limit_boxes:
            before_limiting = np.copy(y)
            # We only need to check those box coordinates that could possibly have been affected by the cropping
            # For example, if we only crop off the top and/or bottom of the image, there is no need to check the x-coordinates
            if crop_top > 0:
                y_coords = y[:, [ymin, ymax]]
                y_coords[y_coords < 0] = 0
                y[:, [ymin, ymax]] = y_coords
            if crop_bottom > 0:
                y_coords = y[:, [ymin, ymax]]
                y_coords[y_coords >= img_height] = img_height - 1
                y[:, [ymin, ymax]] = y_coords
            if crop_left > 0:
                x_coords = y[:, [xmin, xmax]]
                x_coords[x_coords < 0] = 0
                y[:, [xmin, xmax]] = x_coords
            if crop_right > 0:
                x_coords = y[:, [xmin, xmax]]
                x_coords[x_coords >= img_width] = img_width - 1
                y[:, [xmin, xmax]] = x_coords
            # Some objects might have gotten pushed so far outside the image boundaries in the transformation
            # process that they don't serve as useful training examples anymore, because too little of them is
            # visible. We'll remove all boxes that we had to limit so much that their area is less than
            # `include_thresh` of the box area before limiting.
            before_area = (before_limiting[:, xmax] - before_limiting[:, xmin]) * (
                    before_limiting[:, ymax] - before_limiting[:, ymin])
            after_area = (y[:, xmax] - y[:, xmin]) * (
                    y[:, ymax] - y[:, ymin])
            if include_thresh == 0:
                y = y[after_area > include_thresh * before_area]  # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
            else:
                y = y[after_area >= include_thresh * before_area]  # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all

    return X, y



def resize_image_and_annot(X, y, dsize, bbox_idxs=None):
    if bbox_idxs is None:
        bbox_idxs = bbox_idxs_default
    xmin, ymin, xmax, ymax = bbox_idxs['xmin'], bbox_idxs['ymin'], \
                             bbox_idxs['xmax'], bbox_idxs['ymax']

    img_height, img_width = X.shape[:2]
    height, width = dsize
    X = cv2.resize(X, dsize=(height, width))
    if (y is not None) and len(y) > 0:
        y[:, [ymin, ymax]] = y[:, [ymin, ymax]] * (height / img_height)
        y[:, [xmin, xmax]] = y[:, [xmin, xmax]] * (width / img_width)

    return X, y

def visualize_boxes_on_image(boxes, image, ssd_config=None, save_filename=None, display_image=False,
                             do_nms=False, scale=None, target_size=None,
                             line_thickness=2,
                             max_boxes_to_draw=100,
                             min_score_thresh=.1,
                             fontsize=24,
                             #agnostic_mode=False,
                             skip_scores=False,
                             skip_labels=False):
    # boxes can be :
    #  1) numpy.ndarray encoded box prediction  (need ssd_config to decode)
    #  2) numpy.ndarray of gt labels
    #  3) dict with [box_norm, classes, scores]
    # - image should be an RGB image
    # - if filename is specified, saves the image to a file
    box_ids = None
    if isinstance(boxes, np.ndarray):
        # anchor box predictions
        if boxes.shape[1] < 10:
            boxes_dict = boxarray2dict(boxes, ssd_config=ssd_config, imsize=image.shape[:2])
        else:
            boxes_dict = decode_y(boxes, ssd_config=ssd_config, do_nms=do_nms, to_dict=True, imsize=image.shape[:2])[0]

    elif isinstance(boxes, dict):
        boxes_dict = boxes
    else:
        raise ValueError('Unrecognized format for boxes')

    im_use = image.copy()
    # img_width, img_height = ssd_config.img_width, ssd_config.img_height
    #if network_ar and im_use.shape[:2] != (img_width, img_height):
    #    im_use = cv2.resize(im_use, (img_width, img_height))

    if scale is not None and scale != 1.0:
        cur_h, cur_w = im_use.shape[:2]
        dsize = (int(cur_w * scale), int(cur_h * scale))  # dsize takes (w,h)
        im_use = cv2.resize(im_use, dsize, interpolation=cv2.INTER_LINEAR)

    # PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
    if np.all(boxes_dict['scores'] == 1):
        skip_scores = True
        skip_labels = False

    skip_scores = False
    #category_index = ssd.get_tf_coco_labels(add_missing_labels=True)  # ssd.coco_labels
    category_index = ssd.get_tf_pascal_labels()  # ssd.coco_labels
    im_use2 = vis_util.visualize_boxes_and_labels_on_image_array(
        im_use,
        boxes_dict['boxes_norm'],  # [:,[1,0,3,2]],
        boxes_dict['classes'],
        boxes_dict['scores'],
        category_index,
        box_ids=boxes_dict['box_idxs'],
        instance_masks=None,
        use_normalized_coordinates=True,
        line_thickness=line_thickness,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        fontsize=fontsize,
        skip_scores=skip_scores,
        skip_labels=skip_labels,
    )

    if save_filename is not None:
        save_dir = os.path.dirname(save_filename)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('Created directory %s' % save_dir)
        imageio.imwrite(save_filename, im_use2)
        print('Saved image with %d boxes to %s' % (
            boxes_dict['num_detections'], save_filename))

    if display_image:
        cv2.imshow('Image with boxes', im_use2[..., ::-1])
        cv2.waitKey(0)

    return im_use2





'''
A few utilities that are useful when working with the MS COCO datasets.
'''

import json
from tqdm import trange
from math import ceil
import sys

#from data_generator.object_detection_2d_geometric_ops import Resize
#from data_generator.object_detection_2d_patch_sampling_ops import RandomPadFixedAR
#from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
#from ssd_encoder_decoder.ssd_output_decoder import decode_detections
#from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

def get_coco_category_maps(annotations_file):
    '''
    Builds dictionaries that map between MS COCO category IDs, transformed category IDs, and category names.
    The original MS COCO category IDs are not consecutive unfortunately: The 80 category IDs are spread
    across the integers 1 through 90 with some integers skipped. Since we usually use a one-hot
    class representation in neural networks, we need to map these non-consecutive original COCO category
    IDs (let's call them 'cats') to consecutive category IDs (let's call them 'classes').

    Arguments:
        annotations_file (str): The filepath to any MS COCO annotations JSON file.

    Returns:
        1) cats_to_classes: A dictionary that maps between the original (keys) and the transformed category IDs (values).
        2) classes_to_cats: A dictionary that maps between the transformed (keys) and the original category IDs (values).
        3) cats_to_names: A dictionary that maps between original category IDs (keys) and the respective category names (values).
        4) classes_to_names: A list of the category names (values) with their indices representing the transformed IDs.
    '''
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    cats_to_classes = {}
    classes_to_cats = {}
    cats_to_names = {}
    classes_to_names = []
    classes_to_names.append('background') # Need to add the background class first so that the indexing is right.
    for i, cat in enumerate(annotations['categories']):
        cats_to_classes[cat['id']] = i + 1
        classes_to_cats[i + 1] = cat['id']
        cats_to_names[cat['id']] = cat['name']
        classes_to_names.append(cat['name'])

    return cats_to_classes, classes_to_cats, cats_to_names, classes_to_names

def predict_all_to_json(model,
                        ssd_config=None,
                        out_file=None,
                        data_generator=None,
                        img_height=None,
                        img_width=None,
                        classes_to_cats=None,
                        batch_size=None,
                        data_generator_mode='resize',
                        model_mode='training',
                        confidence_thresh=0.01,
                        iou_threshold=0.45,
                        top_k=200,
                        pred_coords='centroids',
                        normalize_coords=True,
                        numpy_output=True):

    '''
    Runs detection predictions over the whole dataset given a model and saves them in a JSON file
    in the MS COCO detection results format.

    Arguments:
        out_file: The file name (full path) under which to save the results JSON file.
        model (Keras model): A Keras SSD model object.
        img_height (int): The input image height for the model.
        img_width (int): The input image width for the model.
        classes_to_cats (dict): A dictionary that maps the consecutive class IDs predicted by the model
            to the non-consecutive original MS COCO category IDs.
        data_generator (DataGenerator): A `DataGenerator` object with the evaluation dataset.
        batch_size (int): The batch size for the evaluation.
        data_generator_mode (str, optional): Either of 'resize' or 'pad'. If 'resize', the input images will
            be resized (i.e. warped) to `(img_height, img_width)`. This mode does not preserve the aspect ratios of the images.
            If 'pad', the input images will be first padded so that they have the aspect ratio defined by `img_height`
            and `img_width` and then resized to `(img_height, img_width)`. This mode preserves the aspect ratios of the images.
        model_mode (str, optional): The mode in which the model was created, i.e. 'training', 'inference' or 'inference_fast'.
            This is needed in order to know whether the model output is already decoded or still needs to be decoded. Refer to
            the model documentation for the meaning of the individual modes.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage. Defaults to 200, following the paper.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`.

    Returns:
        None.
    '''
    import datasets.ssd_data_augmentations as ssd_aug

    if ssd_config is not None:
        img_height = ssd_config.img_height
        img_width = ssd_config.img_width
        confidence_thresh = ssd_config.score_thresh
        iou_threshold = ssd_config.nms_iou_thresh
        top_k = ssd_config.max_total_size
        pred_coords = ssd_config.pred_coords
        normalize_coords = ssd_config.normalize_coords

    # Set the generator parameters.
    #def generate(self,
    #             batch_size=32,
    #             shuffle=True,
    #             file_order=False,
    #             seed=None,
    #             returns=('processed_images', 'encoded_labels')
    data_generator.build(img_width=img_width,img_height=img_height,
                         pad_in_eval_mode=data_generator_mode=='pad',
                         keep_images_without_gt=True,
                         )
    generator = data_generator.generate(batch_size=batch_size,
                                        shuffle=False,
                                        returns=['processed_images',
                                                 'image_ids',
                                                 'inverse_transform'])

    # Put the results in this list.
    results = []
    # Compute the number of batches to iterate over the entire dataset.
    n_images = data_generator.get_dataset_size()
    print("Number of images in the evaluation dataset: {}".format(n_images))
    n_batches = int(ceil(n_images / batch_size))
    # Loop over all batches.
    tr = trange(n_batches, file=sys.stdout)
    tr.set_description('Producing results file')
    for i in tr:
        # Generate batch.
        batch_X, batch_image_ids, batch_inverse_transforms = next(generator)
        # Predict.
        y_pred_raw = model.predict(batch_X)
        # If the model was created in 'training' mode, the raw predictions need to
        # be decoded and filtered, otherwise that's already taken care of.
        if model_mode == 'training':
            # Decode.

            y_pred = decode_detections_ssd_repo(y_pred_raw,
                              confidence_thresh=confidence_thresh,
                              iou_threshold=iou_threshold,
                              top_k=top_k,
                              input_coords=pred_coords,
                              normalize_coords=normalize_coords,
                              img_height=img_height,
                              img_width=img_width)

            y_pred2 = decode_y(y_pred_raw, ssd_config=ssd_config)
            m = np.array_equal(y_pred, y_pred2)
            m2 = np.allclose(y_pred, y_pred2)
            a = 1
        else:
            # Filter out the all-zeros dummy elements of `y_pred`.
            y_pred_filtered = []
            for i in range(len(y_pred)):
                y_pred_filtered.append(y_pred[i][y_pred[i,:,0] != 0])
            y_pred = y_pred_filtered
        # Convert the predicted box coordinates for the original images.
        #pred_labels_seq = ('class_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax')
        #pred_labels_format = seq2labelsformat(pred_labels_seq)

        y_pred = ssd_aug.apply_inverse_transforms(y_pred, batch_inverse_transforms)

        # Convert each predicted box into the results format.

        for k, batch_item in enumerate(y_pred):
            for box in batch_item:
                class_id = box[0]
                # Transform the consecutive class IDs back to the original COCO category IDs.
                cat_id = classes_to_cats[class_id]
                # Round the box coordinates to reduce the JSON file size.
                xmin = float(round(box[2], 1))
                ymin = float(round(box[3], 1))
                xmax = float(round(box[4], 1))
                ymax = float(round(box[5], 1))
                width = xmax - xmin
                height = ymax - ymin
                bbox = [xmin, ymin, width, height]
                result = {}
                result['image_id'] = batch_image_ids[k]
                result['category_id'] = cat_id
                result['score'] = float(round(box[1], 3))
                result['bbox'] = bbox
                results.append(result)


    if out_file is not None:
        with open(out_file, 'w') as f:
            json.dump(results, f)

        print("Prediction results saved in '{}'".format(out_file))

    elif numpy_output:
        # Convert result data into a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        num_results = len(results)
        results_array = np.zeros((num_results, 7))
        for i,r in enumerate(results):
            results_row = np.array([r['image_id']] + r['bbox'] + [r['score'], r['category_id']])
            results_array[i,:] = results_row

        return results_array

    else:
        print('Completed predictions, but no way to output data')


# Copied from keras 2.3.0  library, for compatibility with keras version 2.2.5,
# which doesn't support the 'from_logits' parameter
def binary_crossentropy_keras_2_3_0(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)
        y_true = K.switch(K.greater(smoothing, 0),
                          lambda: y_true * (1.0 - smoothing) + 0.5 * smoothing,
                          lambda: y_true)
    return K.mean(
        K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)



def categorical_crossentropy_2_3_0(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
    return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)


if keras.__version__ == '2.2.5':  # backwards compatibility
    binary_crossentropy = binary_crossentropy_keras_2_3_0
    categorical_crossentropy = categorical_crossentropy_2_3_0
else:
    binary_crossentropy = keras.losses.binary_crossentropy # orig function
    categorical_crossentropy = keras.losses.categorical_crossentropy

if __name__ == "__main__":

    import datasets
    import ssd.ssd_info as ssd
    import numpy as np
    from networks import net_utils_core as net

    test_pycoco = True
    if test_pycoco:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_split = 'microval64'
        datasets_eval = ssd.SSDDatasetType(name='coco', years=2014, split=coco_split)

        datasets_info = ssd.get_datasets_info(datasets_eval.spec_list())
        images_dirs = datasets_info['image_dirs']
        annotations_filenames = datasets_info['annot_files']
        assert len(images_dirs) == 1
        assert len(annotations_filenames) == 1
        annotations_filename = annotations_filenames[0]

        # TODO: Set the desired output file name and the batch size.
        #results_file = 'detections_val2017_ssd300_results.json'
        batch_size = 20  # Ideally, choose a batch size that divides the number of images in the dataset.

        results_file = 'F:/SRI/bitnet/gtc-tensorflow/test_code/detections_val2017_ssd300_results.json'
        with open(results_file, 'r') as f:
            results = json.load(f)

        num_results = len(results)
        results_array = np.zeros((num_results, 7))
        for i, r in enumerate(results):
            results_row = np.array([r['image_id']] + r['bbox'] + [r['score'], r['category_id']])
            results_array[i, :] = results_row
        #  We need the `classes_to_cats` dictionary. Read the documentation of this function to understand why.
        cats_to_classes, classes_to_cats, cats_to_names, classes_to_names = get_coco_category_maps(
            annotations_filename)

        coco_gt = COCO(annotations_filename)
        coco_dt = coco_gt.loadRes(results_file)
        image_ids = sorted(coco_gt.getImgIds())

        cocoEval = COCOeval(cocoGt=coco_gt,
                            cocoDt=coco_dt,
                            iouType='bbox')
        cocoEval.params.imgIds = image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()

        from contextlib import redirect_stdout
        import io
        import re

        f = io.StringIO()
        with redirect_stdout(f):
            cocoEval.summarize()
            a = 1
        eval_output = f.getvalue()
        eval_lines = eval_output.split('\n')
        'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.432'

        # sample line: with mAP at IoU=0.50 ' Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.432'
        pattern = 'IoU=0.50 [\w\W]* = ([.\d]+)'

        [float(re.search(pattern, x).groups()[0]) for x in eval_lines if re.search(pattern, x)]




    np.set_printoptions(linewidth=280)
    # compare encoders/decoders:
    sess = tf.InteractiveSession()

    net_type = net.MobilenetSSDType(mobilenet_version=2, style='tf')
    ssd_config = SSDConfig(net_type=net_type, class_activation_in_network=True)
    predictor_sizes = predictor_sizes_mobilenet_300
    ssd_box_encoder = SSDBoxEncoder(ssd_config, predictor_sizes)
    imsize = (300,300)

    post_processor = get_post_processor(ssd_config=ssd_config)

    save_scale = 3.0
    scale_str = '_zoom%d' % save_scale if save_scale != 1 else ''
    split = 'train'
    n_batches_label = 10
    loader = datasets.GTCCocoLoader(years=2014, split=split, imsize=imsize,
                                    ssd_box_encoder=ssd_box_encoder)
    loader.build(resize=imsize)

    batch_size = 1
    it = loader.flow(batch_size=batch_size, file_order=True,
                     returns=('filenames', 'processed_images', 'processed_labels', 'encoded_labels', 'matched_anchors'),
                     keep_images_without_gt=True)

    fn,x,labels, y_true, y_anchors = it[1]

    y_true_decoded_tf = post_processor(y_true)

    #if mbox_encoding is None and mbox_anchors is None: # all concatenated in mbox_class
    num_classes = ssd_config.n_classes + 1
    num_box_params = 4
    num_anchor_params = 4
    num_variances = 4 if ssd_config.add_variances_to_anchors else 0
    n_tot = num_classes + num_box_params + num_anchor_params + num_variances

    box_class_idxs = slice(0, num_classes)  # 0:91
    box_encoding_idxs = slice(num_classes, num_classes+num_box_params) # 91:95
    box_anchor_idxs = slice(num_classes+num_box_params, num_classes+num_box_params+num_anchor_params)  # 95:99
    num_boxes = 1917
    y_pred_placeholder = tf.placeholder(tf.float32, [None, num_boxes, n_tot])

    mbox_class = y_pred_placeholder[:, :, box_class_idxs]
    mbox_encoding = y_pred_placeholder[:, :, box_encoding_idxs]
    mbox_anchors = y_pred_placeholder[0, :, box_anchor_idxs]


    # mbox_class =  tf.constant(expected_activations['concat_1'])
    # mbox_encoding = tf.constant( expected_activations['concat'] )
    # mbox_anchors = tf.constant( expected_activations['MultipleGridAnchorGenerator/Concatenate/concat'] )

    #mbox_class = expected_activations['concat_1']
    mbox_class_no_background_class = mbox_class[:, :, 1:]
    mbox_class_convert_scores = tf.math.sigmoid(mbox_class_no_background_class)

    # mbox_encoding = expected_activations['concat']
    # mbox_anchors = expected_activations['MultipleGridAnchorGenerator/Concatenate/concat']

    #mbox_class_tf = tf.constant(expected_activations['concat_1'])
    #mbox_encoding_tf = tf.constant(expected_activations['concat'])
    #mbox_anchors_tf = tf.constant(expected_activations['MultipleGridAnchorGenerator/Concatenate/concat'])

    box_coder_scales = ssd_config.box_coder_scales # [10.0, 10.0, 5.0, 5.0]
    box_coder_faster_rcnn = FasterRcnnBoxCoder(box_coder_scales)

    decoded_boxes, decoded_keypoints = post_processing.batch_decode(
        box_coder_faster_rcnn, mbox_encoding, mbox_anchors)

    feed_dict={y_pred_placeholder:y_true}
    decoded_boxes_tf = sess.run(decoded_boxes, feed_dict=feed_dict)
    # PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = ssd.get_tf_coco_labels(remove_missing_ids=False,
                                            add_missing_labels=True)  # ssd.coco_labels
    y_decoded = decode_y(y_true, iou_threshold=2, top_k=1000, ssd_config=ssd_config, do_nms=False, to_dict=True)

    decoded_boxes_tf2 = decoded_boxes_tf * 300
    y_decoded_coords = y_decoded[:,:, -4:]

    ismatch = np.allclose(decoded_boxes_tf2, y_decoded_coords)

    a = 1

