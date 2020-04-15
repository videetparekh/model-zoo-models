import sys
sys.path.append("/home/manish/MobileNet-ssd-keras")
import keras
import numpy as np 
import cv2
import keras.backend as K
import keras.layers as KL
#from models.depthwise_conv2d import DepthwiseConvolution2D
from keras.layers import DepthwiseConv2D as DepthwiseConvolution2D
from keras.models import Model
from keras.layers import Input, Lambda, Activation,Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,BatchNormalization, Add, Conv2DTranspose
from keras.regularizers import l2
#from models.mobilenet_v1 import mobilenet
#from misc.keras_layer_L2Normalization import L2Normalization
#from misc.keras_layer_AnchorBoxes import AnchorBoxes
#from misc.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
import networks.net_utils as net
from ssd.ssd_utils import convert_coordinates, SSDConfig


def mobilenet_v1_ssd_300(mode='training', config=None,
                         ssd_config=None):
    assert isinstance(ssd_config, SSDConfig)

    image_size = ssd_config.imsize
    n_classes = ssd_config.n_classes
    #l2_regularization=0.0005

    if config.do_weight_regularization:
        l2_regularization = config.weight_decay_factor
        kernel_regularizer = l2(l2_regularization)
    else:
        kernel_regularizer = None


    min_scale, max_scale = None, None
    scales=ssd_config.scales


    aspect_ratios_global=None
    aspect_ratios_per_layer=ssd_config.aspect_ratios
    two_boxes_for_ar1=ssd_config.two_boxes_for_ar1
    steps=ssd_config.steps
    offsets=ssd_config.offsets
    limit_boxes=ssd_config.limit_boxes
    variances=ssd_config.variances
    pred_coords=ssd_config.pred_coords
    normalize_coords=ssd_config.normalize_coords
    #op_list = [('subtract', 127.5), ('divide', 127.5)]
    subtract_mean= [127.5, 127.5, 127.5]
    divide_by_stddev=127.5
    swap_channels=True
    #return_predictor_sizes=False
    namer=None

    if namer is None:
        namer = lambda x: x + '_v1'

    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1  # Account for the background class.
    #l2_reg = l2_regularization  # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]


    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers



    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(lambda z: z, output_shape=(img_height, img_width, img_channels), name=namer('identity_layer'))(x)
    if not (subtract_mean is None):
        x1 = Lambda(lambda z: z - np.array(subtract_mean), output_shape=(img_height, img_width, img_channels),
                    name=namer('input_mean_normalization'))(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(lambda z: z / np.array(divide_by_stddev), output_shape=(img_height, img_width, img_channels),
                    name=namer('input_stddev_normalization'))(x1)
    if swap_channels and (img_channels == 3):
        x1 = Lambda(lambda z: z[..., ::-1], output_shape=(img_height, img_width, img_channels),
                    name=namer('input_channel_swap'))(x1)


    conv4_3_norm , fc7 ,test= mobilenet(input_tensor=x1, namer=namer)  # conv11, conv13, test=conv5

    print ("conv11 shape: ", conv4_3_norm.shape)
    print ("conv13 shape: ", fc7.shape)


    v1 = '_v1'
    conv6_1 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
        kernel_regularizer=kernel_regularizer, name=namer('conv14_1'), use_bias=False)(fc7)
    conv6_1 = BatchNormalization( momentum=0.99, epsilon=0.00001, name=namer('conv14_1/bn'))(conv6_1)
    conv6_1 = Activation('relu', name=namer('relu_conv6_1'))(conv6_1)

    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name=namer('conv6_padding'))(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
        kernel_regularizer=kernel_regularizer, name=namer('conv14_2'), use_bias=False)(conv6_1)
    conv6_2 = BatchNormalization( momentum=0.99, epsilon=0.00001, name=namer('conv14_2/bn'))(conv6_2)
    conv6_2 = Activation('relu', name=namer('relu_conv6_2'))(conv6_2)

    print ('conv14 shape', conv6_2.shape)



    conv7_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal',
        kernel_regularizer=kernel_regularizer, name=namer('conv15_1'),use_bias=False)(conv6_2)
    conv7_1 = BatchNormalization( momentum=0.99, epsilon=0.00001, name=namer('conv15_1/bn'))(conv7_1)
    conv7_1 = Activation('relu', name=namer('relu_conv7_1'))(conv7_1)

    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name=namer('conv7_padding'))(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
        kernel_regularizer=kernel_regularizer, name=namer('conv15_2'),use_bias=False)(conv7_1)
    conv7_2 = BatchNormalization( momentum=0.99, epsilon=0.00001, name=namer('conv15_2/bn'))(conv7_2)
    conv7_2 = Activation('relu', name=namer('relu_conv7_2'))(conv7_2)


    print ('conv15 shape', conv7_2.shape)

    conv8_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal',
        kernel_regularizer=kernel_regularizer, name=namer('conv16_1'),use_bias=False)(conv7_2)
    conv8_1 = BatchNormalization( momentum=0.99, epsilon=0.00001, name=namer('conv16_1/bn'))(conv8_1)
    conv8_1 = Activation('relu', name=namer('relu_conv8_1'))(conv8_1)
    conv8_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name=namer('conv8_padding'))(conv8_1)
    conv8_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
        kernel_regularizer=kernel_regularizer, name=namer('conv16_2'),use_bias=False)(conv8_1)
    conv8_2 = BatchNormalization( momentum=0.99, epsilon=0.00001, name=namer('conv16_2/bn'))(conv8_2)
    conv8_2 = Activation('relu', name=namer('relu_conv8_2'))(conv8_2)

    print ('conv16 shape', conv8_2.shape)
    
    conv9_1 = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal',
        kernel_regularizer=kernel_regularizer, name=namer('conv17_1'),use_bias=False)(conv8_2)
    conv9_1 = BatchNormalization( momentum=0.99, epsilon=0.00001, name=namer('conv17_1/bn'))(conv9_1)
    conv9_1 = Activation('relu', name=namer('relu_conv9_1'))(conv9_1)
    conv9_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name=namer('conv9_padding'))(conv9_1)
    conv9_2 = Conv2D(128, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
        kernel_regularizer=kernel_regularizer, name=namer('conv17_2'),use_bias=False)(conv9_1)
    conv9_2 = BatchNormalization( momentum=0.99, epsilon=0.00001, name=namer('conv17_2/bn'))(conv9_2)
    conv9_2 = Activation('relu', name=namer('relu_conv9_2'))(conv9_2)

    print ('conv17 shape', conv9_2.shape)

    # Feed conv4_3 into the L2 normalization layer
    # conv4_3_norm = L2Normalization(gamma_init=20, name=namer('conv4_3_norm'))(conv4_3_norm)
   

    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=kernel_regularizer, name=namer('conv11_mbox_conf'))(conv4_3_norm)

    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=kernel_regularizer, name=namer('conv13_mbox_conf'))(fc7)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=kernel_regularizer, name=namer('conv14_2_mbox_conf'))(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=kernel_regularizer, name=namer('conv15_2_mbox_conf'))(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=kernel_regularizer, name=namer('conv16_2_mbox_conf'))(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=kernel_regularizer, name=namer('conv17_2_mbox_conf'))(conv9_2)
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=kernel_regularizer, name=namer('conv11_mbox_loc'))(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=kernel_regularizer, name=namer('conv13_mbox_loc'))(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=kernel_regularizer, name=namer('conv14_2_mbox_loc'))(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=kernel_regularizer, name=namer('conv15_2_mbox_loc'))(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=kernel_regularizer, name=namer('conv16_2_mbox_loc'))(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=kernel_regularizer, name=namer('conv17_2_mbox_loc'))(conv9_2)

    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                             aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                             this_offsets=offsets[0], limit_boxes=limit_boxes,
                                             variances=variances, pred_coords=pred_coords, normalize_coords=normalize_coords,
                                             name=namer('conv4_3_norm_mbox_priorbox'))(conv4_3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                    aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                                    limit_boxes=limit_boxes,
                                    variances=variances, pred_coords=pred_coords, normalize_coords=normalize_coords,
                                    name=namer('fc7_mbox_priorbox'))(fc7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                                        aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                        this_offsets=offsets[2], limit_boxes=limit_boxes,
                                        variances=variances, pred_coords=pred_coords, normalize_coords=normalize_coords,
                                        name=namer('conv6_2_mbox_priorbox'))(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                                        aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                        this_offsets=offsets[3], limit_boxes=limit_boxes,
                                        variances=variances, pred_coords=pred_coords, normalize_coords=normalize_coords,
                                        name=namer('conv7_2_mbox_priorbox'))(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                                        aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                        this_offsets=offsets[4], limit_boxes=limit_boxes,
                                        variances=variances, pred_coords=pred_coords, normalize_coords=normalize_coords,
                                        name=namer('conv8_2_mbox_priorbox'))(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                                        aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                        this_offsets=offsets[5], limit_boxes=limit_boxes,
                                        variances=variances, pred_coords=pred_coords, normalize_coords=normalize_coords,
                                        name=namer('conv9_2_mbox_priorbox'))(conv9_2_mbox_loc)

    anchor_box_sizes = [conv4_3_norm_mbox_priorbox._shape_tuple()[1:3],
                       fc7_mbox_priorbox._shape_tuple()[1:3],
                       conv6_2_mbox_priorbox._shape_tuple()[1:3],
                       conv7_2_mbox_priorbox._shape_tuple()[1:3],
                       conv8_2_mbox_priorbox._shape_tuple()[1:3],
                       conv9_2_mbox_priorbox._shape_tuple()[1:3]]

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name=namer('conv4_3_norm_mbox_conf_reshape'))(
        conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name=namer('fc7_mbox_conf_reshape'))(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name=namer('conv6_2_mbox_conf_reshape'))(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name=namer('conv7_2_mbox_conf_reshape'))(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name=namer('conv8_2_mbox_conf_reshape'))(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name=namer('conv9_2_mbox_conf_reshape'))(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name=namer('conv4_3_norm_mbox_loc_reshape'))(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name=namer('fc7_mbox_loc_reshape'))(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name=namer('conv6_2_mbox_loc_reshape'))(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name=namer('conv7_2_mbox_loc_reshape'))(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name=namer('conv8_2_mbox_loc_reshape'))(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name=namer('conv9_2_mbox_loc_reshape'))(conv9_2_mbox_loc)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name=namer('conv4_3_norm_mbox_priorbox_reshape'))(
        conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name=namer('fc7_mbox_priorbox_reshape'))(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name=namer('conv6_2_mbox_priorbox_reshape'))(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name=namer('conv7_2_mbox_priorbox_reshape'))(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name=namer('conv8_2_mbox_priorbox_reshape'))(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name=namer('conv9_2_mbox_priorbox_reshape'))(conv9_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name=namer('mbox_conf'))([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name=namer('mbox_loc'))([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name=namer('mbox_priorbox'))([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name=namer('mbox_conf_softmax'))(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name=namer('predictions'))([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    model = Model(inputs=x, outputs=predictions)
    model.anchor_box_sizes = anchor_box_sizes
    model.ssd_config = ssd_config
    model.type = net.MobilenetSSDType(mobilenet_version=1, ssd_lite=False, style='keras')


    # return model

    #predictor_sizes

    if mode == 'inference':
        print ('in inference mode')
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=0.01,
                                                   iou_threshold=0.45,
                                                   top_k=100,
                                                   nms_max_output_size=100,
                                                   pred_coords='centroids',
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name=namer('decoded_predictions'))(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        print ('in training mode')

    return model


def mobilenet(input_tensor, namer=None):
    if input_tensor is None:
        input_tensor = Input(shape=(300, 300, 3))

    v1 = '_v1'
    if namer is None:
        namer = lambda x: x

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=namer('conv1_padding'))(input_tensor)
    x = Convolution2D(32, (3, 3), strides=(2, 2), padding='valid', use_bias=False, name=namer("conv0"))(x)

    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv0/bn"))(x)

    x = Activation('relu')(x)

    x = DepthwiseConvolution2D((3, 3), strides=(1, 1), padding='same', use_bias=False, name=namer("conv1/dw"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv1/dw/bn"))(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, (1, 1), strides=(1, 1), padding='same', use_bias=False, name=namer("conv1"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv1/bn"))(x)
    x = Activation('relu')(x)

    print("conv1 shape: ", x.shape)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=namer('conv2_padding'))(x)
    x = DepthwiseConvolution2D((3, 3), strides=(2, 2), padding='valid', use_bias=False, name=namer("conv2/dw"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv2/dw/bn"))(x)
    x = Activation('relu')(x)
    x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False, name=namer("conv2"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv2/bn"))(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D((3, 3), strides=(1, 1), padding='same', use_bias=False, name=namer("conv3/dw"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv3/dw/bn"))(x)
    x = Activation('relu')(x)
    x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False, name=namer("conv3"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv3/bn"))(x)
    x = Activation('relu')(x)

    print("conv3 shape: ", x.shape)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=namer('conv3_padding'))(x)
    x = DepthwiseConvolution2D((3, 3), strides=(2, 2), padding='valid', use_bias=False, name=namer("conv4/dw"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv4/dw/bn"))(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, name=namer("conv4"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv4/bn"))(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D((3, 3), strides=(1, 1), padding='same', use_bias=False, name=namer("conv5/dw"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv5/dw/bn"))(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, name=namer("conv5"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv5/bn"))(x)
    x = Activation('relu')(x)

    print("conv5 shape: ", x.shape)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=namer('conv4_padding'))(x)
    x = DepthwiseConvolution2D((3, 3), strides=(2, 2), padding='valid', use_bias=False, name=namer("conv6/dw"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv6/dw/bn"))(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, name=namer("conv6"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv6/bn"))(x)
    x = Activation('relu')(x)

    test = x

    for i in range(5):
        x = DepthwiseConvolution2D((3, 3), strides=(1, 1), padding='same', use_bias=False,
                                   name=("conv" + str(7 + i) + "/dw" + v1))(x)
        x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv" + str(7 + i) + "/dw/bn"))(x)
        x = Activation('relu')(x)
        x = Convolution2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, name=namer("conv" + str(7 + i)))(
            x)
        x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv" + str(7 + i) + "/bn"))(x)
        x = Activation('relu')(x)

    # print ("conv11 shape: ", x.shape)
    conv11 = x

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=namer('conv5_padding'))(x)
    x = DepthwiseConvolution2D( (3, 3), strides=(2, 2), padding='valid', use_bias=False, name=namer("conv12/dw"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv12/dw/bn"))(x)
    x = Activation('relu')(x)
    x = Convolution2D(1024, (1, 1), strides=(1, 1), padding='same', use_bias=False, name=namer("conv12"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv12/bn"))(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D( (3, 3), strides=(1, 1), padding='same', use_bias=False, name=namer("conv13/dw"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv13/dw/bn"))(x)
    x = Activation('relu')(x)
    x = Convolution2D(1024, (1, 1), strides=(1, 1), padding='same', use_bias=False, name=namer("conv13"))(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=namer("conv13/bn"))(x)
    x = Activation('relu')(x)

    conv13 = x

    # print ("conv13 shape: ", x.shape)

    # model = Model(inputs=input_tensor, outputs=x)

    return [conv11, conv13, test]



class AnchorBoxes(keras.layers.Layer):
    '''
    A Keras layer to create an output tensor containing anchor box coordinates
    and variances based on the input tensor and the passed arguments.

    A set of 2D anchor boxes of different aspect ratios is created for each spatial unit of
    the input tensor. The number of anchor boxes created per unit depends on the arguments
    `aspect_ratios` and `two_boxes_for_ar1`, in the default case it is 4. The boxes
    are parameterized by the coordinate tuple `(xmin, xmax, ymin, ymax)`.

    The logic implemented by this layer is identical to the logic in the module
    `ssd_box_encode_decode_utils.py`.

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
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 two_boxes_for_ar1=True,
                 this_steps=None,
                 this_offsets=None,
                 limit_boxes=True,
                 variances=[1.0, 1.0, 1.0, 1.0],
                 pred_coords='centroids',
                 normalize_coords=False,
                 **kwargs):
        '''
        All arguments need to be set to the same values as in the box encoding process, otherwise the behavior is undefined.
        Some of these arguments are explained in more detail in the documentation of the `SSDBoxEncoder` class.

        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
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
            pred_coords (str, optional): The box coordinate format to be used. Can be either 'centroids' for the format
                `(cx, cy, w, h)` (box center coordinates, width, and height) or 'minmax' for the format
                `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids'.
            normalize_coords (bool, optional): Set to `True` if the model uses relative instead of absolute coordinates,
                i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates. Defaults to `False`.
        '''
        if K.backend() != 'tensorflow':
            raise TypeError("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))

        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError("`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(this_scale, next_scale))

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.limit_boxes = limit_boxes
        self.variances = variances
        self.pred_coords = pred_coords
        self.normalize_coords = normalize_coords
        # Compute the number of boxes per cell
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [keras.layers.InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        Return an anchor box tensor based on the shape of the input tensor.

        The logic implemented here is identical to the logic in the module `ssd_box_encode_decode_utils.py`.

        Note that this tensor does not participate in any graph computations at runtime. It is being created
        as a constant once during graph creation and is just being output along with the rest of the model output
        during runtime. Because of this, all logic is implemented as Numpy array operations and it is sufficient
        to convert the resulting Numpy array into a Keras tensor at the very end before outputting it.

        Arguments:
            x (tensor): 4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
                or `(batch, height, width, channels)` if `dim_ordering = 'tf'`. The input for this
                layer must be the output of the localization predictor layer.
        '''

        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                # Compute the regular anchor box for aspect ratio 1.
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # Compute one slightly larger version using the geometric mean of this scale value and the next.
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)

        # We need the shape of the input tensor
        if K.image_data_format() == 'channels_last':  # K.image_dim_ordering() == 'tf'
            batch_size, feature_map_height, feature_map_width, feature_map_channels = x._keras_shape
        else: # Not yet relevant since TensorFlow is the only supported backend right now, but it can't harm to have this in here for the future
            batch_size, feature_map_channels, feature_map_height, feature_map_width = x._keras_shape

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        if (self.this_steps is None):
            step_height = self.img_height / feature_map_height
            step_width = self.img_width / feature_map_width
        else:
            if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        if (self.this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets
        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # If `limit_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if self.limit_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 2]] = x_coords
            y_coords = boxes_tensor[:,:,:,[1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1, 3]] = y_coords

        # If `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth.
        if self.pred_coords == 'centroids':
            # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids')
        elif self.pred_coords == 'minmax':
            # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax')

        if True:
            # Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
            # as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
            variances_tensor = np.zeros_like(boxes_tensor) # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
            variances_tensor += self.variances # Long live broadcasting
            # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
            boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        if K.image_data_format() == 'channels_last': #K.image_dim_ordering() == 'tf':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        else: # Not yet relevant since TensorFlow is the only supported backend right now, but it can't harm to have this in here for the future
            batch_size, feature_map_channels, feature_map_height, feature_map_width = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'limit_boxes': self.limit_boxes,
            'variances': list(self.variances),
            'pred_coords': self.pred_coords,
            'normalize_coords': self.normalize_coords
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    # model = mobilenet(None)

    # for layer in model.layers:
    #     print (layer.name)


    img_height = 300  # Height of the input images
    img_width = 300  # Width of the input images
    img_channels = 3  # Number of color channels of the input images
    subtract_mean = [123, 117, 104]  # The per-channel mean of the images in the dataset
    swap_channels = True  # The color channel order in the original SSD is BGR
    n_classes = 2  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
    scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
                  1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
    scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
                   1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
    scales = scales_coco
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
    two_boxes_for_ar1 = True
    steps = [8, 16, 32, 64, 100, 300]  # The space between two adjacent anchor box center points for each predictor layer.
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
               0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
    limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
    variances = [0.1, 0.1, 0.2,
                 0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
    pred_coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
    normalize_coords = True

    # 1: Build the Keras model

    K.clear_session()  # Clear previous models from memory.

    model = mobilenet_v1_ssd_300("training",
                    image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    limit_boxes=limit_boxes,
                    variances=variances,
                    pred_coords=pred_coords,
                    normalize_coords=normalize_coords,
                    subtract_mean=subtract_mean,
                    divide_by_stddev=None,
                    swap_channels=swap_channels)













