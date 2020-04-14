#from config import config
import numpy as np
import tensorflow.compat.v1 as tf
import keras.regularizers
import gtc
import networks
import utility_scripts.weights_readers as readers
import quantization.quant_utils as quant_utl
from ssd import ssd_utils as ssd_utl
from ssd.layers.anchor_boxes import AnchorBoxes
from ssd.layers.l2_normalization import L2Normalization
from networks import net_utils as net
from collections import namedtuple
from networks.VGG import VGG16

import ssd.ssd_info as ssd
#from ssd.anchor_boxes_tf.multiple_grid_anchor_generator import create_ssd_anchors
#import re

Conv = namedtuple('Conv', ['kernel', 'filters', 'stride', 'padding'])

def get_extra_ssd_layer_specs(ssd_defaults):
    filters = ssd_defaults.extra_layer_filters
    strides = ssd_defaults.extra_layer_strides
    paddings = ssd_defaults.extra_layer_paddings

    ssd_layers_specs = [
        Conv(filters=filters[0], kernel=1, stride=1, padding='same'),
        Conv(filters=filters[1], kernel=3, stride=strides[0], padding=paddings[0]),  # ==> feature layer # 3

        Conv(filters=filters[2], kernel=1, stride=1, padding='same'),
        Conv(filters=filters[3], kernel=3, stride=strides[1], padding=paddings[1]),  # ==> feature layer # 4

        Conv(filters=filters[4], kernel=1, stride=1, padding='same'),
        Conv(filters=filters[5], kernel=3, stride=strides[2], padding=paddings[2]),  # ==> feature layer # 5

        Conv(filters=filters[6], kernel=1, stride=1, padding='same'),
        Conv(filters=filters[7], kernel=3, stride=strides[3], padding=paddings[3]),  # ==> feature layer # 6
    ]

    return ssd_layers_specs



def SSDNet(config, x_placeholder, ssd_config, ssd_quantizers=None):


    ssd_model_type = config.net_type
    ssd_defaults = net.SSDDefaults(ssd_model_type, config)

    base_net_type = ssd_model_type.base_net_type
    if isinstance(base_net_type, net.MobilenetType):
    # Build the base network
        base_network_prefix = 'FeatureExtractor/MobilenetV%d/' % base_net_type.version
        default_name = 'MobileNet_SSD_GTC'

    elif base_net_type.name.lower() == 'vgg':
        base_network_prefix = 'FeatureExtractor/VGG%d/' % base_net_type.num_layers
        default_name = 'SSD_GTC'

    else:
        raise ValueError('Unrecognized base network : %s ' % str(base_net_type))

    network_name = net.get_network_full_name_with_tag(
        config, default=default_name)


    print('Building base network %s ...' % str(base_net_type))

    # Make modifications to the default config for both base & SSD layers
    config.activation = ssd_defaults.activation
    config.explicit_zero_prepadding_for_stride2 = ssd_defaults.explicit_zero_prepadding_for_stride2
    config.bn_epsilon = ssd_defaults.bn_kwargs['epsilon']
    config.bn_momentum = ssd_defaults.bn_kwargs['momentum']
    config.do_batchnorm = ssd_defaults.do_batchnorm

    # Get config for just the base network (and make necessary adjustments)
    base_config = config.copy()
    base_config.network_name = network_name + base_network_prefix
    base_config.resize_inputs = False  # Resizing will be done by the SSD model.
    base_config.include_preprocessing = False  # Preprocessing will be done in the SSD model.
    if config.ssd_extra_quant:
        # If we are quantizing the extra layers of the SSD model,
        # don't skip quantization on the final layers of the base model
        # base_config.quant_post_skip = 0
        assert base_config.quant_post_skip == 0

    base_network, feat_extractor_prefix, box_conv_kernel_size = None, None, None
    if isinstance(base_net_type, net.MobilenetType):
        # Allow base network to determine its own quantizers
        base_network = net.MobileNet(base_net_type) \
            (base_config, x_placeholder, all_quantizers=None)

        mobilenet_version = base_net_type.version
        if mobilenet_version == 1:
            feat_extractor_prefix = 'Conv2d_13_pointwise'
            box_conv_kernel_size = 1

        elif mobilenet_version == 2:
            feat_extractor_prefix = 'layer_19'
            box_conv_kernel_size = 3

        elif mobilenet_version == 3:
            feat_extractor_prefix = 'layer_17'
            box_conv_kernel_size = 3


    elif base_net_type.name.lower() == 'vgg':
        base_config.fully_convolutional = True
        base_config.do_batchnorm = False
        base_config.do_dropout = False
        base_network = VGG16(base_config, x_placeholder)
        feat_extractor_prefix = ''
        box_conv_kernel_size = 3

    else:
        raise ValueError('Unrecognized base network : %s ' % str(base_net_type))

    config.quant_pre_skip = 0  # First layer quant skipped in base network

    # Determine parameters for SSD network.
    depthwise_feature_extractor = ssd_defaults.depthwise_feature_extractor
    depthwise_box_predictors = ssd_defaults.depthwise_box_predictors

    extra_feature_layers_specs = get_extra_ssd_layer_specs(ssd_defaults)
    n_predictor_layers = ssd_config.n_predictor_layers  # 6

    box_predictors_share_quantizers=config.ssd_box_share_quant # True

    extra_feat_n_quant = len(extra_feature_layers_specs) if not depthwise_feature_extractor else \
        len(extra_feature_layers_specs) * 1.5  # only second stages (half) are depthwise

    if box_predictors_share_quantizers:
        box_pred_n_quant = 1  # same quantizer for all the box predictors (usually identity anyway)
        if not config.ssd_box_quant:
            config.quant_post_skip = 1   # use Identity for final quantizer(for all boxes)
    else:
        box_pred_n_quant = (n_predictor_layers * 2) * 2 if depthwise_box_predictors else \
            (n_predictor_layers * 2)
        if not config.ssd_box_quant:   # use Identity for all final quantizers (for all n boxes)
            config.quant_post_skip = box_pred_n_quant

    num_quant = int(extra_feat_n_quant) + int(box_pred_n_quant)
    num_quant_in_base = base_network.userdata['num_quant']
    if ssd_quantizers is None:
        if not config.ssd_extra_quant:
            config.identity_quantization = True

        ssd_quantizers = quant_utl.QuantizerSet(
            num_quant, config=config, prefix=network_name,
            idx_start=num_quant_in_base)  # each rep contains 3 convolutional layers, plus 3 (1 before and 2 after all the blocks)

    print('SSD quantizers:')
    ssd_quantizers.print_pattern()

    # Build the SSD network:
    model = gtc.GTCModel(name=network_name)
    model.type = ssd_model_type
    model.ssd_config = ssd_config
    model.add(x_placeholder, name='input')

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    #model.add( gtc.Lambda(lambda z: z, name=network_name + 'identity_layer') )

    if config.resize_inputs:
        model.add(gtc.Resize(new_shape=(config.image_size[0], config.image_size[1]),
                             name=network_name + 'Resize'))

    if config.include_preprocessing:
        model.add( gtc.Preprocess(op_list=ssd_defaults.preprocess_op_list,
                                  name=network_name + 'Preprocess') )



    #mobilenet_base.compile(verbose=True)
    #mobilenet_base.print_model()


    base_network_layers = base_network.get_layers()
    # Add modules from the base mobilenet model, until the pooling/flatten layers
    # Also keep track of total stride so that we can extract feature layers.
    total_stride = 1
    last_conv_output_stride16, last_conv_output_stride32 = None, None
    for layer_obj in base_network_layers:
        # Check if is a tensorflow placeholder ('Placeholder:0') or a keras placeholder ('input:0')
        # Also don't add the logits layer of the base network
        if 'Placeholder' in layer_obj.name or 'input' in layer_obj.name \
                or 'Logits' in layer_obj.name:
            continue

        if isinstance(layer_obj, (gtc.Flatten, gtc.Reshape)):
            break # Stop before adding any of these to the SSD model

        if isinstance(base_net_type, net.MobilenetType) and isinstance(layer_obj, gtc.Pooling):
            # For Mobilenet-SSD: stop before adding the final pooling module from
            # the end of the mobilenet architecture
            # For VGG-SSD: add the pooling modules (it is used instead of strided convolutions)
            break

        if config.freeze_classification_layers:
            layer_obj.trainable = False

        model.add(layer_obj)

        if isinstance(layer_obj, gtc.ZeroPadding2D):
            # Add this to the model, but don't use it as an output feature layer
            continue

        elif isinstance(layer_obj, (gtc.Conv2D, gtc.Conv2D_bn)):
            total_stride *= layer_obj._strides[0]
        elif isinstance(layer_obj, (gtc.Pooling)):
            total_stride *= layer_obj._layer_config['strides']

        if total_stride == ssd_defaults.feat_extractor_strides[0]:
            last_conv_output_stride16 = layer_obj.name # ==> feature layer # 1
        elif total_stride == ssd_defaults.feat_extractor_strides[1]:
            last_conv_output_stride32 = layer_obj.name # ==> feature layer # 2


    conv_feature_output_names = []
    conv_feature_output_names.append(last_conv_output_stride16)
    conv_feature_output_names.append(last_conv_output_stride32)
    #print('c4 = %s' % last_conv_output_stride16)
    #print('c5 = %s' % last_conv_output_stride32)

    # Extra feature extractor layers after ones taken from mobilenet backbone
    # These get added to the top of the mobilenet network:
    ssd_namer = LayerNamer(prefix=network_name, style=base_net_type.name,
                           feat_extractor_prefix=base_network_prefix + feat_extractor_prefix)

    print('Adding an extra %d layers to base network for feature extraction ...' % len(extra_feature_layers_specs))

    conv_kwargs = ssd_defaults.conv_kwargs
    bn_kwargs = ssd_defaults.bn_kwargs

    #config.explicit_zero_prepadding_for_stride2 = False
    step_activations = {1: ssd_defaults.feat_extractor_activation_1,
                        2: ssd_defaults.feat_extractor_activation_2}

    for spec_idx, layer_spec in enumerate(extra_feature_layers_specs):
        stage_id = (spec_idx // 2) + 2  #  2,2,  3,3,  4,4,  5,5
        step_id = (spec_idx % 2) + 1    #  1,2   1,2,  1,2,  1,2

        use_bias = ssd_defaults.extra_layers_use_bias
        if depthwise_feature_extractor and layer_spec.kernel == 3:
            # for mobilenet-v2/v3, the extra layers are depthwise
            net.add_separable_conv_layers(model,
                filters=layer_spec.filters, kernel_size=layer_spec.kernel,
                strides=layer_spec.stride, use_bias=use_bias,
                activation=step_activations[step_id],
                name_dict_depth=ssd_namer.get('extra', stage=stage_id, step=step_id, depthwise=True),
                name_dict_point=ssd_namer.get('extra', stage=stage_id, step=step_id),
                quantizers=ssd_quantizers,
                config=config, **conv_kwargs, **bn_kwargs) # name='conv14_1',

        else:
            net.add_conv_bn_act_layers(
                model, layer_spec.filters, kernel_size=layer_spec.kernel,
                stride=layer_spec.stride, padding=layer_spec.padding,
                use_bias=use_bias, activation=step_activations[step_id],
                name_dict=ssd_namer.get('extra', stage=stage_id, step=step_id),
                quant_dict=ssd_quantizers.next(),
                config=config, **conv_kwargs, **bn_kwargs) # name='conv14_1',

        if step_id == 2:
            last_name = model.last_layer_name()
            conv_feature_output_names.append(last_name)


    # end loop over specs
    assert len(conv_feature_output_names) == n_predictor_layers, \
        "Should have %d feature tensors, but have %d" % (
            n_predictor_layers, len(conv_feature_output_names))

    n_boxes = ssd_config.n_boxes

    if ssd_defaults.do_L2_norm_on_first_feature_map:
        feature_norm_name = conv_feature_output_names[0].replace('relu', 'conv') + '_norm'
        model.add( L2Normalization(gamma_init=20, name=feature_norm_name,
                                   layer_input=conv_feature_output_names[0]) )
        conv_feature_output_names[0] = feature_norm_name


    #print('conv_feature_output_names : ', conv_feature_output_names)
    if box_predictors_share_quantizers:
        box_quant_dict=ssd_quantizers.next()
    else:
        box_quant_dict=None

    # 1 A:  Add classifiers for each box
    box_class_predictors = []
    for box_id in range(n_predictor_layers):
        # 1. Add ClassPredictor
        # Feed conv4_3 into the L2 normalization layer
        # conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3_norm)
        n_classes = ssd_config.n_classes + 1  # include background class
        bias_initializer = 'zeros'

        n_filters = n_boxes[box_id] * n_classes
        bias_for_background_class = bool(config.ssd_background_bias) # is not None

        if bias_for_background_class:
            # See RetinaNet paper: (https://arxiv.org/abs/1708.02002)
            prior_prob = 1 - config.ssd_background_bias
            assert 0 < prior_prob < 1, "invalid prior probability"
            log_prior_prob_bias = np.log((n_classes - 1) * (1 - prior_prob) / (prior_prob))
            bias_initializer = np.zeros(n_filters)
            put_bias_on_background = True   # shouldn't matter either way
            if put_bias_on_background:
                bias_initializer[::n_classes] = log_prior_prob_bias
            else:
                bias_initializer[:] = -log_prior_prob_bias
                bias_initializer[::n_classes] = 0
            # softmax(bias_initializer.reshape(-1, 21), axis=1) should now have
            # 0.99 for background and small values for all foreground classes

        if depthwise_box_predictors:
            if box_predictors_share_quantizers:
                quant_dict_depth, quant_dict_point = box_quant_dict, box_quant_dict
            else:
                quant_dict_depth = ssd_quantizers.next()
                quant_dict_point = ssd_quantizers.next()

            net.add_separable_conv_layers(
                model, filters=n_boxes[box_id] * n_classes,
                kernel_size=box_conv_kernel_size,  padding='same',
                activation1='relu6',
                name_dict_depth=ssd_namer.get('box_class', stage=box_id, depthwise=True),
                name_dict_point=ssd_namer.get('box_class', stage=box_id, depthwise=False),
                kwargs_depth=dict(use_bias=False, do_batchnorm=True),
                kwargs_point=dict(use_bias=True, do_batchnorm=False, bias_initializer=bias_initializer),
                quant_dict_depth=quant_dict_depth,
                quant_dict_point=quant_dict_point,
                layer_input=conv_feature_output_names[box_id],
                config=config, **conv_kwargs, **bn_kwargs)

        else:
            if box_predictors_share_quantizers:
                quant_dict = box_quant_dict
            else:
                quant_dict = ssd_quantizers.next()

            model.add(gtc.Conv2D(
                filters=n_boxes[box_id] * n_classes,
                kernel_size=box_conv_kernel_size, padding='same',
                **conv_kwargs,
                bias_initializer=bias_initializer,
                name=ssd_namer.get('box_class', stage=box_id)['conv'],
                quantizer=quant_dict['conv'],
                layer_input=conv_feature_output_names[box_id]) )

        # 1.B Reshape
        # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
        # We want the classes isolated in the last axis to perform softmax on them
        box_class_name_reshape = model.last_layer_name() + '_reshape'

        model.add( gtc.Reshape((-1, n_classes), name=box_class_name_reshape))

        box_class_predictors.append(box_class_name_reshape)

    # 1.C Concatenate all classifiers to mbox_class
    # Output shape of `mbox_class`: (batch, n_boxes_total, n_classes)
    mbox_class_name = network_name + 'mbox_class'
    model.add( gtc.Concatenate(box_class_predictors, axis=1, name= mbox_class_name) )
    if ssd_config.class_activation_in_network:
        mbox_class_name = network_name + 'mbox_class_softmax'
        model.add( gtc.Activation('softmax', name=mbox_class_name) )


    # 2.A:  Add box location encoders for each box
    box_encoding_predictors = []
    box_encoding_predictors_reshaped = []
    for box_id in range(n_predictor_layers):
        # 1. Add BoxEncodingPredictor

        # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
        # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`

        if depthwise_box_predictors:
            if box_predictors_share_quantizers:
                quant_dict_depth, quant_dict_point = box_quant_dict, box_quant_dict
            else:
                quant_dict_depth = ssd_quantizers.next()
                quant_dict_point = ssd_quantizers.next()

            net.add_separable_conv_layers(
                model, filters=n_boxes[box_id] * 4,
                kernel_size=box_conv_kernel_size, padding='same',
                activation1='relu6',
                name_dict_depth=ssd_namer.get('box_loc', stage=box_id, depthwise=True),
                name_dict_point=ssd_namer.get('box_loc', stage=box_id, depthwise=False),
                kwargs_depth=dict(use_bias=False, do_batchnorm=True),
                kwargs_point=dict(use_bias=True, do_batchnorm=False),
                quant_dict_depth=quant_dict_depth,
                quant_dict_point=quant_dict_point,
                layer_input=conv_feature_output_names[box_id],
                config=config, **conv_kwargs, **bn_kwargs)

        else:
            if box_predictors_share_quantizers:
                quant_dict = box_quant_dict
            else:
                quant_dict = ssd_quantizers.next()

            model.add( gtc.Conv2D(
                filters=n_boxes[box_id] * 4,
                kernel_size=box_conv_kernel_size, padding='same',
                name=ssd_namer.get('box_loc', stage=box_id)['conv'],
                quantizer=quant_dict['conv'],
                layer_input=conv_feature_output_names[box_id],
                **conv_kwargs) )

        box_encoding_predictors.append(model.last_layer_name())

        # 2.B Reshape
        # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
        # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
        box_loc_name_reshape = model.last_layer_name() + '_reshape'
        model.add( gtc.Reshape((-1, 4), name=box_loc_name_reshape) )

        box_encoding_predictors_reshaped.append(box_loc_name_reshape)

    # 2.C: Concatenate all box location encoders
    # Output shape of `mbox_encoding`: (batch, n_boxes_total, 4)
    mbox_encoding_name = network_name + 'mbox_encoding'
    model.add( gtc.Concatenate(box_encoding_predictors_reshaped,
                               axis=1, name= mbox_encoding_name) )


    # 3.A Anchor boxes
    ### Generate the anchor boxes (called "priors" in the original
    ### Caffe/C++ implementation, so I'll keep their layer names)
    #anchor_box_sizes ( for 300x300: = [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]

    anchor_box_names = []
    box_priors = []
    for box_id in range(ssd_config.n_predictor_layers):
        # Note: Anchors are never quantized

        # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
        anchors_name = box_encoding_predictors[box_id].replace('EncodingPredictor', 'Anchors')
        anchor_box_names.append(anchors_name)
        model.add( AnchorBoxes(ssd_config, box_id, name= anchors_name,
                                   layer_input=box_encoding_predictors[box_id]))

        # 3.B Reshape
        # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
        anchors_name_reshape = anchors_name + '_reshape'
        model.add( gtc.Reshape((-1, ssd_config.anchors_size),
                               name=anchors_name_reshape) )

        box_priors.append(anchors_name_reshape)

    # Store the names of the anchor boxes. When model is built, we will
    # retrieve these layer by name to see what their sizes are.
    model.anchor_box_names = anchor_box_names

    # 3.C:  Concatenate the class and box predictions and the anchors to one large predictions vector
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_anchors`: (batch, n_boxes_total, 8)
    #model.anchor_box_sizes = anchor_box_sizes

    mbox_anchors_name = network_name + 'mbox_anchors'
    model.add( gtc.Concatenate( box_priors, axis=1, name=mbox_anchors_name) )


    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 4/8)
    # 4 : Concatenate the results of 1.C, 2.C, 3.C
    model.add(
        gtc.Concatenate([mbox_class_name, mbox_encoding_name, mbox_anchors_name],
                        axis=2, name= network_name + 'predictions') )

    ssd_quantizers.confirm_done()

    model.nettask = 'detection'
    return model



class LayerNamer():
    # Class to provide names for each layer (to match saved weights in checkpoints / saved models)
    def __init__(self, prefix='', feat_extractor_prefix='FeatureExtractor/MobilenetV1/',
                 style='mobilenet',
                 block_id_start=0):
        self._block_id = block_id_start
        self._style = style
        self._prefix = prefix
        self._feat_extractor_prefix = feat_extractor_prefix

    def get(self, layer_type, **kwargs):



        # Individual layer names:
        if layer_type == 'extra':
            prefix = self._prefix + self._feat_extractor_prefix
            stage = kwargs.pop('stage')
            step = kwargs.pop('step')
            if self._style == 'mobilenet':
                is_depthwise = kwargs.pop('depthwise', False)
                s = kwargs.pop('spec', None)

                depthwise_str = '_depthwise' if is_depthwise else ''
                spec_str = ''
                if s is not None:
                    stride_str = '' if s.stride == 1 else '_s%d' % s.stride
                    spec_str ='_%dx%d%s_%d' % (
                        s.kernel, s.kernel, stride_str, s.filters)

                block_name = prefix + '_%d_Conv2d_%d%s%s' % \
                             (step, stage, spec_str, depthwise_str)

                name_dict = {'conv': block_name,
                             'batchnorm': block_name + '/BatchNorm',
                             'activation': block_name + '/Activation',
                             'zeropad': block_name + '/ZeroPadding'}

            elif self._style == 'vgg':
                stage_offset = 5 - 1  # stage_id starts at 2.
                conv_name = prefix + 'block%d_conv%d' % (stage + stage_offset, step)
                bn_name = prefix + 'block%d_bn%d' % (stage + stage_offset, step)
                act_name = prefix + 'block%d_relu%d' % (stage + stage_offset, step)

                name_dict = {'conv': conv_name,
                             'batchnorm': bn_name,
                             'activation': act_name,
                             'zeropad': conv_name + '_padding'}

            else:
                raise ValueError('Unrecognized style : %s' % self._style)


            return name_dict


        elif layer_type in ['box_class', 'box_loc']:
            prefix = self._prefix
            stage_id = kwargs['stage']
            is_depthwise = kwargs.pop('depthwise', False)

            depthwise_str = '_depthwise' if is_depthwise else ''
            layer_str = 'ClassPredictor' if layer_type == 'box_class' else 'BoxEncodingPredictor'

            block_name = 'BoxPredictor_%d/%s%s' % (stage_id, layer_str, depthwise_str)
            name_dict = {'conv': prefix + block_name,
                         'batchnorm': prefix + block_name + '/BatchNorm',
                         'activation': prefix + block_name + '/Activation',
                         'zeropad': prefix + block_name + '/ZeroPadding'}
            return name_dict


        #assert name_dict is not None
        #return name_dict


if __name__ == "__main__":
    # Test compilation of mobilenet-ssd
    import cv2
    from datasets.pascal import GTCPascalLoader
    import datasets
    from utility_scripts.sample_image_loader import SampleImageLoader
    import keras.backend as K
    K.set_learning_phase(False)

    imsize = (300, 300)
    img_height, img_width = imsize
    dataset_test = 'PascalVOC'
    if dataset_test == 'PascalVOC':
        loader = GTCPascalLoader(years=2012)
        class_names = ssd.get_pascal_class_names(include_bckg_class=True)

    elif dataset_test == 'COCO':
        raise NotImplementedError('Testing on COCO not implemented yet')

    else:
        raise ValueError('Unrecognized dataset name : %s' % dataset_test)


    ssd_mobilenet_v1_keras = net.MobilenetSSDType(mobilenet_version=1, ssd_lite=False, style='keras')

    imsize_with_channels = imsize + (3,)
    config = net.NetConfig(image_size=imsize_with_channels,
                           num_classes=len(class_names),
                           do_merged_batchnorm=False,
                           net_type=ssd_mobilenet_v1_keras)

    sess = tf.InteractiveSession()
    x_placeholder = tf.placeholder(
        tf.float32, [None] + list(config.image_size[:3]) )

    ssd_config = ssd_utl.SSDConfig(imsize=imsize, net_type=ssd_mobilenet_v1_keras)
    ssd_model = SSDNet(config, x_placeholder, ssd_config)
    ssd_model.compile(verbose=True)
    print('Succesfully compiled Mobilenet-SSD')

    ssd_model.print_model()
    ssd_model.initialize()


    # im = utl.get_sample_ssd_image()
    loader = SampleImageLoader(imsize=imsize[:2])
    img, _ = loader.load()

    test_batch_size = 1
    data_im_orig_size, data_annot = loader.load()

    data_im_resized_list = [cv2.resize(im, imsize) for im in data_im_orig_size]
    data_im_resized = np.asarray(data_im_resized_list)



    #img = np.array(PIL.Image.open('panda.jpg').resize((224, 224))).astype(np.float) / 128 - 1
    #img = np.expand_dims(img, axis=0)
    data_dict = {x_placeholder: data_im_resized}
    #ans1 = mobilenet.eval_one_example(data_dict)

    reader = net.get_weights_reader(ssd_mobilenet_v1_keras)
    reader.print_all_tensors()
    ssd_model.load_pretrained_weights_from_reader(reader, sess=sess)

    y_pred = ssd_model.predict_detections(data_dict)

    test_load_save_def = False
    test_weights_file = 'F:/models/test/ssd_mobilenet2_model.h5'
    test_model_file = test_weights_file.replace('.h5', '.json')
    test_weights_file_hp = test_weights_file.replace('.h5', '_hp.h5')
    test_weights_file_lp = test_weights_file.replace('.h5', '_lp.h5')
    if test_load_save_def:
        ssd_model.save_model_def(test_model_file)
        ssd_model.save_weights(test_weights_file)

        ssd_model.save_weights_for_keras_model(test_weights_file_hp, hp=True)
        ssd_model.save_weights_for_keras_model(test_weights_file_lp, hp=False)

        ssd_model3 = gtc.singlePrecisionKerasModel(model_file=test_model_file, weights_file=test_weights_file_hp, hp=True)
        ssd_model4 = gtc.singlePrecisionKerasModel(model_file=test_model_file, weights_file=test_weights_file_lp, hp=False)

        ssd_model2 = gtc.GTCModel(model_file=test_model_file, weights_file=test_weights_file)
        y_pred2 = ssd_model.predict_detections(data_im_resized)
        assert np.array_equal(y_pred, y_pred2)

    test_pure_keras_model = True
    if test_pure_keras_model:
        load_hp = True
        weights_file_single = test_weights_file_hp if load_hp else test_weights_file_lp
        ssd_model_keras = gtc.singlePrecisionKerasModel(
            model_file=test_model_file,weights_file=weights_file_single,hp=load_hp)

    y_pred_decoded_all = ssd_utl.decode_y(y_pred, ssd_config=ssd_config)

    test_load_save_json = True
    if test_load_save_json:
        test_weight_file = 'F:/models/test_ssd_mobilenet2_weights.h5'
        test_model_file = 'F:/models/test_ssd_mobilenet2_model.json'
        ssd_model.save_weights(test_weight_file, sess, verbose=True)
        ssd_model.save_model_def(test_model_file)

        model2 = gtc.GTCModel(model_file=test_model_file, weights_file=test_weight_file)
        pred_class_id2 = model2.predict_class(img, hp=True, top_k=1).flatten()[0]
        logits2 = model2.forward(img)
        #assert pred_class_id2 == pred_class_id0
        #assert np.array_equal(logits0, logits2)


    for im_idx in range(test_batch_size):
        orig_im = data_im_orig_size[im_idx]
        #y_pred_decoded = y_pred_decoded_all[im_idx]
        y_pred_decoded = y_pred_decoded_all

        for box in y_pred_decoded[0]:
            xmin, ymin, xmax, ymax = ssd_utl.scaleBoxPredToOrigImage(
                box[-4:], orig_im.shape, imsize)

            # print int(box[-4]), int(box[-2]) , int(box[-3]) , int(box[-1])
            print( xmin,xmax,ymin,ymax )
            cv2.rectangle(data_im_orig_size[0],(xmin, ymin), (xmax, ymax),(0,255,255),2)
            label = '%s[%.3f]' % (class_names[int(box[0].item())], box[1])
            thickness=1
            cv2.putText(orig_im, label, (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255),thickness)
            # putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None): # real signature unknown; restored from __doc__


        cv2.imshow("image1",orig_im[...,::-1])
        cv2.waitKey(0)

    a = 1
