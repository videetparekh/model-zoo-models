import re
from networks import net_types as net

def get_conv_bn_act_layer_names(
        prefix_gtc, prefix_tf=None, conv=True, bn=True, act=True,
        conv_name='/Conv2D', bn_name='/BatchNorm/batchnorm/add_1', act_name='/Relu6'):

    layers = []
    if prefix_tf is None:
        prefix_tf = prefix_gtc
    if conv:
        layers.append((prefix_gtc + '', prefix_tf + conv_name))
    if bn:
        layers.append((prefix_gtc + '/BatchNorm', prefix_tf + bn_name))
    if act:
        layers.append((prefix_gtc + '/Activation', prefix_tf + act_name))

    return layers





def rep(x):
    return x + '/' + x

#def keras_conv_bn_act_layer_names(name, conv=True, bn=True, act=True):


def get_mobilenet_v1_activations(args):

    all_layers = []

    if args.net_style == 'tensorflow':

        layer_name_prefix = 'MobilenetV1/'
        gtc_prefix = layer_name_prefix
        tf_prefix = layer_name_prefix
        tf_prefix_rep = layer_name_prefix + 'MobilenetV1/'
        layers_initial = get_conv_bn_act_layer_names(gtc_prefix + 'Conv2d_0', tf_prefix_rep + 'Conv2d_0',
                                                     bn_name='/BatchNorm/FusedBatchNorm')
        all_layers.extend(layers_initial)
        n_separable_layers = 13
        for layer_id in range(1, n_separable_layers + 1):
            depth_layer_str = 'Conv2d_%d_depthwise' % layer_id
            point_layer_str = 'Conv2d_%d_pointwise' % layer_id
            all_layers.extend(get_conv_bn_act_layer_names(gtc_prefix + depth_layer_str, tf_prefix_rep + depth_layer_str,
                                                          conv_name='/depthwise/', bn_name='/BatchNorm/FusedBatchNorm'))
            all_layers.extend(get_conv_bn_act_layer_names(gtc_prefix + point_layer_str, tf_prefix_rep + point_layer_str,
                                                          bn_name='/BatchNorm/FusedBatchNorm'))

            all_layers.extend([('MobilenetV1/AvgPool', 'MobilenetV1/Logits/AvgPool_1a/AvgPool'),
                               ('MobilenetV1/Logits/Conv2d_1c_1x1', 'MobilenetV1/Predictions/Reshape_1')])
    else:
        all_layers = []
        gtc_prefix = 'MobilenetV1/'
        #gtc_prefix = ''

        layers_initial = get_conv_bn_act_layer_names(gtc_prefix + 'Conv2d_0', 'conv1',
                                                     conv_name='', bn_name='_bn', act_name='_relu')
        all_layers.extend(layers_initial)
        n_separable_layers = 13
        for layer_id in range(1, n_separable_layers + 1):
            depth_layer_str = 'Conv2d_%d_depthwise' % layer_id
            point_layer_str = 'Conv2d_%d_pointwise' % layer_id
            all_layers.extend(get_conv_bn_act_layer_names(gtc_prefix + depth_layer_str, 'conv_dw_%d' % layer_id,
                                                          conv_name='', bn_name='_bn', act_name='_relu'))

            all_layers.extend(get_conv_bn_act_layer_names(gtc_prefix + point_layer_str, 'conv_pw_%d' % layer_id,
                                                          conv_name='', bn_name='_bn', act_name='_relu'))

        all_layers.extend([(gtc_prefix + 'AvgPool', 'global_average_pooling2d_1'),
                           (gtc_prefix + 'Logits/Conv2d_1c_1x1', 'conv_preds')])

    if args.net_style == 'tensorflow':
        layers_chk_old = [
            ('MobilenetV1/Conv2d_0', 'MobilenetV1/MobilenetV1/Conv2d_0/Conv2D'),
            ('MobilenetV1/Conv2d_0/BatchNorm', 'MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm'),
            ('MobilenetV1/Conv2d_0/Activation', 'MobilenetV1/MobilenetV1/Conv2d_0/Relu6'),
            ('MobilenetV1/Conv2d_1_depthwise', 'MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise'),
            ('MobilenetV1/Conv2d_1_depthwise/BatchNorm',
             'MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNorm'),
            ('MobilenetV1/Conv2d_1_depthwise/Activation', 'MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6'),
            ('MobilenetV1/Conv2d_1_depthwise', 'MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise'),
            ('MobilenetV1/Conv2d_1_depthwise/BatchNorm',
             'MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNorm'),
            ('MobilenetV1/Conv2d_1_depthwise/Activation', 'MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6'),
            ('MobilenetV1/Conv2d_13_pointwise/Activation', 'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6'),
            ('MobilenetV1/AvgPool', 'MobilenetV1/Logits/AvgPool_1a/AvgPool'),
            ('MobilenetV1/Logits/Conv2d_1c_1x1', 'MobilenetV1/Predictions/Reshape_1')
        ]

    else:

        layers_chk2 = [
            ('MobilenetV1/Conv2d_0', 'conv1'),
            ('MobilenetV1/Conv2d_0/BatchNorm', 'conv1_bn'),
            ('MobilenetV1/Conv2d_0/Activation', 'conv1_relu'),
            ('MobilenetV1/Conv2d_1_depthwise', 'conv_dw_1'),
            ('MobilenetV1/Conv2d_2_depthwise', 'conv_dw_2'),
            ('MobilenetV1/Conv2d_13_depthwise', 'conv_dw_13'),
            ('MobilenetV1/Conv2d_1_pointwise', 'conv_pw_1'),
            ('MobilenetV1/Conv2d_13_pointwise', 'conv_pw_13'),
            ('MobilenetV1/Logits/Conv2d_1c_1x1', 'conv_preds')
        ]

    return all_layers


def get_mobilenet_v2_activations(args):

    all_layers = []

    if args.net_style == 'tensorflow':
        layer_prefix = 'MobilenetV2/'
        #layer_prefix = ''
        all_layers.extend(get_conv_bn_act_layer_names(layer_prefix + 'Conv', layer_prefix + 'Conv',
                                                      bn_name='/BatchNorm/FusedBatchNorm'))

        n_expansion_layers = 13
        for layer_id in range(0, n_expansion_layers + 1):
            layer_id_str = '' if layer_id == 0 else '_%d' % layer_id
            block_name = layer_prefix + 'expanded_conv' + layer_id_str + '/'

            all_layers.extend(get_conv_bn_act_layer_names(block_name + 'expand',
                                                          bn_name='/BatchNorm/FusedBatchNorm'))
            all_layers.extend(get_conv_bn_act_layer_names(block_name + 'depthwise',
                                                          conv_name='/depthwise', bn_name='/BatchNorm/FusedBatchNorm'))
            all_layers.extend(get_conv_bn_act_layer_names(block_name + 'project', act=False,
                                                          bn_name='/BatchNorm/FusedBatchNorm'))

        all_layers.extend(get_conv_bn_act_layer_names(layer_prefix + 'Conv_1', layer_prefix + 'Conv_1',
                                                      bn_name='/BatchNorm/FusedBatchNorm'))

        all_layers.extend([(layer_prefix + 'Conv_1/AvgPool', 'MobilenetV2/Logits/AvgPool'),
                           (layer_prefix + 'Logits/Conv2d_1c_1x1', 'MobilenetV2/Predictions/Softmax')])

    else:
        all_layers = []
        gtc_prefix = 'MobilenetV2/'
        layers_initial = [(gtc_prefix + 'Conv', 'Conv1'),
                          (gtc_prefix + 'Conv/BatchNorm', 'bn_Conv1'),
                          (gtc_prefix + 'Conv/Activation', 'Conv1_relu')]

        all_layers.extend(layers_initial)

        n_expansion_layers = 16
        for layer_id in range(0, n_expansion_layers + 1):
            layer_id_str = '' if layer_id == 0 else '_%d' % layer_id
            gtc_block = gtc_prefix + 'expanded_conv' + layer_id_str + '/'
            if layer_id == 0:
                keras_block = 'expanded_conv_'
                do_expand = False
            else:
                keras_block = 'block_%d_' % layer_id
                do_expand = True

            if do_expand:
                all_layers.extend(get_conv_bn_act_layer_names(gtc_block + 'expand', keras_block + 'expand',
                                                              conv_name='', bn_name='_BN', act_name='_relu'))

            all_layers.extend(get_conv_bn_act_layer_names(gtc_block + 'depthwise', keras_block + 'depthwise',
                                                          conv_name='', bn_name='_BN', act_name='_relu'))
            all_layers.extend(get_conv_bn_act_layer_names(gtc_block + 'project', keras_block + 'project', act=False,
                                                          conv_name='', bn_name='_BN', act_name='_relu'))

        all_layers.extend([(gtc_prefix + 'Conv_1', 'Conv_1'),
                           (gtc_prefix + 'Conv_1/BatchNorm', 'Conv_1_bn'),
                           (gtc_prefix + 'Conv_1/Activation', 'out_relu'),
                           (gtc_prefix + 'Conv_1/AvgPool', 'global_average_pooling2d_1'),
                           (gtc_prefix + 'Logits/Conv2d_1c_1x1', 'Logits')])

    if args.net_style == 'tensorflow':

        layers_chk2 = [
            ('MobilenetV2/Conv', 'MobilenetV2/Conv/Conv2D'),
            ('MobilenetV2/Conv/BatchNorm', 'MobilenetV2/Conv/BatchNorm/FusedBatchNorm'),
            ('MobilenetV2/Conv/Activation', 'MobilenetV2/Conv/Relu6'),
            ('MobilenetV2/expanded_conv/depthwise', 'MobilenetV2/expanded_conv/depthwise/depthwise'),
            ('MobilenetV2/expanded_conv/depthwise/BatchNorm',
             'MobilenetV2/expanded_conv/depthwise/BatchNorm/FusedBatchNorm'),
            ('MobilenetV2/expanded_conv/project', 'MobilenetV2/expanded_conv/project/Conv2D'),
            ('MobilenetV2/expanded_conv_3/expand', 'MobilenetV2/expanded_conv_3/expand/Conv2D'),
            ('MobilenetV2/expanded_conv_7/expand', 'MobilenetV2/expanded_conv_7/expand/Conv2D'),
            ('MobilenetV2/expanded_conv_15/expand', 'MobilenetV2/expanded_conv_15/expand/Conv2D'),
            ('MobilenetV2/Conv_1', 'MobilenetV2/Conv_1/Conv2D'),
            ('MobilenetV2/Conv_1/AvgPool', 'MobilenetV2/Logits/AvgPool'),
            ('MobilenetV2/Logits/Conv2d_1c_1x1', 'MobilenetV2/Logits/output'),
        ]

    else:
        layers_chk = [
            ('MobilenetV2/Conv', 'Conv1'),
            ('MobilenetV2/Conv/BatchNorm', 'bn_Conv1'),
            ('MobilenetV2/expanded_conv/depthwise', 'expanded_conv_depthwise'),
            ('MobilenetV2/expanded_conv/depthwise/BatchNorm', 'expanded_conv_depthwise_BN'),
            ('MobilenetV2/expanded_conv/project', 'expanded_conv_project'),
            ('MobilenetV2/expanded_conv/project/BatchNorm', 'expanded_conv_project_BN'),
            ('MobilenetV2/expanded_conv_1/expand', 'block_1_expand'),
            ('MobilenetV2/expanded_conv_1/expand/BatchNorm', 'block_1_expand_BN'),
            ('MobilenetV2/expanded_conv_1/depthwise', 'block_1_depthwise'),
            ('MobilenetV2/expanded_conv_1/depthwise/BatchNorm', 'block_1_depthwise_BN'),
            ('MobilenetV2/expanded_conv_1/project', 'block_1_project'),
            ('MobilenetV2/expanded_conv_1/project/BatchNorm', 'block_1_project_BN'),
            ('MobilenetV2/block_2_merge', 'block_2_add'),
            ('MobilenetV2/block_4_merge', 'block_4_add'),
            ('MobilenetV2/block_7_merge', 'block_7_add'),
            ('MobilenetV2/block_14_merge', 'block_14_add'),
            ('MobilenetV2/Conv_1/BatchNorm', 'Conv_1_bn'),
            ('MobilenetV2/Conv_1/Activation', 'out_relu'),
            ('MobilenetV2/Conv_1/AvgPool', 'global_average_pooling2d_1'),
            ('MobilenetV2/Logits/Conv2d_1c_1x1', 'Logits')]

        if args.do_merged_batchnorm:
            layers_chk = [
                ('Conv1.bn_Conv1', 'bn_Conv1'),
                ('mobl0_conv_0_depthwise.bn0_conv_0_bn_depthwise', 'expanded_conv_depthwise_BN'),
                ('mobl0_conv_0_project.bn0_conv_0_bn_project', 'expanded_conv_project_BN'),
                ('block_2_merge', 'block_2_add'),
                ('block_4_merge', 'block_4_add'),
                ('block_7_merge', 'block_7_add'),
                ('block_14_merge', 'block_14_add'),
                ('average_pooling2d_1', 'global_average_pooling2d_1'),
                ('Logits', 'Logits')
            ]

    return all_layers


def get_mobilenet_v3_activations(config):
    all_layers = []

    if config.net_style == 'tensorflow':
        gtc_prefix = 'Mobilenet_V3_GTC/'
        tf_prefix = 'MobilenetV3/'
        all_layers.extend(get_conv_bn_act_layer_names(gtc_prefix + 'Conv', tf_prefix + 'Conv',
                                                      bn_name='/BatchNorm/FusedBatchNorm', act_name='/hard_swish/mul_1'))

        n_expansion_layers = 14
        for layer_id in range(0, n_expansion_layers + 1):
            layer_id_str = '' if layer_id == 0 else '_%d' % layer_id
            block_name = 'expanded_conv' + layer_id_str + '/'
            act_name = '/hard_swish/mul_1' if layer_id >= 6 else '/Relu'
            do_squeeze_excite = layer_id in [3,4,5, 10,11,12,13,14]

            do_expand = layer_id > 0
            if do_expand:
                all_layers.extend(get_conv_bn_act_layer_names(gtc_prefix + block_name + 'expand',
                                                              tf_prefix  + block_name + 'expand',
                                                              bn_name='/BatchNorm/FusedBatchNorm', act_name=act_name))

            all_layers.extend(get_conv_bn_act_layer_names(gtc_prefix + block_name + 'depthwise',
                                                          tf_prefix + block_name + 'depthwise',
                                                          conv_name='/depthwise',
                                                          bn_name='/BatchNorm/FusedBatchNorm', act_name=act_name))
            if do_squeeze_excite:
                all_layers.append( (gtc_prefix + block_name + 'squeeze_excite',
                                    tf_prefix + block_name + 'squeeze_excite/mul' ) )

            all_layers.extend(get_conv_bn_act_layer_names(gtc_prefix + block_name + 'project',
                                                          tf_prefix+ block_name + 'project', act=False,
                                                          bn_name='/BatchNorm/FusedBatchNorm', act_name=act_name))

        all_layers.extend(get_conv_bn_act_layer_names(gtc_prefix + 'Conv_1',
                                                      tf_prefix + 'Conv_1',
                                                      bn_name='/BatchNorm/FusedBatchNorm', act_name='/hard_swish/mul_1'))

        all_layers.extend([(gtc_prefix + 'Conv_1/AvgPool',
                            tf_prefix + 'AvgPool2D/AvgPool') ])

        all_layers.extend(get_conv_bn_act_layer_names(gtc_prefix + 'Conv_2',
                                                      tf_prefix + 'Conv_2', bn=False,
                                                      conv_name='/BiasAdd', act_name='/hard_swish/mul_1'))

        all_layers.extend([(gtc_prefix + 'Logits/Conv2d_1c_1x1',
                            tf_prefix + 'Predictions/Softmax')])

    layers_chk2 = [
        ('MobilenetV3/Conv', 'MobilenetV3/Conv/Conv2D'),
        ('MobilenetV3/Conv/BatchNorm', 'MobilenetV3/Conv/BatchNorm/FusedBatchNorm'),
        ('MobilenetV3/Conv/Activation', 'MobilenetV3/Conv/hard_swish/mul_1'),
        ('MobilenetV3/expanded_conv/depthwise', 'MobilenetV3/expanded_conv/depthwise/depthwise'),
        ('MobilenetV3/expanded_conv/project', 'MobilenetV3/expanded_conv/project/Conv2D'),
        ('MobilenetV3/Conv/Activation', 'MobilenetV3/Conv/hard_swish/mul_1'),
        ('MobilenetV3/expanded_conv/depthwise', 'MobilenetV3/expanded_conv/depthwise/depthwise'),
        ('MobilenetV3/expanded_conv/depthwise/Activation', 'MobilenetV3/expanded_conv/depthwise/Relu'),
        ('MobilenetV3/expanded_conv/project', 'MobilenetV3/expanded_conv/project/Conv2D'),
        ('MobilenetV3/expanded_conv_1/expand', 'MobilenetV3/expanded_conv_1/expand/Conv2D'),
        ('MobilenetV3/expanded_conv_1/depthwise', 'MobilenetV3/expanded_conv_1/depthwise/depthwise'),
        ('MobilenetV3/expanded_conv_1/project', 'MobilenetV3/expanded_conv_1/project/Conv2D'),
        ('MobilenetV3/block_2_merge', 'MobilenetV3/expanded_conv_2/output'),
        ('MobilenetV3/block_4_start', 'MobilenetV3/expanded_conv_3/output'),
        ('MobilenetV3/block_4_merge', 'MobilenetV3/expanded_conv_4/output'),
        ('MobilenetV3/Conv_1/AvgPool', 'MobilenetV3/AvgPool2D/AvgPool'),
        ('MobilenetV3/Logits/Conv2d_1c_1x1', 'MobilenetV3/Logits/output'),
        ('MobilenetV3/Logits/Conv2d_1c_1x1', 'MobilenetV3/Predictions/Softmax')
    ]

    return all_layers


def get_vgg16_activations():
    all_layers = []
    print('Warning : If comparing to VGG layers, remember to turn off batchnorm in the gtc model for precise matching')

    block_reps = [2, 2, 3, 3, 3]

    for block_idx, nreps_i in enumerate(block_reps):
        block_id = block_idx + 1

        for rep_id in range(1, nreps_i + 1):
            gtc_name = 'block%d_relu%d' % (block_id, rep_id)
            keras_name = 'block%d_conv%d' % (block_id, rep_id)

            all_layers.append((rep(gtc_name), keras_name))

    all_layers.extend([(rep('fc1_relu'), 'fc1'),
                       (rep('fc2_relu'), 'fc2'),
                       (rep('predictions'), 'predictions')])

    layers_chk_old = [
        ('block1_relu1/block1_relu1', 'block1_conv1'),
        ('block1_relu2/block1_relu2', 'block1_conv2'),
        ('block2_relu2/block2_relu2', 'block2_conv2'),
        ('block3_relu2/block3_relu2', 'block3_conv2'),
        ('block4_relu2/block4_relu2', 'block4_conv2'),
        ('block5_relu2/block5_relu2', 'block5_conv2'),
        ('fc1_relu/fc1_relu', 'fc1'),
        ('fc2_relu/fc2_relu', 'fc2'),
        ('predictions/predictions', 'predictions')
    ]

    return all_layers



def get_resnet_activations_ic(num_layers):
    all_layers = []
    layers_initial = [
        ('bn_data', 'bn_data'),
        ('conv0', 'conv0'),
        ('bn0', 'bn0'),
        ('pooling0', 'pooling0'),
    ]
    all_layers.extend(layers_initial)

    stage_reps = {
        18:[2, 2, 2, 2],
        34:[3, 4, 6, 3],
        50:[3, 4, 6, 3],
        101:[3, 4, 23, 3],
        152:[3, 8, 36, 3]}.get(num_layers)

    num_convs_per_unit = 2 if num_layers in [18, 34] else 3

    #num_sub_blocks = [3, 4, 6, 3]
    for stage_i, num_reps in enumerate(stage_reps):

        for unit_i in range(num_reps):
            for conv_i in range(num_convs_per_unit):
                bn_name = 'stage%d_unit%d_bn%d' % (stage_i+1, unit_i+1, conv_i+1)
                relu_name = 'stage%d_unit%d_relu%d' % (stage_i + 1, unit_i + 1, conv_i+1)
                conv_name = 'stage%d_unit%d_conv%d' % (stage_i + 1, unit_i + 1, conv_i+1)

                all_layers.extend([(bn_name, bn_name),
                                   (relu_name, relu_name),
                                   (conv_name, conv_name)])


    all_layers.extend([('bn1', 'bn1'),
                       ('relu1', 'relu1'),
                       ('avg_pool', 'pool1'),
                       ('Logits', 'fc1')])


    return all_layers



def get_resnet_activations_kapps(num_layers):
    layers_chk_use = [('zero_padding2d_1', 'conv1_pad'),
     ('conv0', 'conv1'),
     ('bn0', 'bn_conv1'),
     ('stage1_unit1_sc', 'res2a_branch1'),
     ('stage1_unit1_sc_bn', 'bn2a_branch1'),
     ('stage1_unit1_conv1', 'res2a_branch2a'),
     ('stage1_unit1_bn1', 'bn2a_branch2a')]

    all_layers = []
    layers_initial = [
        ('zero_padding2d_1', 'conv1_pad'),
        ('conv0', 'conv1'),
        ('bn0', 'bn_conv1'),
    ]
    all_layers.extend(layers_initial)
    if num_layers == 50:
        block_ids = [2, 3, 4, 5]
        num_sub_blocks = [3, 4, 6, 3]
    else:
        raise ValueError('unsupported number of layers')

    for block_idx, block_id in enumerate(block_ids):

        num_sub_blocks_i = num_sub_blocks[block_idx]
        sub_blocks = [chr(ord('a') + i) for i in range(num_sub_blocks_i)]

        for sub_block_idx, sub_block in enumerate(sub_blocks):
            branch_ids = [1, 2]
            for branch_id in branch_ids:
                if branch_id == 1 and sub_block == 'a':
                    branch_str = '%d%s_branch%d' % (
                        block_id, sub_block, branch_id)
                    conv_str = 'res' + branch_str
                    bn_str = 'bn' + branch_str
                    # all_layers.extend([ (rep(conv_str) + '_start', conv_str)])

                    gtc_name = 'stage%d_unit%d' % (block_idx + 1, sub_block_idx + 1)
                    gtc_conv_str = gtc_name + '_sc'
                    gtc_bn_str = gtc_name + '_sc_bn'

                    all_layers.extend([(gtc_conv_str, conv_str),
                                       (gtc_bn_str, bn_str)])

                elif branch_id == 2:
                    sections = ['a', 'b', 'c']

                    for section_idx, section in enumerate(sections):
                        branch_str = '%d%s_branch%d%s' % (
                            block_id, sub_block, branch_id, section)
                        conv_str = 'res' + branch_str
                        bn_str = 'bn' + branch_str

                        gtc_name = 'stage%d_unit%d' % (block_idx+1, sub_block_idx+1)
                        gtc_conv_str = gtc_name + '_conv%d' % (section_idx+1)
                        gtc_bn_str = gtc_name + '_bn%d' % (section_idx+1)

                        all_layers.extend([(gtc_conv_str, conv_str),
                                           (gtc_bn_str, bn_str)])

    all_layers.extend([('avg_pool', 'avg_pool'),
                       ('Logits', 'fc1000')])

    return all_layers


def get_ssd_mobilenet_v1_keras_activations(final_layers_only=False):
    all_layers = [
        #('identity_layer', 'identity_layer_v1'),
        #('input_mean_normalization', 'input_mean_normalization_v1'),
        #('input_stddev_normalization', 'input_stddev_normalization_v1'),
        #('input_channel_swap', 'input_channel_swap_v1'),
        #('preprocess', 'input_channel_swap_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_0', 'conv0_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm', 'conv0/bn_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_0/Activation', 'activation_1_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/Activation', 'activation_2_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_1_pointwise/Activation', 'activation_3_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_2_depthwise', 'conv2/dw_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_3_depthwise', 'conv3/dw_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_4_depthwise', 'conv4/dw_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_5_depthwise', 'conv5/dw_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_6_depthwise', 'conv6/dw_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_7_depthwise', 'conv7/dw_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_7_depthwise/BatchNorm', 'conv7/dw/bn_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_7_depthwise/Activation', 'activation_14_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_8_depthwise', 'conv8/dw_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_9_depthwise', 'conv9/dw_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_10_depthwise', 'conv10/dw_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_11_depthwise', 'conv11/dw_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_11_pointwise', 'conv11_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_11_pointwise/Activation', 'activation_23_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5', 'conv17_2_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5/BatchNorm', 'conv17_2/bn_v1'),
        ('FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5/Activation', 'relu_conv9_2_v1'),
       ]


    all_layers2 = [('conv11_mbox_conf', 'conv11_mbox_conf_v1'),
        ('conv17_2_mbox_loc', 'conv17_2_mbox_loc_v1'),
        ('conv17_2_mbox_conf', 'conv17_2_mbox_conf_v1'),
        ('conv9_2_mbox_priorbox', 'conv9_2_mbox_priorbox_v1'),
        ('conv9_2_mbox_priorbox_reshape', 'conv9_2_mbox_priorbox_reshape_v1'),
    ]

    all_layers.extend([
        ('BoxPredictor_0/ClassPredictor_reshape', 'conv4_3_norm_mbox_conf_reshape_v1'),
        ('BoxPredictor_1/ClassPredictor_reshape', 'fc7_mbox_conf_reshape_v1'),
        ('BoxPredictor_2/ClassPredictor_reshape', 'conv6_2_mbox_conf_reshape_v1'),
        ('BoxPredictor_3/ClassPredictor_reshape', 'conv7_2_mbox_conf_reshape_v1'),
        ('BoxPredictor_4/ClassPredictor_reshape', 'conv8_2_mbox_conf_reshape_v1'),
        ('BoxPredictor_5/ClassPredictor_reshape', 'conv9_2_mbox_conf_reshape_v1'),
        ('mbox_class', 'mbox_conf_v1')
    ])

    all_layers.extend([
        ('BoxPredictor_0/BoxEncodingPredictor_reshape', 'conv4_3_norm_mbox_loc_reshape_v1'),
        ('BoxPredictor_1/BoxEncodingPredictor_reshape', 'fc7_mbox_loc_reshape_v1'),
        ('BoxPredictor_2/BoxEncodingPredictor_reshape', 'conv6_2_mbox_loc_reshape_v1'),
        ('BoxPredictor_3/BoxEncodingPredictor_reshape', 'conv7_2_mbox_loc_reshape_v1'),
        ('BoxPredictor_4/BoxEncodingPredictor_reshape', 'conv8_2_mbox_loc_reshape_v1'),
        ('BoxPredictor_5/BoxEncodingPredictor_reshape', 'conv9_2_mbox_loc_reshape_v1'),
        ('mbox_encoding', 'mbox_loc_v1')
    ])

    all_layers.extend([
        ('BoxPredictor_0/BoxAnchors_reshape', 'conv4_3_norm_mbox_priorbox_reshape_v1'),
        ('BoxPredictor_1/BoxAnchors_reshape', 'fc7_mbox_priorbox_reshape_v1'),
        ('BoxPredictor_2/BoxAnchors_reshape', 'conv6_2_mbox_priorbox_reshape_v1'),
        ('BoxPredictor_3/BoxAnchors_reshape', 'conv7_2_mbox_priorbox_reshape_v1'),
        ('BoxPredictor_4/BoxAnchors_reshape', 'conv8_2_mbox_priorbox_reshape_v1'),
        ('BoxPredictor_5/BoxAnchors_reshape', 'conv9_2_mbox_priorbox_reshape_v1'),
        ('mbox_anchors', 'mbox_priorbox_v1'),
        ('mbox_class_softmax', 'mbox_conf_softmax_v1'),
        ('predictions', 'predictions_v1')
    ])

    if final_layers_only:
        all_layers = [
            ('mbox_encoding', 'mbox_loc_v1'),
            ('mbox_class', 'mbox_conf_v1'),
            # ('mbox_class_softmax', 'mbox_conf_softmax_v1'),
            ('mbox_anchors', 'mbox_priorbox_v1'),
            ('mbox_class_softmax', 'mbox_conf_softmax_v1'),
            ('predictions', 'predictions_v1')
        ]

    return all_layers


def get_ssd_vgg_activations(final_layers_only=False):

    #('identity_layer', 'identity_layer_v1'),
    #('input_mean_normalization', 'input_mean_normalization_v1'),
    #('input_stddev_normalization', 'input_stddev_normalization_v1'),
    #('input_channel_swap', 'input_channel_swap_v1'),
    #('preprocess', 'input_channel_swap_v1'),

    reps = [2, 2, 3, 3, 3,  2, 2, 2, 2]
    gtc_prefix = 'SSD_GTC/'
    gtc_feat_prefix = gtc_prefix + 'FeatureExtractor/VGG16/'
    all_layers = [(gtc_prefix + 'identity_layer', 'identity_layer'),
                  (gtc_prefix + 'Preprocess_subtract_swap_channels', 'input_channel_swap'),]


    for block_i, nreps_i in enumerate(reps):
        for rep_i in range(nreps_i):
            all_layers.append(
                ('%sblock%d_relu%d' % (gtc_feat_prefix, block_i+1, rep_i+1),
                 'conv%d_%d' % (block_i+1, rep_i+1) ) )

        if block_i+1 == 5:

            all_layers.extend([ (gtc_feat_prefix + 'block4_conv3_norm', 'conv4_3_norm'),
                                (gtc_feat_prefix + 'fc6_relu', 'fc6'),
                                (gtc_feat_prefix + 'fc7_relu', 'fc7')])
            #all_layers.extend([(gtc_feat_prefix + 'fc6', 'fc6'),
            #                   (gtc_feat_prefix + 'fc6_relu', 'fc6_relu'),
            #                   (gtc_feat_prefix + 'fc7', 'fc7'),
            #                   (gtc_feat_prefix + 'fc7_relu', 'fc7_relu')])

        if block_i+1 <= 5:
            all_layers.append( ( 'FeatureExtractor/VGG16/maxpool_%d' % (block_i+1),  'pool%d' % (block_i+1) )  )


    all_layers2 = [('conv11_mbox_conf', 'conv11_mbox_conf_v1'),
        ('conv17_2_mbox_loc', 'conv17_2_mbox_loc_v1'),
        ('conv17_2_mbox_conf', 'conv17_2_mbox_conf_v1'),
        ('conv9_2_mbox_priorbox', 'conv9_2_mbox_priorbox_v1'),
        ('conv9_2_mbox_priorbox_reshape', 'conv9_2_mbox_priorbox_reshape_v1'),
    ]

    #v1_str = '_v1'
    v1_str = ''

    all_layers.extend([
        (gtc_prefix + 'BoxPredictor_0/ClassPredictor', 'conv4_3_norm_mbox_conf' + v1_str),
        (gtc_prefix + 'BoxPredictor_1/ClassPredictor', 'fc7_mbox_conf' + v1_str),
        (gtc_prefix + 'BoxPredictor_2/ClassPredictor', 'conv6_2_mbox_conf' + v1_str),
        (gtc_prefix + 'BoxPredictor_3/ClassPredictor', 'conv7_2_mbox_conf' + v1_str),
        (gtc_prefix + 'BoxPredictor_4/ClassPredictor', 'conv8_2_mbox_conf' + v1_str),
        (gtc_prefix + 'BoxPredictor_5/ClassPredictor', 'conv9_2_mbox_conf' + v1_str),
    ])
    all_layers.extend([
        (gtc_prefix + 'BoxPredictor_0/BoxEncodingPredictor', 'conv4_3_norm_mbox_loc' + v1_str),
        (gtc_prefix + 'BoxPredictor_1/BoxEncodingPredictor', 'fc7_mbox_loc' + v1_str),
        (gtc_prefix + 'BoxPredictor_2/BoxEncodingPredictor', 'conv6_2_mbox_loc' + v1_str),
        (gtc_prefix + 'BoxPredictor_3/BoxEncodingPredictor', 'conv7_2_mbox_loc' + v1_str),
        (gtc_prefix + 'BoxPredictor_4/BoxEncodingPredictor', 'conv8_2_mbox_loc' + v1_str),
        (gtc_prefix + 'BoxPredictor_5/BoxEncodingPredictor', 'conv9_2_mbox_loc' + v1_str),
        (gtc_prefix + 'mbox_encoding', 'mbox_loc' + v1_str)
    ])

    do_reshape = False
    if do_reshape:
        all_layers.extend([
            (gtc_prefix + 'BoxPredictor_0/ClassPredictor_reshape', 'conv4_3_norm_mbox_conf_reshape' + v1_str),
            (gtc_prefix + 'BoxPredictor_1/ClassPredictor_reshape', 'fc7_mbox_conf_reshape' + v1_str),
            (gtc_prefix + 'BoxPredictor_2/ClassPredictor_reshape', 'conv6_2_mbox_conf_reshape' + v1_str),
            (gtc_prefix + 'BoxPredictor_3/ClassPredictor_reshape', 'conv7_2_mbox_conf_reshape' + v1_str),
            (gtc_prefix + 'BoxPredictor_4/ClassPredictor_reshape', 'conv8_2_mbox_conf_reshape' + v1_str),
            (gtc_prefix + 'BoxPredictor_5/ClassPredictor_reshape', 'conv9_2_mbox_conf_reshape' + v1_str),
            (gtc_prefix + 'mbox_class', 'mbox_conf' + v1_str)
        ])

        all_layers.extend([
            (gtc_prefix + 'BoxPredictor_0/BoxEncodingPredictor_reshape', 'conv4_3_norm_mbox_loc_reshape' + v1_str),
            (gtc_prefix + 'BoxPredictor_1/BoxEncodingPredictor_reshape', 'fc7_mbox_loc_reshape' + v1_str),
            (gtc_prefix + 'BoxPredictor_2/BoxEncodingPredictor_reshape', 'conv6_2_mbox_loc_reshape' + v1_str),
            (gtc_prefix + 'BoxPredictor_3/BoxEncodingPredictor_reshape', 'conv7_2_mbox_loc_reshape' + v1_str),
            (gtc_prefix + 'BoxPredictor_4/BoxEncodingPredictor_reshape', 'conv8_2_mbox_loc_reshape' + v1_str),
            (gtc_prefix + 'BoxPredictor_5/BoxEncodingPredictor_reshape', 'conv9_2_mbox_loc_reshape' + v1_str),
            (gtc_prefix + 'mbox_encoding', 'mbox_loc' + v1_str)
        ])

    all_layers.extend([
        (gtc_prefix + 'BoxPredictor_0/BoxAnchors_reshape', 'conv4_3_norm_mbox_priorbox_reshape' + v1_str),
        (gtc_prefix + 'BoxPredictor_1/BoxAnchors_reshape', 'fc7_mbox_priorbox_reshape' + v1_str),
        (gtc_prefix + 'BoxPredictor_2/BoxAnchors_reshape', 'conv6_2_mbox_priorbox_reshape' + v1_str),
        (gtc_prefix + 'BoxPredictor_3/BoxAnchors_reshape', 'conv7_2_mbox_priorbox_reshape' + v1_str),
        (gtc_prefix + 'BoxPredictor_4/BoxAnchors_reshape', 'conv8_2_mbox_priorbox_reshape' + v1_str),
        (gtc_prefix + 'BoxPredictor_5/BoxAnchors_reshape', 'conv9_2_mbox_priorbox_reshape' + v1_str),
        (gtc_prefix + 'mbox_anchors', 'mbox_priorbox' + v1_str),
        (gtc_prefix + 'mbox_class_softmax', 'mbox_conf_softmax' + v1_str),
        (gtc_prefix + 'predictions', 'predictions' + v1_str)
    ])

    if final_layers_only:
        all_layers = [
            ('mbox_encoding', 'mbox_loc' + v1_str),
            ('mbox_class', 'mbox_conf' + v1_str),
            # ('mbox_class_softmax', 'mbox_conf_softmax' + v1_str),
            ('mbox_anchors', 'mbox_priorbox' + v1_str),
            ('mbox_class_softmax', 'mbox_conf_softmax' + v1_str),
            ('predictions', 'predictions' + v1_str)
        ]

    return all_layers


def get_ssd_mobilenet_tf_activations(model_name, config=None, final_layers_only=False):
    if isinstance(model_name, net.SSDNetType):
        model_name = model_name.get_name(version_only=False)


    def get_conv_bn_act_layers(prefix_gtc, prefix_tf=None, conv=True, bn=True, act=True, conv_name='Conv2D', bn_name='BatchNorm/batchnorm/add_1', act_name='Relu6'):
        layers = []
        if prefix_tf is None:
            prefix_tf = prefix_gtc
        if conv:
            layers.append( (prefix_gtc + '', prefix_tf + '/' + conv_name) )
        if bn:
            layers.append( (prefix_gtc + '/BatchNorm', prefix_tf + '/' + bn_name) )
        if act:
            layers.append( (prefix_gtc + '/Activation', prefix_tf + '/' + act_name) )

        return layers

    all_layers = []

    if 'ssd_mobilenet_v1' in model_name:
        layer_name_prefix = 'FeatureExtractor/MobilenetV1/'
        gtc_prefix = layer_name_prefix
        tf_prefix = layer_name_prefix
        tf_prefix_rep = layer_name_prefix + 'MobilenetV1/'
        layers_initial = get_conv_bn_act_layers( gtc_prefix + 'Conv2d_0', tf_prefix_rep + 'Conv2d_0')
        all_layers.extend(layers_initial)
        n_separable_layers = 13
        for layer_id in range(1, n_separable_layers+1):
            depth_layer_str = 'Conv2d_%d_depthwise' % layer_id
            point_layer_str = 'Conv2d_%d_pointwise' % layer_id
            all_layers.extend( get_conv_bn_act_layers( gtc_prefix + depth_layer_str, tf_prefix_rep + depth_layer_str, conv_name='depthwise' ) )
            all_layers.extend( get_conv_bn_act_layers( gtc_prefix + point_layer_str, tf_prefix_rep + point_layer_str) )

        extra_feat_layers = [
         ('Conv2d_13_pointwise_1_Conv2d_2', 'Conv2d_13_pointwise_1_Conv2d_2_1x1_256'),
         ('Conv2d_13_pointwise_2_Conv2d_2', 'Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512'),
         ('Conv2d_13_pointwise_1_Conv2d_3', 'Conv2d_13_pointwise_1_Conv2d_3_1x1_128'),
         ('Conv2d_13_pointwise_2_Conv2d_3', 'Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256'),
         ('Conv2d_13_pointwise_1_Conv2d_4', 'Conv2d_13_pointwise_1_Conv2d_4_1x1_128'),
         ('Conv2d_13_pointwise_2_Conv2d_4', 'Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256'),
         ('Conv2d_13_pointwise_1_Conv2d_5', 'Conv2d_13_pointwise_1_Conv2d_5_1x1_64'),
         ('Conv2d_13_pointwise_2_Conv2d_5', 'Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128')]

        for extra_feat_layer in extra_feat_layers:
            gtc_layer, tf_layer = extra_feat_layer
            all_layers.extend(get_conv_bn_act_layers(gtc_prefix +gtc_layer, tf_prefix + tf_layer))

        n_boxes = 6
        for box_id in range(n_boxes):
            box_prefix = 'BoxPredictor_%d/' % box_id

            all_layers.extend( get_conv_bn_act_layers(
                box_prefix + 'ClassPredictor', conv_name='BiasAdd', bn=False, act=False) )
            all_layers.extend( get_conv_bn_act_layers(
                box_prefix + 'BoxEncodingPredictor', conv_name='BiasAdd', bn=False, act=False) )

            suffix_id = '' if box_id == 0 else '_%d' % box_id
            all_layers.extend( [(box_prefix + 'BoxAnchors_reshape', 'MultipleGridAnchorGenerator/concat%s' % suffix_id) ] )

            #all_layers.extend([('mbox_anchors', 'MultipleGridAnchorGenerator/Concatenate/concat')])
        tf_mbox_anchors_name = 'MultipleGridAnchorGenerator/Concatenate/concat'


    elif 'ssd_mobilenet_v2' in model_name or 'ssdlite_mobilenet_v2' in model_name:
        all_layers = []

        layer_prefix = 'FeatureExtractor/MobilenetV2/'
        layers_initial = get_conv_bn_act_layers( layer_prefix + 'Conv', layer_prefix + 'Conv')
        all_layers.extend(layers_initial)
        n_expansion_layers = 13
        for layer_id in range(0, n_expansion_layers+1):
            layer_id_str = '' if layer_id == 0 else '_%d' % layer_id
            block_name = layer_prefix + 'expanded_conv' + layer_id_str + '/'
            if layer_id > 0:
                all_layers.extend( get_conv_bn_act_layers( block_name + 'expand',  ) )
            all_layers.extend( get_conv_bn_act_layers( block_name + 'depthwise', conv_name='depthwise') )
            all_layers.extend( get_conv_bn_act_layers( block_name + 'project', act=False) )

        extra_feat_layers = [
         ('layer_19_1_Conv2d_2', 'layer_19_1_Conv2d_2_1x1_256'),
         ('layer_19_2_Conv2d_2', 'layer_19_2_Conv2d_2_3x3_s2_512'),
         ('layer_19_1_Conv2d_3', 'layer_19_1_Conv2d_3_1x1_128'),
         ('layer_19_2_Conv2d_3', 'layer_19_2_Conv2d_3_3x3_s2_256'),
         ('layer_19_1_Conv2d_4', 'layer_19_1_Conv2d_4_1x1_128'),
         ('layer_19_2_Conv2d_4', 'layer_19_2_Conv2d_4_3x3_s2_256'),
         ('layer_19_1_Conv2d_5', 'layer_19_1_Conv2d_5_1x1_64'),
         ('layer_19_2_Conv2d_5', 'layer_19_2_Conv2d_5_3x3_s2_128')]

        for extra_feat_layer in extra_feat_layers:
            gtc_layer, tf_layer = extra_feat_layer
            all_layers.extend(get_conv_bn_act_layers(layer_prefix +gtc_layer, layer_prefix + tf_layer))

        n_boxes = 6
        for box_id in range(n_boxes):
            box_prefix = 'BoxPredictor_%d/' % box_id

            if model_name == 'mobilenet_v2_lite':
                all_layers.extend(get_conv_bn_act_layers(
                    box_prefix + 'BoxEncodingPredictor_depthwise',
                    conv_name='depthwise', bn_name='BatchNorm/FusedBatchNorm'))
            all_layers.extend( get_conv_bn_act_layers(
                box_prefix + 'BoxEncodingPredictor', conv_name='BiasAdd', bn=False, act=False) )

            if model_name == 'mobilenet_v2_lite':
                all_layers.extend(get_conv_bn_act_layers(
                    box_prefix + 'ClassPredictor_depthwise',
                    conv_name='depthwise', bn_name='BatchNorm/FusedBatchNorm'))
            all_layers.extend( get_conv_bn_act_layers(
                box_prefix + 'ClassPredictor', conv_name='BiasAdd', bn=False, act=False) )


            suffix_id = '' if box_id == 0 else '_%d' % box_id
            all_layers.extend( [(box_prefix + 'BoxAnchors_reshape',
                                 'MultipleGridAnchorGenerator/concat%s' % suffix_id) ] )

        tf_mbox_anchors_name = 'Concatenate/concat'



    elif 'ssd_mobilenet_v3_large' in model_name:

        base_network_layers = get_mobilenet_v3_activations(config)
        all_layers = []
        for layer_pair in base_network_layers:
            if 'Conv_1' in layer_pair[0]:
                break
            new_pair = (layer_pair[0].replace('Mobilenet_V3_GTC/', 'MobileNet_SSD_GTC/FeatureExtractor/MobilenetV3/'),
                        'import/FeatureExtractor/' + layer_pair[1])
            all_layers.append(new_pair)

        gtc_layer_prefix = 'MobileNet_SSD_GTC/FeatureExtractor/MobilenetV3/'
        tf_layer_prefix = 'import/FeatureExtractor/MobilenetV3/'
        bn_name = '/BatchNorm/FusedBatchNorm'
        hwish_name = 'hard_swish/mul_1'

        extra_feat_layers = [
            ('layer_17_1_Conv2d_2', 'layer_17_1_Conv2d_2_1x1_256'),
            ('layer_17_2_Conv2d_2', 'layer_17_2_Conv2d_2_3x3_s2_512'),
            ('layer_17_1_Conv2d_3', 'layer_17_1_Conv2d_3_1x1_128'),
            ('layer_17_2_Conv2d_3', 'layer_17_2_Conv2d_3_3x3_s2_256'),
            ('layer_17_1_Conv2d_4', 'layer_17_1_Conv2d_4_1x1_128'),
            ('layer_17_2_Conv2d_4', 'layer_17_2_Conv2d_4_3x3_s2_256'),
            ('layer_17_1_Conv2d_5', 'layer_17_1_Conv2d_5_1x1_64'),
            ('layer_17_2_Conv2d_5', 'layer_17_2_Conv2d_5_3x3_s2_128')]

        for extra_feat_layer in extra_feat_layers:
            gtc_layer, tf_layer = extra_feat_layer
            all_layers.extend(get_conv_bn_act_layers(gtc_layer_prefix + gtc_layer,
                                                     tf_layer_prefix + tf_layer))

        n_boxes = 6
        for box_id in range(n_boxes):
            box_prefix = 'BoxPredictor_%d/' % box_id

            gtc_prefix = ''
            tf_prefix = 'import/'

            all_layers.extend(get_conv_bn_act_layers(
                gtc_prefix + box_prefix + 'BoxEncodingPredictor_depthwise',
                tf_prefix + box_prefix + 'BoxEncodingPredictor_depthwise',
                conv_name='depthwise', bn_name='BatchNorm/FusedBatchNorm'))
            all_layers.extend(get_conv_bn_act_layers(
                gtc_prefix + box_prefix + 'BoxEncodingPredictor',
                tf_prefix + box_prefix + 'BoxEncodingPredictor',
                conv_name='BiasAdd', bn=False, act=False))

            all_layers.extend(get_conv_bn_act_layers(
                gtc_prefix + box_prefix + 'ClassPredictor_depthwise',
                tf_prefix + box_prefix + 'ClassPredictor_depthwise',
                conv_name='depthwise', bn_name='BatchNorm/FusedBatchNorm'))
            all_layers.extend(get_conv_bn_act_layers(
                gtc_prefix + box_prefix + 'ClassPredictor',
                tf_prefix + box_prefix + 'ClassPredictor',
                conv_name='BiasAdd', bn=False, act=False))

            suffix_id = '' if box_id == 0 else '_%d' % box_id
            all_layers.extend([(gtc_prefix + box_prefix + 'BoxAnchors_reshape',
                                'MultipleGridAnchorGenerator/concat%s' % suffix_id)])

        tf_mbox_anchors_name = 'anchors'



    elif  'ssd_mobilenet_v3_small' in model_name:
        all_layers = []
        tf_mbox_anchors_name = ''
        pass

    else:
        raise ValueError('Unknown model')

    if final_layers_only:
        all_layers = []

    all_layers.extend([
        ('mbox_class', 'concat_1'),
        ('mbox_encoding', 'concat'),
        ('mbox_anchors', tf_mbox_anchors_name),
    ])


    return all_layers


def get_activation_layer_names(net_type, config=None, final_layers_only=False):
    model_name = str(net_type)

    if isinstance(net_type, net.SSDNetType):
        base_net = net_type.base_net_type
        if isinstance(base_net, net.MobilenetType):
            if net_type.style_abbrev == 'keras':
                return get_ssd_mobilenet_v1_keras_activations(final_layers_only)

            elif net_type.style_abbrev == 'tf':
                return get_ssd_mobilenet_tf_activations(net_type, config, final_layers_only)

        else:
            assert base_net.name.lower() == 'vgg'
            return get_ssd_vgg_activations()


    #elif isinstance(net_type, net) model_name == 'ssd_mobilenet_v2-t':
    #    return get_ssd_mobilenet_v1_keras_activations(final_layers_only)


    elif 'mobilenet_v1' in model_name:
        return get_mobilenet_v1_activations(config)

    elif 'mobilenet_v2' in model_name:
        return get_mobilenet_v2_activations(config)

    elif 'mobilenet_v3' in model_name:
        return get_mobilenet_v3_activations(config)

    elif model_name == 'vgg16':
        return get_vgg16_activations()

    elif 'resnet' in model_name:
        num_layers = int(re.search('resnet(\d+)', model_name).groups()[0])
        if net_type.style == 'image-classifiers':
            return get_resnet_activations_ic(num_layers)

        elif net_type.style == 'keras.applications':
            return get_resnet_activations_kapps(num_layers)


    else:
        raise ValueError('Unknown model')

def get_mobilenet_v1_ssd_keras_namer():

    def name_converter(name):
        name_dict = {
            '''
        'input_1_v1':
         'conv1_padding_v1',
         'conv0_v1',
         'conv0/bn_v1',
         'activation_1_v1',
         'conv1/dw_v1',
         'conv1/dw/bn_v1',
         'activation_2_v1',
         'conv1_v1',
         'conv1/bn_v1',
         'activation_3_v1',
         'conv2_padding_v1',
         'conv2/dw_v1',
         'conv2/dw/bn_v1',
         'activation_4_v1',
         'conv2_v1',
         'conv2/bn_v1',
         'activation_5_v1',
         'conv3/dw_v1',
         'conv3/dw/bn_v1',
         'activation_6_v1',
         'conv3_v1',
         'conv3/bn_v1',
         'activation_7_v1',
         'conv3_padding_v1',
         'conv4/dw_v1',
         'conv4/dw/bn_v1',
         'activation_8_v1',
         'conv4_v1',
         'conv4/bn_v1',
         'activation_9_v1',
         'conv5/dw_v1',
         'conv5/dw/bn_v1',
         'activation_10_v1',
         'conv5_v1',
         'conv5/bn_v1',
         'activation_11_v1',
         'conv4_padding_v1',
         'conv6/dw_v1',
         'conv6/dw/bn_v1',
         'activation_12_v1',
         'conv6_v1',
         'conv6/bn_v1',
         'activation_13_v1',
         'conv7/dw_v1',
         'conv7/dw/bn_v1',
         'activation_14_v1',
         'conv7_v1',
         'conv7/bn_v1',
         'activation_15_v1',
         'conv8/dw_v1',
         'conv8/dw/bn_v1',
         'activation_16_v1',
         'conv8_v1',
         'conv8/bn_v1',
         'activation_17_v1',
         'conv9/dw_v1',
         'conv9/dw/bn_v1',
         'activation_18_v1',
         'conv9_v1',
         'conv9/bn_v1',
         'activation_19_v1',
         'conv10/dw_v1',
         'conv10/dw/bn_v1',
         'activation_20_v1',
         'conv10_v1',
         'conv10/bn_v1',
         'activation_21_v1',
         'conv11/dw_v1',
         'conv11/dw/bn_v1',
         'activation_22_v1',
         'conv11_v1',
         'conv11/bn_v1',
         'activation_23_v1',
         'conv5_padding_v1',
         'conv12/dw_v1',
         'conv12/dw/bn_v1',
         'activation_24_v1',
         'conv12_v1',
         'conv12/bn_v1',
         'activation_25_v1',
         'conv13/dw_v1',
         'conv13/dw/bn_v1',
         'activation_26_v1',
         'conv13_v1',
         'conv13/bn_v1',
         'activation_27_v1',
         'conv14_1_v1',
         'conv14_1/bn_v1',
         'relu_conv6_1_v1',
         'conv6_padding_v1',
         'conv14_2_v1',
         'conv14_2/bn_v1',
         'relu_conv6_2_v1',
         'conv15_1_v1',
         'conv15_1/bn_v1',
         'relu_conv7_1_v1',
         'conv7_padding_v1',
         'conv15_2_v1',
         'conv15_2/bn_v1',
         'relu_conv7_2_v1',
         'conv16_1_v1',
         'conv16_1/bn_v1',
         'relu_conv8_1_v1',
         'conv8_padding_v1',
         'conv16_2_v1',
         'conv16_2/bn_v1',
         'relu_conv8_2_v1',
         'conv17_1_v1',
         'conv17_1/bn_v1',
         'relu_conv9_1_v1',
         'conv9_padding_v1',
         'conv17_2_v1',
         'conv17_2/bn_v1',
         'relu_conv9_2_v1',
         'conv11_mbox_conf_v1',
         'conv13_mbox_conf_v1',
         'conv14_2_mbox_conf_v1',
         'conv15_2_mbox_conf_v1',
         'conv16_2_mbox_conf_v1',
         'conv17_2_mbox_conf_v1',
         'conv11_mbox_loc_v1',
         'conv13_mbox_loc_v1',
         'conv14_2_mbox_loc_v1',
         'conv15_2_mbox_loc_v1',
         'conv16_2_mbox_loc_v1',
         'conv17_2_mbox_loc_v1',
         'conv4_3_norm_mbox_conf_reshape_v1',
         'fc7_mbox_conf_reshape_v1',
         'conv6_2_mbox_conf_reshape_v1',
         'conv7_2_mbox_conf_reshape_v1',
         'conv8_2_mbox_conf_reshape_v1',
         'conv9_2_mbox_conf_reshape_v1',
         'conv4_3_norm_mbox_priorbox_v1',
         'fc7_mbox_priorbox_v1',
         'conv6_2_mbox_priorbox_v1',
         'conv7_2_mbox_priorbox_v1',
         'conv8_2_mbox_priorbox_v1',
         'conv9_2_mbox_priorbox_v1',
         'mbox_conf_v1',
         'conv4_3_norm_mbox_loc_reshape_v1',
         'fc7_mbox_loc_reshape_v1',
         'conv6_2_mbox_loc_reshape_v1',
         'conv7_2_mbox_loc_reshape_v1',
         'conv8_2_mbox_loc_reshape_v1',
         'conv9_2_mbox_loc_reshape_v1',
         'conv4_3_norm_mbox_priorbox_reshape_v1',
         'fc7_mbox_priorbox_reshape_v1',
         'conv6_2_mbox_priorbox_reshape_v1',
         'conv7_2_mbox_priorbox_reshape_v1',
         'conv8_2_mbox_priorbox_reshape_v1',
         'conv9_2_mbox_priorbox_reshape_v1',
         'mbox_conf_softmax_v1',
         'mbox_loc_v1',
         'mbox_priorbox_v1',
         'predictions_v1',
         'decoded_predictions_v1'
         '''
        }
        output_name = name_dict.get(name, name)
        return output_name

    return name_converter



if __name__ == '__main__':
    layers_chk = get_activation_layer_names('resnet18')
    #layers_chk = get_resnet50_activations()
    print(layers_chk)

