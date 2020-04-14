import quantization as quant
import gtc
import tensorflow.compat.v1 as tf

from quantization import quant_utils as quant_utl
from networks import net_utils as net

def ResNet(config, x_placeholder, num_layers=18, all_quantizers=None):  # published ResNet version for imagenet

    network_name = net.get_network_full_name_with_tag(
        config, default='Resnet%d_GTC' % num_layers)

    if all_quantizers is None:
        # TODO: calculate exactly how many quantizers are needed
        all_quantizers = quant_utl.QuantizerSet(
            num_layers= num_layers + 5, config=config, prefix=network_name) # extra 3 to handle any shortcut- layer-increase convolutions


    if num_layers == 18:    # ResNet18
        reps = [2, 2, 2, 2]
        do_bottleneck = False
    elif num_layers == 34 : # ResNet34
        reps = [3, 4, 6, 3]
        do_bottleneck = False
    elif num_layers == 50:  # ResNet50
        reps = [3, 4, 6, 3]
        do_bottleneck = True
    elif num_layers == 101: # ResNet101
        reps = [3, 4, 23, 3]
        do_bottleneck = True
    elif num_layers == 152: # ResNet152
        reps = [3, 8, 36, 3]
        do_bottleneck = True
    else:
        raise ValueError('Unsupported number of ResNet layers')

    filters_init = 64
    filters = [64, 128, 256, 512]

    namer = LayerNamer(prefix=network_name)


    if do_bottleneck:
        filters_expand = [f * 4 for f in filters]
        filters_in = [filters_init] + filters_expand[:-1]
    else:
        filters_expand = [None] * len(filters)
        filters_in = [filters_init] + filters[:-1]


    resnet_config = net.ResnetDefaults(config.net_style)
    net_type = net.NetworkType(name='resnet', num_layers=num_layers,
                               style=config.net_style)

    if config.net_style == 'keras.applications':
        defaults_abbrev = '_keras'
    elif config.net_style == 'image-classifiers':
        defaults_abbrev = '_ic'
    else:
        raise ValueError('Unrecognized model style: %s' % config.net_style)

    conv_kwargs = resnet_config.conv_kwargs
    bn_kwargs = resnet_config.bn_kwargs

    # Create Model
    model = gtc.GTCModel(name=network_name)
    model.type = net_type
    model.add(x_placeholder, name='input')

    if config.resize_inputs:
        model.add(gtc.Resize(new_shape=(config.image_size[0], config.image_size[1]),
                             name=network_name + 'Resize'))

    # 0a. Preprocessing of inputs
    if resnet_config.do_preprocessing:
        defaults_name = 'resnet_' + net_type.style_abbrev
        model.add( gtc.Preprocess(defaults=defaults_name,
                                  name=network_name + 'Preprocess'))

    # 0a. Alternatively: apply batch-normalization layer to inputs
    if resnet_config.do_input_batchnorm:
        model.add(gtc.BatchNormalization(scale=False,
                                         name=network_name + 'bn_data'))

    # 0b. Quantizing inputs
    if config.quantize_input:
        model.add_linear_quantizer(config.num_bits_linear, quant_name='LQ_in',
                                    layer_name='input_quantized')


    # 1. Inital convolutional layers
    model.add( gtc.ZeroPadding2D(padding=(3, 3), name=namer.get('initial_pad')) )
    conv_kwargs_init = conv_kwargs.copy()
    conv_kwargs_init['explicit_zero_prepadding'] = False
    net.add_conv_bn_act_layers(model, filters=filters_init,kernel_size=7,stride=2,
                               padding='VALID', activation='relu',
                               name_dict=namer.get('initial_conv'),
                               config=config, quant_dict=all_quantizers.next(),
                               **conv_kwargs_init, **bn_kwargs)

    model.add( gtc.ZeroPadding2D(padding=(1, 1), name= network_name + 'pool1_pad' ))
    model.add( gtc.MaxPooling2D(strides=2, pool_size=3, padding='valid',
                                name=namer.get('pooling')) )

    # 2. Residual Blocks 2, 3, 4, 5
    for block_id, nreps in enumerate(reps):
        stride = 1 if block_id == 0 else 2
        add_expansion_shortcut_blocks(
            model, filters_in[block_id], filters[block_id],
            filters_expand[block_id],
            stride=stride, resnet_config=resnet_config,
            stage=block_id, nreps=nreps,
            conv_before_bn=resnet_config.do_conv_before_batchnorm,
            namer=namer, config=config, quantizers=all_quantizers,
            **conv_kwargs, **bn_kwargs)

    quant_final = all_quantizers.next()
    if resnet_config.do_batchnorm_after_last_block:
        model.add(gtc.BatchNormalization(name=network_name + 'bn1', **bn_kwargs,
                                         quantizer=quant_final['batchnorm']))
        model.add(gtc.Activation('relu', name=network_name + 'relu1',
                                 quantizer=quant_final['activation']))

    # 3. Final layers (pooling, logits)
    model.add(gtc.GlobalAveragePooling2D(name=namer.get('final_pool'),
                                         quantizer=all_quantizers.next('activation')))
    model.add(gtc.Flatten(name=network_name+'flatten_1'))

    if config.quantize_features:
        model.add_linear_quantizer(config.num_bits_linear, quant_name='LQ_feat',
                                   layer_name='features_quantized')

    model.add(gtc.Dense(units=config.num_classes, use_bias=True,
                        quantizer=quant_final['conv'],
                        name=namer.get('logits')) )

    if config.quantize_logits_output:
        model.add_linear_quantizer(config.num_bits_linear, quant_name='LQ_logits',
                                   layer_name='logits_quantized')


    all_quantizers.confirm_done(print_only=False)

    return model





def add_expansion_shortcut_blocks(
        model, filters_in=None, filters_calc=None, filters_expand=None,
        kernel_size=3, stride=1, activation='relu', resnet_config=None,
        conv_before_bn=True,
        stage=None, nreps=1, namer=None, config=None,  quantizers=None, **kwargs):

    grp_str = ''
    if stage is not None:
        grp_str = '_%d' % stage
        conv_name = 'conv' + grp_str
        bn_name = 'bn' + grp_str
        act_name = 'act' + grp_str

    filters_in_cur_layer = filters_in
    shortcut_start_layer_name, shortcut_end_layer_name = None, None

    for rep_i in range(nreps):
        #rep_str = '' if rep == 1 else ('_%d' % (rep_i + 1))
        stride_use = stride if rep_i == 0 else 1

        do_shortcut = True #stride_use == 1
        if do_shortcut:
            if resnet_config.shortcut_after_first_activation and rep_i == 0:
                # Start the shortcut after the first batchnorm+relu (further down)
                shortcut_start_layer_name = namer.get_block(
                    stage, rep_i, conv_id=1)['activation']
            else:
                # The last layer added to the network
                shortcut_start_layer_name = model.last_layer_name()

            # namer.get_block(stage, rep_i, shortcut=True)['conv'] + '_start'
            # add identity so that can refer to this part later for bottleneck
            #model.add(gtc.Activation(activation='linear', name=layer_start_name,
            #                         quantizer= None))
            #shortcut_layer_name = layer_start_name

        final_activation = activation if resnet_config.final_activation_before_add \
            else None

        if filters_expand is None:
            # 'building block' for smaller networks, e.g. ResNet18 and ResNet34
            # Two identical standard convolutional layers
            net.add_conv_bn_act_layers(
                model, filters_calc, kernel_size=kernel_size, stride=stride_use,
                conv_before_bn=conv_before_bn,
                activation=activation, quant_dict=quantizers.next(),
                name_dict=namer.get_block(stage, rep_i, conv_id=1),
                config=config, **kwargs)

            net.add_conv_bn_act_layers(
                model, filters_calc, kernel_size=kernel_size, stride=1,
                conv_before_bn=conv_before_bn,
                activation=final_activation, quant_dict=quantizers.next(),
                name_dict=namer.get_block(stage, rep_i, conv_id=2),
                config=config, **kwargs)

            filters_out = filters_calc
        else:
            # '"bottleneck" building block' for larger networks, e.g. ResNet50 and ResNet101, ResNet152
            if resnet_config.stride2_in_first_expansion_block:
                stride_conv1, stride_conv2, stride_conv3 = stride_use, 1, 1
            else:
                stride_conv1, stride_conv2, stride_conv3 = 1, stride_use, 1

            # Three layers: contract(1x1), filter(3x3), expand(1x1)
            net.add_conv_bn_act_layers(
                model, filters_calc, kernel_size=1, stride=stride_conv1,
                conv_before_bn=conv_before_bn,
                activation=activation, quant_dict=quantizers.next(),
                name_dict=namer.get_block(stage, rep_i, conv_id=1),
                config=config, **kwargs)

            net.add_conv_bn_act_layers(
                model, filters_calc, kernel_size=kernel_size, stride=stride_conv2,
                conv_before_bn=conv_before_bn,
                activation=activation, quant_dict=quantizers.next(),
                name_dict=namer.get_block(stage, rep_i, conv_id=2),
                config=config, **kwargs)

            net.add_conv_bn_act_layers(
                model, filters_expand, kernel_size=1, stride=stride_conv3,
                conv_before_bn=conv_before_bn,
                activation=final_activation, quant_dict=quantizers.next(),
                name_dict=namer.get_block(stage, rep_i, conv_id=3),
                config=config, **kwargs)

            filters_out = filters_expand



        if do_shortcut:
            shortcut_end_layer_name = model.last_layer_name()

            force_conv = (stage == 0 and rep_i == 0) and \
                             resnet_config.force_conv_before_first_shortcut

            if filters_in_cur_layer != filters_out or force_conv : # Shortcut connection must be between tensors of the same size. if not the same size, use a 1x1 convolutional layer to increase the size
                kwargs_shortcut = kwargs.copy()
                kwargs_shortcut['explicit_zero_prepadding'] = False
                do_batchnorm = resnet_config.do_batchnorm_after_shortcuts
                net.add_conv_bn_act_layers(
                    model, filters_out, kernel_size=1, stride=stride_use,
                    do_batchnorm=do_batchnorm, activation=None, name_dict=namer.get_block(stage, rep_i, shortcut=True),
                    layer_inputs=shortcut_start_layer_name, quant_dict=quantizers.next(), config=config, **kwargs_shortcut)

                shortcut_start_layer_name = model.last_layer_name()
            # residual shortcut connection : add original + final layer, then apply ReLU
            layer_merge_name = shortcut_end_layer_name + '_add'

            model.add(gtc.Add([shortcut_start_layer_name, shortcut_end_layer_name],
                              name=layer_merge_name))

            if not resnet_config.final_activation_before_add:
                # If final activation was not done above (before the Add), do it now, after the Add.
                model.add(gtc.Activation(activation='relu', quantizer=quant.IdentityQuantization(),
                                         name=layer_merge_name+ '_relu'))


        filters_in_cur_layer = filters_out




class LayerNamer():  # Class to provide names for each layer (to match saved weights in weights.h5 file)
    def __init__(self, prefix='', backend='keras'):
        self._backend = backend
        self._branch_id = 0
        self._prefix = prefix

    def get_block(self, stage, block, shortcut=False, conv_id=None):
        name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
        if not shortcut:
            conv_name, bn_name, act_name = 'conv', 'bn', 'relu'
        else:
            conv_name, bn_name, act_name = 'sc', 'sc_bn', 'sc_relu'

        conv_id_str = '' if conv_id is None else '%d' % conv_id
        zeropad_name = 'zeropad'

        name_dict = {'conv': self._prefix + name_base + conv_name + conv_id_str,
                     'batchnorm': self._prefix + name_base + bn_name  + conv_id_str,
                     'activation': self._prefix + name_base + act_name + conv_id_str,
                     'zeropad': self._prefix + name_base + zeropad_name + conv_id_str}

        return name_dict


    def get(self, layer_type):

        if layer_type == 'bn_input':
            #return self.rep('conv1')
            return self._prefix + 'bn_data'

        if layer_type == 'initial_pad':
            return self._prefix + 'zero_padding2d_1'  # 'conv1_pad'

        if layer_type == 'pooling':
            return self._prefix + 'pooling0'  # 'max_pooling2d_1'

        if layer_type == 'final_pool':
            return self._prefix + 'avg_pool'  # 'avg_pool'

        if layer_type == 'initial_conv':
            #return self.rep('conv1')
            return {'conv': self._prefix + 'conv0',
                    'batchnorm': self._prefix + 'bn0',
                    'activation': self._prefix + 'relu0',
                    'zeropad': self._prefix + 'zeropad'}

        if layer_type == 'final_bn':
            return 'bn1'
            #return {'conv': self._prefix + self.rep('conv0'),
            #        'batchnorm': self._prefix + self.rep('bn0'),
            #        'activation': self._prefix + self.rep('relu0'),
            #        'zeropad': self._prefix + self.rep('zeropad')}

            #return {'conv': self._prefix + self.rep('conv1'),
            #        'batchnorm': self._prefix + self.rep('bn0'),
            #        'activation': self._prefix + self.rep('relu0'),
            #        'zeropad': self._prefix + self.rep('zeropad')}

        if layer_type == 'logits':
            return self._prefix + 'Logits'  #self.rep('fc1000')

            #self._dense_id += 1
            #bn_name = 'fc%d_bn' % (self._dense_id)
            #act_name = 'fc%d_relu' % (self._dense_id)
            #name_dict = {'conv': self.rep(conv_name), 'batchnorm': self.rep(bn_name), 'activation': self.rep(act_name)}
            #return name_dict

        raise ValueError('unhandled input')




class LayerNamer_old():  # Class to provide names for each layer (to match saved weights in weights.h5 file)
    def __init__(self, prefix='', backend='keras'):
        self._backend = backend
        self._branch_id = 0
        self._prefix = prefix

    def rep(self, s):
        return s + '/' + s

    def get_block(self, group, rep, branch, stage=''):
        #basename = 'block%d_conv%c' % (block_id, rep_id)
        rep_letter = chr(ord('a') + rep)

        conv_name = 'res%d%c_branch%d%s' % (group, rep_letter, branch, stage)
        bn_name = 'bn%d%c_branch%d%s' % (group, rep_letter, branch, stage)
        relu_name = conv_name + '_relu'

        name_dict = {'conv': self._prefix + self.rep(conv_name),
                     'batchnorm': self._prefix + self.rep(bn_name),
                     'activation': self._prefix + self.rep(relu_name),
                     'zeropad': self._prefix + self.rep('zeropad')}

        return name_dict

    def get(self, layer_type):

        if layer_type == 'initial_conv':
            #return self.rep('conv1')
            return {'conv': self._prefix + self.rep('conv1'),
                    'batchnorm': self._prefix + self.rep('bn_conv1'),
                    'activation': self._prefix + self.rep('relu_conv1')}

        if layer_type == 'dense':
            return self._prefix + self.rep('fc1000')
            #self._dense_id += 1
            #bn_name = 'fc%d_bn' % (self._dense_id)
            #act_name = 'fc%d_relu' % (self._dense_id)
            #name_dict = {'conv': self.rep(conv_name), 'batchnorm': self.rep(bn_name), 'activation': self.rep(act_name)}
            #return name_dict





def GetResNet(num_layers):

    if num_layers == 18:
        return ResNet18
    elif num_layers == 34:
        return ResNet34
    elif num_layers == 50:
        return ResNet50
    elif num_layers == 101:
        return ResNet101
    else:
        raise ValueError('Invalid number of ResNet layers : %d' % num_layers)



def ResNet18(config, x_placeholder, all_quantizers=None):
    return ResNet(config, x_placeholder, num_layers=18, all_quantizers=all_quantizers)

def ResNet34(config, x_placeholder, all_quantizers=None):
    return ResNet(config, x_placeholder, num_layers=34, all_quantizers=all_quantizers)

def ResNet50(config, x_placeholder, all_quantizers=None):
    return ResNet(config, x_placeholder, num_layers=50, all_quantizers=all_quantizers)

def ResNet101(config, x_placeholder, all_quantizers=None):
    return ResNet(config, x_placeholder, num_layers=101, all_quantizers=all_quantizers)



def ResNet18_base_cifar(config, x_placeholder, all_quantizers):  # Saurabh's version, for CIFAR10

    ResNet18model = gtc.GTCModel()
    ResNet18model.add(x_placeholder, name='input')

    # conv
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[0],filters=16,kernel_size=3,stride=1,name='conv2d',use_bias=False,input_shape=config.image_size),inputs='input')
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization'))
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(), activation='relu', name='relu'))

    # ResNet18model.add(#    gtcPooling(#        pool_function=tf.nn.max_pool,#        strides=2,#        pool_size=3,#        name='max_pool_1'))

    # Block 1, conv 1
    # first projection
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[1],filters=16,kernel_size=1,stride=1,padding='SAME',use_bias=False,name='conv2d_1'),inputs='relu')
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_1'))

    # conv 2
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[2],filters=16,kernel_size=3,stride=1,padding='SAME',use_bias=False,name='conv2d_2'),inputs=['relu'])

    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_2'))
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(), activation='relu',name='relu_2'))

    # conv 3
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[3],filters=16,kernel_size=3,use_bias=False,name='conv2d_3'))
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_3'))

    # adding the residual connections
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='linear',name='res_1'),inputs=['batch_normalization_1', 'batch_normalization_3'])

    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(), activation='relu',name='relu_3'))

    # conv 4
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[4],filters=16,kernel_size=3,use_bias=False,name='conv2d_4'),inputs=['relu_3'])
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_4'))
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(), activation='relu',name='relu_4'))

    # covn 5
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[5],filters=16,kernel_size=3,use_bias=False,name='conv2d_5'))
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_5'))
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='linear',name='res_2'),inputs=['res_1', 'batch_normalization_5'])
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(), activation='relu',name='relu_5'))

    # conv 6
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[6],filters=16,kernel_size=3,use_bias=False,name='conv2d_6'),inputs=['relu_5'])
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_6'))
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(), activation='relu',name='relu_6'))

    # covn 7
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[7],filters=16,kernel_size=3,use_bias=False,name='conv2d_7'))

    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_7'))

    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='linear',name='res_3'),inputs=['res_2', 'batch_normalization_7'])

    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(), activation='relu',name='relu_7'))

    # second shortcut connections
    # Block 2
    # conv 8
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[8],filters=32,kernel_size=1,stride=2,padding='SAME',use_bias=False,name='conv2d_8'),inputs='relu_7')
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_8'))

    # conv 9
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[9],filters=32,kernel_size=3,stride=2,padding='SAME',use_bias=False,name='conv2d_9'),inputs=['relu_7'])
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_9'))
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(), activation='relu',name='relu_9'))

    # conv 10
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[10],filters=32,kernel_size=3,use_bias=False,name='conv2d_10'))
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_10'))

    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='linear',name='res_4'),inputs=['batch_normalization_8', 'batch_normalization_10'])

    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='relu',name='relu_10'))

    # conv 11
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[11],filters=32,kernel_size=3,use_bias=False,name='conv2d_11'),inputs=['relu_10'])
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_11'))
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='relu',name='relu_11'))

    # conv 12
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[12],filters=32,kernel_size=3,use_bias=False,name='conv2d_12'))
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_12'))

    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='linear',name='res_5'),inputs=['res_4', 'batch_normalization_12'])

    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='relu',name='relu_12'))

    # conv 13
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[13],filters=32,kernel_size=3,use_bias=False,name='conv2d_13'),inputs=['relu_12'])
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_13'))
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='relu',name='relu_13'))

    # conv 14
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[14],filters=32,kernel_size=3,use_bias=False,name='conv2d_14'))
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_14'))

    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='linear',name='res_6'),inputs=['res_5', 'batch_normalization_14'])

    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='relu',name='relu_14'))

    # Block 3
    # shortcut 3, convd2_15
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[15],filters=64,kernel_size=1,stride=2,padding='SAME',use_bias=False,name='conv2d_15'),inputs='relu_14')

    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_15'))

    # conv 16
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[16],filters=64,kernel_size=3,stride=2,padding='SAME',use_bias=False,name='conv2d_16'),inputs=['relu_14'])
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_16'))
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='relu',name='relu_16'))
    # conv 17
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[17],filters=64,kernel_size=3,use_bias=False,name='conv2d_17'))
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_17'))
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='linear',name='res_7'),inputs=['batch_normalization_15', 'batch_normalization_17'])

    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='relu',name='relu_17'))
    #
    # conv 18
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[18],filters=64,kernel_size=3,use_bias=False,name='conv2d_18'))
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_18'))
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='relu',name='relu_18'))

    # conv 19
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[19],filters=64,kernel_size=3,use_bias=False,name='conv2d_19'))
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_19'))

    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='linear',name='res_8'),inputs=['res_7', 'batch_normalization_19'])

    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='relu',name='relu_19'))

    # conv 20
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[20],filters=64,kernel_size=3,stride=1,use_bias=False,name='conv2d_20'),inputs='relu_19')
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_20'))

    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='relu',name='relu_20'))

    # conv 21
    ResNet18model.add(gtc.Conv2D(quantizer=all_quantizers[21],filters=64,kernel_size=3,stride=1,use_bias=False,name='conv2d_21'))
    ResNet18model.add(gtc.BatchNormalization(name='batch_normalization_21'))
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='linear',name='res_9'),inputs=['res_8', 'batch_normalization_21'])
    ResNet18model.add(gtc.Activation(quantizer=quant.IdentityQuantization(),activation='relu',name='relu_21'))

    # TODO: RESNET POOLING!
    ResNet18model.add(gtc.MaxPooling2D(strides=8,pool_size=8,name='pool_1'),inputs=['relu_21'])

    ResNet18model.add(gtc.Flatten(name='flatten_1'), inputs=['pool_1'])
    ResNet18model.add(gtc.Dense(quantizer=all_quantizers[22],units=10,use_bias=True,name='dense'))

    return ResNet18model






def single_ResNet18(config, x_placeholder):
    print('single quantizer for all ')
    quantizers = quant_utl.QuantizerSet(quant_utl.getFixedResnetQuantIdxs(num_quant=1))
    return ResNet18_base_cifar(config, x_placeholder, quantizers)


def f18_ResNet18(config, x_placeholder):
    print('seperate quantizer for all ')
    quantizers = quant_utl.QuantizerSet(num_layers=23)
    return ResNet18_base_cifar(config, x_placeholder, quantizers)


def f12_ResNet18(config, x_placeholder):
    print('f12 quantizer')
    quantizers = quant_utl.QuantizerSet(quant_utl.getFixedResnetQuantIdxs(num_quant=12))
    return ResNet18_base_cifar(config, x_placeholder, quantizers)


def nine_ResNet18(config, x_placeholder):
    print('nine quantizer')
    quantizers = quant_utl.QuantizerSet(quant_utl.getFixedResnetQuantIdxs(num_quant=9))
    return ResNet18_base_cifar(config, x_placeholder, quantizers)

def six_ResNet18(config, x_placeholder):
    print('six quantizer')
    quantizers = quant_utl.QuantizerSet(quant_utl.getFixedResnetQuantIdxs(num_quant=6))
    return ResNet18_base_cifar(config, x_placeholder, quantizers)


def five_ResNet18(config, x_placeholder):
    print('five quantizer')
    quantizers = quant_utl.QuantizerSet(quant_utl.getFixedResnetQuantIdxs(num_quant=5))
    return ResNet18_base_cifar(config, x_placeholder, quantizers)


def four_ResNet18(config, x_placeholder):
    print('four quantizer')
    quantizers = quant_utl.QuantizerSet(quant_utl.getFixedResnetQuantIdxs(num_quant=4))
    return ResNet18_base_cifar(config, x_placeholder, quantizers)


def three_ResNet18(config, x_placeholder):
    print('three quantizer')
    quantizers = quant_utl.QuantizerSet(quant_utl.getFixedResnetQuantIdxs(num_quant=3))
    return ResNet18_base_cifar(config, x_placeholder, quantizers)


def two_ResNet18(config, x_placeholder):
    print('two quantizer')
    quantizers = quant_utl.QuantizerSet(quant_utl.getFixedResnetQuantIdxs(num_quant=2))
    return ResNet18_base_cifar(config, x_placeholder, quantizers)


def ResNet18ID(config, x_placeholder):
    quantizers = quant_utl.QuantizerSet(num_layers=23, identity=True)
    return ResNet18_base_cifar(config, x_placeholder, quantizers)




if __name__ == "__main__":
    import keras
    keras.backend.set_learning_phase(False)
    #import classification_models.keras
    #ResNet18, preprocess = classification_models.keras.Classifiers.get('resnet18')
    #model = ResNet18((224, 224, 3), weights='imagenet')

    test_save_load_json = False
    if test_save_load_json:
        weights_file = 'F:/SRI/bitnet/test/resnet18.h5'
        model_file = weights_file.replace('.h5', '.json')
        weights_file_hp = weights_file.replace('.h5', '_hp.h5')
        weights_file_lp = weights_file.replace('.h5', '_lp.h5')
        model_new = gtc.singlePrecisionKerasModel(model_file=model_file, weights_file=weights_file_hp)

    # nlayers_test = [18, 34, 50, 101, 152]
    nlayers_test = [18]
    # nlayers_test = [101, 152]
    sess = tf.InteractiveSession()
    for nlayers in nlayers_test:
        net_type = net.NetworkType('resnet', nlayers, 'ic')

        # Test compilation of Resnet networks
        config = net.NetConfig(net_style=net_type.style, num_classes=1000)

        x_placeholder = tf.placeholder(
            tf.float32, [None]+list(config.image_size[:3]) )

        print('ResNet-%d ...' % nlayers)
        model = ResNet(config, x_placeholder, num_layers=nlayers)
        model.compile(verbose=True)
        model.print_model()

        reader = net.get_weights_reader(net_type)
        model.load_pretrained_weights_from_reader(reader, sess=sess)
        print('Succesfully loaded pretrained weights for %s' % str(net_type))

        model.print_model()
        model.initialize(initialize_weights=True)

        print('  Succesfully compiled ResNet-%d' % nlayers)
        print('  --- Done!')

        test_save_load_json = True
        if test_save_load_json:
            weights_file = 'F:/SRI/bitnet/test/resnet18.h5'
            model_file = weights_file.replace('.h5', '.json')
            weights_file_hp = weights_file.replace('.h5', '_hp.h5')
            weights_file_lp = weights_file.replace('.h5', '_lp.h5')
            model.save_model_def(model_file)
            model.save_weights(weights_file)
            model.save_weights_for_keras_model(weights_file_hp, hp=True)
            model.save_weights_for_keras_model(weights_file_lp, hp=False)

            model_new = gtc.singlePrecisionKerasModel(model_file=model_file, weights_file=weights_file_hp)

    print('Completed all')
