import gtc
#from config import config
from gtc import gtcModel
import tensorflow.compat.v1 as tf
#from gtc.batch_normalization import BatchNormalization as gtcBatchNorm
from quantization import quant_utils as quant_utl

from networks import net_utils as net

# TODO: put dropout rate in config

def get_vgg_specs(num_layers):

    filters = [64, 128, 256, 512, 512]
    if num_layers == 11:        # VGG 11
        reps = [1, 1, 2, 2, 2]
    elif num_layers == 13:      # VGG 13
        reps = [2, 2, 2, 2, 2]
    elif num_layers == 16:      # VGG 16
        reps = [2, 2, 3, 3, 3]
    elif num_layers == 19:      # VGG 19
        reps = [2, 2, 4, 4, 4]
    else:
        raise ValueError('Unsupported number of VGG layers')

    vgg_specs = {'filters': filters, 'reps': reps}
    return vgg_specs



def VGG(config, x_placeholder, num_layers=16, all_quantizers=None):

    network_name = net.get_network_full_name_with_tag(
        config, default='VGG%d_GTC' % num_layers)

    if all_quantizers is None:
        all_quantizers = quant_utl.QuantizerSet(num_layers, config=config,
                                                prefix=network_name)

    vgg_specs = get_vgg_specs(num_layers)
    filters = vgg_specs['filters']
    reps = vgg_specs['reps']

    filters_in = [config.image_size[-1]] + filters[:-1]

    assert sum(reps) == num_layers-3  # VGG16 has 13 conv layers + 3 dense layers

    if config.net_style == 'tensorflow':
        defaults_abbrev = '_tf'
    elif config.net_style == 'keras.applications':
        defaults_abbrev = '_keras'
    elif config.net_style == 'image-classifiers':
        defaults_abbrev = '_ic'
    else:
        raise ValueError('Unrecognized model defaults : %s' % config.net_style)


    namer = LayerNamer(prefix=network_name)

    vgg_type = net.NetworkType('vgg', num_layers=num_layers, style=config.net_style)
    vgg_config = net.VGGDefaults(vgg_type, config)

    conv_args = vgg_config.conv_kwargs
    bn_args = vgg_config.bn_kwargs


    model = gtcModel.GTCModel(name=network_name)
    model.type = net.NetworkType(name='vgg', num_layers=num_layers,
                                  style=config.net_style)
    model.add(x_placeholder, name='input')


    if config.include_preprocessing:
        defaults_name = 'vgg' + defaults_abbrev
        model.add( gtc.Preprocess(defaults=defaults_name))

    # 0b. Quantizing inputs
    if config.quantize_input:
        model.add_linear_quantizer(config.num_bits_linear, quant_name='LQ_in',
                                    layer_name='input_quantized')


    pool_strides = vgg_config.pool_strides
    pool_sizes = vgg_config.pool_sizes
    for stage_i in range(len(reps)):
        rep = reps[stage_i]

        for rep_i in range(rep):
            net.add_conv_bn_act_layers(
                model, filters[stage_i], kernel_size=3, activation='relu',
                quant_dict=all_quantizers.next(),
                name_dict=namer.get_block(stage_i+1, rep_i + 1), config=config, **conv_args)

        model.add(gtc.MaxPooling2D(
            strides=pool_strides[stage_i], pool_size=pool_sizes[stage_i],
            name= network_name + 'maxpool_%d' % (stage_i+1) ))


    all_conv = bool(config.fully_convolutional)

    units = vgg_config.fc_units
    #TODO: create dense_bn merged layer
    use_dense_bn_layer = True and not all_conv
    if use_dense_bn_layer:
        #model.add( gtcReshape([1, 1, -1]) )  # like 'flatten', but preserves spatial dimension.
        model.add( gtc.Flatten(name='flatten1') )  # like 'flatten', but preserves spatial dimension.
        #model.add( gtc.Dense_bn(units=4096, name_dict=namer.get('dense'), quantizer=all_quantizers.next(), config=config, **kwargs)  )
        name_dict1 = namer.get('dense', 6)
        quant_dict1 = all_quantizers.next()
        model.add( gtc.Dense(units=units, name=name_dict1['conv'], quantizer=quant_dict1['conv'], config=config, **conv_args)  )
        model.add( gtc.BatchNormalization(config=config, name=name_dict1['batchnorm'], quantizer=quant_dict1['batchnorm'], **bn_args) )
        model.add( gtc.Activation('relu', name=name_dict1['activation'], quantizer=quant_dict1['activation']))
        model.add(gtc.Dropout(rate=.5, name='dropout_1'))

        name_dict2 = namer.get('dense', 7)
        quant_dict2 = all_quantizers.next()
        model.add( gtc.Dense(units=units, name=name_dict2['conv'], quantizer=quant_dict2['conv'], config=config, **conv_args)  )
        model.add( gtc.BatchNormalization(config=config,  name=name_dict2['batchnorm'], quantizer=quant_dict2['batchnorm'], **bn_args))
        model.add( gtc.Activation('relu', name=name_dict2['activation'], quantizer=quant_dict2['activation']))
        model.add(gtc.Dropout(rate=.5, name='dropout_2'))

    else:
        #model.add( gtcReshape([1, 1, -1]) )  # like 'flatten', but preserves spatial dimension.
        if not all_conv:
            model.add( gtc.Reshape( [1, 1, filters[4]*7*7]) )  # like 'flatten', but preserves spatial dimension.

        kernel_size1 = vgg_config.fc_kernel_size
        dilation_rate1 = vgg_config.fc_dilation_rate
        net.add_conv_bn_act_layers(model, filters=units, kernel_size=kernel_size1,
                                   activation='relu', dilation_rate=dilation_rate1,
                                   do_batchnorm=vgg_config.do_batchnorm,
                                   name_dict=namer.get('dense', 6), quant_dict=all_quantizers.next(),
                                   config=config, **conv_args, **bn_args)

        if config.do_dropout:
            model.add(gtc.Dropout(rate=.5, name=network_name + 'dropout_1'))

        net.add_conv_bn_act_layers(model, filters=units, kernel_size=1,
                                   activation='relu',
                                   do_batchnorm=vgg_config.do_batchnorm,
                                   name_dict=namer.get('dense', 7), quant_dict=all_quantizers.next(),
                                   config=config, **conv_args, **bn_args)

        if config.do_dropout:
            model.add(gtc.Dropout(rate=.5, name=network_name + 'dropout_2'))

    # model.add(gtc.Flatten(name='flatten_5'))
    #
    #quant = all_quantizers.next()
    #model.add(gtc.Dense(units=4096, name=namer.get('dense'), quantizer=quant ))
    #model.add(gtc.BatchNormalization(name='bn_6', quantizer=quant))
    #model.add(gtc.Activation(quantizer=IdentityQuantization(),activation='relu', name=namer.get('dense_relu')))
    #
    #model.add(gtcDropout(rate=.5, name='dropout_1'))
    #
    #quant = all_quantizers.next()
    #model.add(gtc.Dense(units=4096,name=namer.get('dense'), quantizer=quant))
    #model.add(gtc.BatchNormalization(name='bn_7', quantizer=quant))
    #model.add(gtc.Activation(quantizer=IdentityQuantization(),activation='relu', name=namer.get('dense_relu')))
    #
    #model.add(gtcDropout(rate=.5, name='dropout_2'))

    model.add(gtc.Flatten(name=network_name + 'flatten_final'))  # like 'flatten', but preserves spatial dimension.

    if config.quantize_features:
        model.add_linear_quantizer(config.num_bits_linear, quant_name='LQ_feat',
                                   layer_name='features_quantized')

    model.add(gtc.Dense(units=config.num_classes, use_bias=True,
                        quantizer=all_quantizers.next('conv'),
                        name=namer.get('Logits')) )

    if config.quantize_logits_output:
        model.add_linear_quantizer(config.num_bits_linear, quant_name='LQ_logits',
                                   layer_name='logits_quantized')


    all_quantizers.confirm_done(print_only=False)
    model.userdata['num_quant'] = num_layers
    return model



# VGG16 Model using GTC Layers
def GetVGG(num_layers):
    if num_layers == 16:
        return VGG16
    elif num_layers == 19:
        return VGG19
    else:
        raise ValueError('Invalid number of VGG layers : %d' % num_layers)


class LayerNamer():  # Class to provide names for each layer (to match saved weights in weights.h5 file)
    def __init__(self, prefix='', backend='keras'):
        self._backend = backend
        self._dense_id = 0
        self._prefix = prefix

    def rep(self, s):
        return s
        #return s + '/' + s

    def get_block(self, block_id, rep_id):
        conv_name = 'block%d_conv%d' % (block_id, rep_id)
        bn_name = 'block%d_bn%d' % (block_id, rep_id)
        act_name = 'block%d_relu%d' % (block_id, rep_id)
        zeropad_name = 'block%d_zeropad%d' % (block_id, rep_id)

        name_dict = {'conv': self._prefix + self.rep(conv_name),
                     'batchnorm': self._prefix + self.rep(bn_name),
                     'activation': self._prefix + self.rep(act_name),
                     'zeropad': self._prefix + self.rep(zeropad_name)}

        return name_dict

    def get(self, layer_type, id=None):

        if layer_type == 'dense':
            #self._dense_id += 1
            dense_id = id
            conv_name = 'fc%d' % (dense_id)
            bn_name = 'fc%d_bn' % (dense_id)
            act_name = 'fc%d_relu' % (dense_id)
            zeropad_name = 'fc%d_zeropad' % (dense_id)
            name_dict = {'conv': self._prefix + self.rep(conv_name),
                         'batchnorm': self._prefix + self.rep(bn_name),
                         'activation': self._prefix + self.rep(act_name),
                         'zeropad': self._prefix + self.rep(zeropad_name)}
            return name_dict

        if layer_type == 'dense_relu':
            dense_id = id
            basename = 'fc%d_relu' % dense_id
            return basename + '/' + basename

        elif layer_type == 'Logits':
            return self.rep('predictions')

        else:
            raise ValueError('unrecognized layer type : %s' % layer_type)


# VGG16 Model using GTC Layers
def VGG16(config, x_placeholder):
    return VGG(config, x_placeholder, num_layers=16)

def VGG19(config, x_placeholder):
    return VGG(config, x_placeholder, num_layers=19)




def single_VGG16(config, x_placeholder):
    print('single quantizer')
    quantizers = quant_utl.QuantizerSet( quant_utl.getFixedVGGQuantIdxs(num_quant=1) )
    return VGG(config, x_placeholder, num_layers=16, all_quantizers=quantizers)


def f12_VGG16(config, x_placeholder):
    print('f12 quantized')
    quantizers = quant_utl.QuantizerSet(quant_utl.getFixedVGGQuantIdxs(num_quant=12))
    return VGG(config, x_placeholder, num_layers=16, all_quantizers=quantizers)


def nine_VGG16(config, x_placeholder):
    print('nine quantized')
    quantizers = quant_utl.QuantizerSet(quant_utl.getFixedVGGQuantIdxs(num_quant=9))
    return VGG(config, x_placeholder, num_layers=16, all_quantizers=quantizers)



def six_VGG16(config, x_placeholder):
    print('six quantized')
    quantizers = quant_utl.QuantizerSet(quant_utl.getFixedVGGQuantIdxs(num_quant=6))
    return VGG(config, x_placeholder, num_layers=16, all_quantizers=quantizers)


def three_VGG16(config, x_placeholder):
    print('three quantized')
    quantizers = quant_utl.QuantizerSet(quant_utl.getFixedVGGQuantIdxs(num_quant=3))
    return VGG(config, x_placeholder, num_layers=16, all_quantizers=quantizers)


def all_VGG16(config, x_placeholder):
    print('all quantized')
    print('hp network is trainable', config.hp_trainable)
    quantizers = quant_utl.QuantizerSet(num_layers=16)
    return VGG(config, x_placeholder, num_layers=16, all_quantizers=quantizers)


def VGG16ID(config, x_placeholder):
    print('ID  quantized')
    quantizers = quant_utl.QuantizerSet(num_layers=16, identity=True)
    return VGG(config, x_placeholder, num_layers=16, all_quantizers=quantizers)


def all_VGG19(config, x_placeholder):
    #print('all quantized')
    #print('hp network is trainable', config.hp_trainable)
    quantizers = quant_utl.QuantizerSet(num_layers=19)
    return VGG(config, x_placeholder, num_layers=19, all_quantizers=quantizers)


if __name__ == "__main__":
    # Test compilation of VGG networks
    config = net.NetConfig(image_size=(300,300,3))
    nlayers_test = [11, 13, 16, 19]

    x_placeholder = tf.placeholder(
        tf.float32, [None, config.image_size[0], config.image_size[1], config.image_size[2]])

    for nlayers in nlayers_test:
        vggnet = VGG(config, x_placeholder, num_layers=nlayers)
        vggnet.compile()
        print('Succesfully compiled VGG-%d' % nlayers)
        print(' --> Done !')


