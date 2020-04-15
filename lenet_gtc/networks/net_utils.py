import gtc
import quantization
from utility_scripts import misc_utl as utl
from collections import namedtuple, OrderedDict
import tensorflow.compat.v1 as tf
import os
import shutil
import imageio
import tensorflow.keras as keras
import time
import wget
import numpy as np
from tqdm import tqdm
from utility_scripts import weights_readers as readers
from ssd import ssd_info
from ssd.ssd_utils import SSDConfig
from colorama import Fore
import hickle as hkl
import warnings
import PIL.Image
import test_code.model_activation_names as act_names
import six
import io
from collections import Iterable
import csv
import classification_models.weights
import classification_models.keras
import re
import tensorflow.keras.backend as K
from google_drive_downloader import GoogleDriveDownloader
import h5py
import keras.engine.saving
#from networks.net_types import *
from networks.net_utils_core import *
import networks.net_types as net_typ
import xxhash
from ssd.ssd_utils import SSDConfig


from keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from keras.applications.vgg16 import preprocess_input as preprocess_vgg_resnet


def MobileNet(mobilenet_type):
    from networks.mobilenet_v1 import MobileNet_v1, get_mobilenet_v1_specs, LayerNamer as MobileNet_v1_namer
    from networks.mobilenet_v2 import MobileNet_v2, get_mobilenet_v2_specs, LayerNamer as MobileNet_v2_namer
    from networks.mobilenet_v3 import MobileNet_v3, MobileNet_v3_Large, MobileNet_v3_Small, \
        get_mobilenet_v3_small_specs, get_mobilenet_v3_large_specs, LayerNamer as MobileNet_v3_namer

    if isinstance(mobilenet_type, int):
        mobilenet_type = MobilenetType(version=mobilenet_type)

    if mobilenet_type.version == 1:
        return MobileNet_v1
    elif mobilenet_type.version == 2:
        return MobileNet_v2
    elif mobilenet_type.version == 3:
        if mobilenet_type.size.lower() == 'small':
            return MobileNet_v3_Small
        else:
            assert mobilenet_type.size.lower() == 'large'
            return MobileNet_v3_Large
    else:
        raise ValueError('Invalid mobilenet version')

def GetNetwork(net_type):
    assert isinstance(net_type, NetworkType)

    from networks.ssdnet import SSDNet

    from networks.VGG import VGG, VGG16, VGG19, GetVGG
    from networks.ResNet import ResNet, ResNet18, ResNet34, ResNet50, ResNet101, GetResNet
    from networks.lenet import LeNet, LeNet_v2, mlp_gtc
    from networks.keras_lenet import LeNet_keras, mlp_keras

    from networks.mobilenet_v2_keras import MobileNetV2 as MobileNetV2_keras
    from networks.ssd_mobilenet_keras import mobilenet_v1_ssd_300 as MobileNetV1_SSD_keras
    from networks.ssd_vgg_keras import ssd_300 as SSD300_keras

    if isinstance(net_type, SSDNetType):
        return SSDNet

    if isinstance(net_type, MobilenetType):
        return MobileNet(net_type)

    elif net_type.name == 'vgg':
        return GetVGG(net_type.num_layers)

    elif net_type.name == 'resnet':
        return GetResNet(net_type.num_layers)

    elif net_type.name == 'lenet':
        return LeNet

    else:
        raise ValueError('Unrecognized network type')


import tensorflow.keras.layers as krs

def add_conv_bn_act_layers(model, filters=None, kernel_size=None, stride=1,
                           explicit_zero_prepadding=False, do_batchnorm=None,
                           conv_before_bn=True,
                           activation=None,
                           name_dict=None, base_name=None,
                           conv_name='conv', bn_name='batchnorm',
                           act_name='activation', zeropad_name = 'zeropad',
                           config=None, quant_dict=None, quantizer=None, **kwargs):

    if do_batchnorm is None:  # can override config.do_batchnorm if we want
        do_batchnorm = config.do_batchnorm

    do_merged_batchnorm = config.do_merged_batchnorm and do_batchnorm
    trainable = config.hp_trainable

    if config.explicit_zero_prepadding_for_stride2 and stride == 2:
        explicit_zero_prepadding = True

    if name_dict is not None:
        conv_name = name_dict['conv']
        bn_name = name_dict['batchnorm']
        act_name = name_dict['activation']
        zeropad_name = name_dict['zeropad']
    elif base_name is not None:
        conv_name = base_name
        bn_name = base_name + '_bn'
        act_name = base_name + '_act'
        zeropad_name = base_name + '_zeropad'


    if isinstance(quant_dict, dict):
        conv_quant = quant_dict['conv']
        bn_quant = quant_dict['batchnorm']
        act_quant = quant_dict['activation']
    elif isinstance(quantizer, quantization.Quantization):
        conv_quant = quantizer
        bn_quant = quantizer
        act_quant = quantizer
    elif quantizer is None and quant_dict is None:
        conv_quant, bn_quant, act_quant = None, None, None
    else:
        raise ValueError('invalid quant_dict or quantizer argument')


    # These inputs are only for conv/conv-bn modules, not for batchnorm or activation
    conv_padding = kwargs.pop('padding', None)
    padding = None
    if explicit_zero_prepadding:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        vert_pad, horiz_pad = kernel_size[0] -1,  kernel_size[1]-1
        top_pad = vert_pad // 2
        bot_pad = vert_pad - top_pad
        left_pad = horiz_pad//2
        right_pad = horiz_pad-left_pad
        padding = ((top_pad, bot_pad), (left_pad, right_pad))

        #model._num_zeropad_layers += 1
        conv_padding = 'VALID'  # overwrite any input value for 'padding'


    conv_kwargs = utl.remove_None_values(
        dict(filters=filters, strides=stride, kernel_size=kernel_size,
             kernel_initializer=kwargs.pop('kernel_initializer', None),
             bias_initializer=kwargs.pop('bias_inintializer', None),
             kernel_regularizer=kwargs.pop('kernel_regularizer', None),
             bias_regularizer=kwargs.pop('bias_regularizer', None),
             dilation_rate=kwargs.pop('dilation_rate', None),
             use_bias=kwargs.pop('use_bias', None),
             bias_quantizer=config.bias_quantizer,
             padding=conv_padding,
             layer_inputs=kwargs.pop('layer_inputs', None),
             layer_input=kwargs.pop('layer_input', None),
             trainable=trainable) )

    bn_kwargs = utl.remove_None_values ( dict(
                     momentum=kwargs.pop('momentum', None),
                     epsilon=kwargs.pop('epsilon', None) ) )


    # Fused Conv+Batchnorm (or just Conv)
    if do_merged_batchnorm:

        if explicit_zero_prepadding:
            model.add(gtc.ZeroPadding2D(padding, name=zeropad_name))

        model.add( gtc.Conv2D_bn(
                quantizer=conv_quant,
                name=conv_name+'.'+bn_name,
                **conv_kwargs,
                **bn_kwargs,
                **kwargs) )

        # Activation
        if activation is not None:
            model.add(gtc.Activation(activation=activation, quantizer=act_quant,
                                     name=act_name, **kwargs))

    else:  # Convolution and batchnorm in separate layers.

        # The batchnorm can either be BEFORE the conv layer (unusual)
        if not conv_before_bn:
            if do_batchnorm:
                model.add( gtc.BatchNormalization(
                    quantizer=bn_quant, share=config.batchnorm_share,
                    name=bn_name, **bn_kwargs, **kwargs) )

            # Activation
            if activation is not None:
                model.add(gtc.Activation(activation=activation, quantizer=act_quant,
                                         name=act_name, **kwargs))


        if explicit_zero_prepadding:
            model.add(gtc.ZeroPadding2D(padding, name=zeropad_name))

        # Add the conv layer
        model.add( gtc.Conv2D( quantizer=conv_quant,
                               **conv_kwargs, name=conv_name, **kwargs) )


        # ... or the batchnorm can be AFTER the conv (typical)
        if conv_before_bn:
            if do_batchnorm:
                model.add( gtc.BatchNormalization(
                    quantizer=bn_quant, share=config.batchnorm_share,
                    name=bn_name, **bn_kwargs, **kwargs) )

            # Activation
            if activation is not None:
                model.add(gtc.Activation(activation=activation, quantizer=act_quant,
                                         name=act_name, **kwargs))



def add_separable_conv_layers(model, filters=None, kernel_size=3, strides=None,
                              activation=None, activation1=None, activation2=None,
                              name_dict_depth=None, name_dict_point=None,                              
                              kwargs_depth=None, kwargs_point=None,
                              quant_dict_depth=None, quant_dict_point=None,
                              base_name=None,
                              config=None, quantizers=None, **kwargs):
    if kernel_size == 1:
        raise ValueError('Expected kernel_size > 1 for depth-separable convolutions')

    if activation is not None:
        assert activation1 is None and activation2 is None, \
            "Either specify activation, or activation1 and/or activation2"

        activation1 = activation
        activation2 = activation

    if quantizers is not None:
        assert quant_dict_depth is None and quant_dict_point is None, \
            "Either specify quantizers, or quantizer_depth and quantizer_point"

    if strides is None:
        strides = [1, 1]
    elif isinstance(strides, int):
        strides = [strides, 1]
    else:
        strides = strides

    # if there is a layer_input argument, make sure it only goes to the first
    # (depthwise) layer
    layer_input = kwargs.pop('layer_input', None)

    if kwargs_depth is None:
        kwargs_depth = {}

    if name_dict_depth is None and base_name is not None:
        kwargs_depth['base_name'] = base_name + '_depth'


    if quant_dict_depth is None and quantizers is not None:
        quant_dict_depth = quantizers.next()

    # first do depthwise (kxk) convolution
    add_conv_bn_act_layers(model, 
        filters='depthwise', kernel_size=kernel_size, stride=strides[0],
        activation=activation1,
        name_dict=name_dict_depth,
        quant_dict=quant_dict_depth,
        layer_input=layer_input,
        config=config, **kwargs, **kwargs_depth)


    # then do pointwise 1x1 feature-convolution
    if kwargs_point is None:
        kwargs_point = {}

    if name_dict_point is None and base_name is not None:
        kwargs_point['base_name'] = base_name + '_point'

    if quant_dict_point is None and quantizers is not None:
        quant_dict_point = quantizers.next()

    add_conv_bn_act_layers(model,
        filters=filters, kernel_size=1,  stride=strides[1],
        activation=activation2,
        name_dict=name_dict_point,
        quant_dict=quant_dict_point,
        config=config, **kwargs, **kwargs_point)




def correct_post_pad(input_size, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    # img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    # input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))






def get_mobilenet_tensorflow_checkpoint_url(mobilenet_type):
    # For mobilenet-v1,    0.5, 0.75 --> '0.50', '0.75'.
    # For mobilenet-v2/v3, 0.5, 0.75 --> '0.5',  '0.75'
    assert mobilenet_type.style == 'tensorflow', \
        "Can only get checkpoints for tensorflow models"

    depth_pattern = '%.f' if mobilenet_type.version == 1 else '%g'

    depth_mult_str = '1.0' if mobilenet_type.depth_multiplier == 1 \
        else depth_pattern % mobilenet_type.depth_multiplier

    if mobilenet_type.version == 1:
        # eg: 'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz',
        url = 'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_%s_%d.tgz' % \
              (depth_mult_str, mobilenet_type.imsize)

    elif mobilenet_type.version == 2:
        # eg: 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz',
        url = 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_%s_%d.tgz' % \
              (depth_mult_str, mobilenet_type.imsize)

    elif mobilenet_type.version == 3:
        # eg: 'https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz'
        url = 'https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-%s_%d_%s_float.tgz' % \
              (mobilenet_type.size.lower(), mobilenet_type.imsize, depth_mult_str)

    else:
        raise ValueError('Invalid mobilenet version : %s ' % str(mobilenet_type.version))

    checkpoint_name = os.path.basename(url)
    if mobilenet_type.version == 3:
        checkpoint_name = 'mobilenet_' + checkpoint_name

    return url, checkpoint_name


def get_mobilenet_v3_filename(mobilenet_type):
    iter_id = '-540000' if mobilenet_type.size.lower() == 'large' else '-388500'
    filename_prefix = 'model' if mobilenet_type.ema else 'model.ckpt'
    filename = filename_prefix + iter_id
    return filename


ssd_tf_ckpt = namedtuple('ssd_tf_ckpt', ['checkpoint', 'placeholder_name', 'imsize'])

def get_keras_apps_vgg_weights_url(nlayers=16):
    if nlayers == 16:
        WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                        'releases/download/v0.1/'
                        'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    else:
        raise ValueError('Invalid number of layers')
    return WEIGHTS_PATH

def get_keras_apps_resnet_weights_url(nlayers=50):
    use_fchollet_weights = False
    if nlayers == 50 and use_fchollet_weights:
        file_name_remote = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        file_name = file_name_remote.replace('.h5', '_fchollet.h5')
        weights_url = 'https://github.com/fchollet/deep-learning-models/' \
                      'releases/download/v0.2/' + file_name_remote
        file_hash = 'a7b3fe01876f51b976af0dea6bc144eb'

    elif nlayers in [50, 101, 152]:
        model_name = 'resnet%d' % nlayers
        weights_hashes = {'resnet50': '2cb95161c43110f7111970584f804107',
                       'resnet101': 'f1aeb4b969a6efcfb50fad2f0c20cfc5',
                       'resnet152': '100835be76be38e30d865e96f2aaae62'}

        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
        file_hash = weights_hashes[model_name]
        weights_url = (
            'https://github.com/keras-team/keras-applications/'
            'releases/download/resnet/') + file_name

    else:
        raise ValueError('Invalid number of layers')

    weights_path = keras.utils.get_file(file_name,
                                        weights_url,
                                        cache_subdir='models',
                                        file_hash=file_hash)

    return weights_path


def get_mobilenet_keras_weights_url(mobilenet_type):
    if mobilenet_type.version == 1:
        base_url = ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.6/')

        depth_multiplier_str = '1.0' if mobilenet_type.depth_multiplier == 1 \
            else '%.f' % mobilenet_type.depth_multiplier

        depth_multiplier_str = depth_multiplier_str.replace('.', '_') # 1.0 -> 1_0

        model_name = 'mobilenet_%s_%d_tf.h5' % (depth_multiplier_str,
                                                mobilenet_type.imsize)


    elif mobilenet_type.version == 2:
        base_url = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/'
                            'releases/download/v1.1/')

        model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_%s_%d.h5' % (
            str(float(mobilenet_type.depth_multiplier)), mobilenet_type.imsize))

    else:
        raise ValueError('Invalid mobilenet version')

    WEIGHTS_PATH = base_url + model_name

    return WEIGHTS_PATH


def get_mobilenet_checkpoint_dir(mobilenet_type):
    if not isinstance(mobilenet_type, net_typ.MobilenetType):
        raise ValueError('expected mobilenet type')

    chkpoint_url, model_ckpt_basename = get_mobilenet_tensorflow_checkpoint_url(mobilenet_type)
    model_ckpt_name = os.path.splitext( model_ckpt_basename )[0]

    cache_subdir = 'models/' + model_ckpt_name + '/'
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
    chkpoint_dir = os.path.join(cache_dir, cache_subdir)

    if mobilenet_type.version in [1, 2]:
        chkpoint_file = chkpoint_dir + model_ckpt_name + '.ckpt.index'

    elif mobilenet_type.version == 3:
        subdir = 'ema' if mobilenet_type.ema else 'pristine'
        chkpoint_dir += model_ckpt_name.replace('mobilenet_', '') + '/' + subdir + '/'
        chkpoint_file = chkpoint_dir + get_mobilenet_v3_filename(mobilenet_type) + '.index'
    else:
        raise ValueError('Invalid mobilenet version : %d ' % mobilenet_type.version )

    #model_ckpt_archive_chk = chkpoint_dir + model_ckpt_basename
    if not os.path.exists(chkpoint_file):
        model_file_archive = tf.keras.utils.get_file(
            fname=model_ckpt_basename,
            cache_subdir=cache_subdir,
            origin=chkpoint_url, extract=True)

        shutil.move(model_file_archive, cache_dir + '/models/')

    return chkpoint_dir


def get_mobilenet_checkpoint(mobilenet_type):

    ckpt_dir = get_mobilenet_checkpoint_dir(mobilenet_type)

    if mobilenet_type.version == 3:
        filename = get_mobilenet_v3_filename(mobilenet_type)
        model_ckpt_name = ckpt_dir + filename
    else:
        model_ckpt_name = ckpt_dir + os.path.basename(os.path.dirname(ckpt_dir)) + '.ckpt'

    return model_ckpt_name





def get_keras_apps_weights_file(net_name):

    if net_name == 'vgg16':
        weights_url = get_keras_apps_vgg_weights_url(nlayers=16)
    elif 'resnet' in net_name:
        nlayers = int(re.search('resnet(\d+)', net_name).groups()[0])
        weights_url = get_keras_apps_resnet_weights_url(nlayers=nlayers)

    elif net_name == 'mobilenet_v1':
        mobilenet_type = net_typ.MobilenetType(version=1, style='keras')
        weights_url = get_mobilenet_keras_weights_url(mobilenet_type)

    elif net_name == 'mobilenet_v2':
        mobilenet_type = net_typ.MobilenetType(version=2, style='keras')
        weights_url = get_mobilenet_keras_weights_url(mobilenet_type)
    else:
        raise ValueError('Invalid network name')

    model_name = os.path.basename(weights_url)
    weights_path = keras.utils.get_file(model_name,
                                        weights_url,
                                        cache_subdir='models')

    return weights_path


def get_image_classifiers_weights_file(net_type):
    weights_specs = classification_models.weights._find_weights(
        str(net_type), 'imagenet', include_top=True)

    assert bool(weights_specs), "Could not fine matching weight file"
    weights_specs = weights_specs[0]

    weights_path = keras.utils.get_file(
        weights_specs['name'],
        weights_specs['url'],
        cache_subdir='models',
        md5_hash=weights_specs['md5']
    )

    return weights_path



def get_mobilenet_v1_ssd_keras_weights_file():
    mobilenet_v2_weights_file_url = \
        'https://github.com/abhinavrai44/MobilenetWeights/raw/' \
        'a95a946e50ebc5fa225276fe42c6506a263fd0b5/converted_model.h5'
    model_name = 'ssd_mobilenet_v1_keras_' + \
                 os.path.basename(mobilenet_v2_weights_file_url)
    weights_path = keras.utils.get_file(model_name,
                                        mobilenet_v2_weights_file_url,
                                        cache_subdir='models')

    return weights_path

def get_mobilenet_v1_ssd_base_keras_weights_file():
    mobilenet_v2_weights_file_url = \
        'https://github.com/abhinavrai44/MobilenetWeights/raw/' \
        'a95a946e50ebc5fa225276fe42c6506a263fd0b5/mobilenet.h5'

    model_name = 'ssd_mobilenet_v1_keras_imagenet_base__' + \
                 os.path.basename(mobilenet_v2_weights_file_url)
    weights_path = keras.utils.get_file(model_name,
                                        mobilenet_v2_weights_file_url,
                                        cache_subdir='models')

    return weights_path






def get_ssd_weights_reader(ssd_net_type, verbose=True):
    assert isinstance(ssd_net_type, net_typ.SSDNetType)
    base_net = ssd_net_type.base_net_type

    if isinstance(base_net, net_typ.MobilenetType):
        if 'keras' in base_net.style:
            assert base_net.version == 1 and not ssd_net_type.ssd_lite
            name_converter = readers.MobilenetSSDKerasWeightsNameConverter()
            weights_file = get_mobilenet_v1_ssd_keras_weights_file()
            reader = readers.H5WeightsFileReader(weights_file, name_converter=name_converter, verbose=verbose)

        elif base_net.style == 'tensorflow':
            ckpt = ssd_info.get_tensorflow_model_ckpt(ssd_net_type)
            name_converter = readers.MobilenetSSD_TF_CheckpointNameConverter()
            reader = readers.CheckpointReader(ckpt, name_converter=name_converter)

        else:
            raise ValueError('Unrecognized model style : %s' % base_net.style)

    elif base_net.name.lower() == 'vgg':
        weights_file = get_vgg_ssd_keras_weights_file()
        name_converter = readers.VGGSSDWeightsNameConverter()
        reader = readers.H5WeightsFileReader(weights_file, name_converter=name_converter)

    else:
        raise ValueError('Unrecognized SSD type : %s' % str(ssd_net_type))

    return reader



def get_mobilenet_weights_reader(mobilenet_type, verbose=False):

    # Mobilenet classification network
    assert isinstance(mobilenet_type, net_typ.MobilenetType)

    if mobilenet_type.style == 'tensorflow':
        chkpoint_name = get_mobilenet_checkpoint(mobilenet_type)
        reader = readers.CheckpointReader(chkpoint_name, exponential_moving_average=mobilenet_type.ema)
    elif mobilenet_type.style_abbrev == 'keras':
        net_name = mobilenet_type.get_name(version_only=True)
        name_converter_keras = readers.KerasAppsWeightsNameConverter(net_name, verbose=verbose)
        keras_apps_weights_file = get_keras_apps_weights_file(net_name)
        reader = readers.H5WeightsFileReader(keras_apps_weights_file, name_converter=name_converter_keras)

    else:
        raise TypeError('Expected mobilenet style')

    return reader


def get_weights_reader(net_type, verbose=False):
    assert isinstance(net_type, net_typ.NetworkType)

    if isinstance(net_type, net_typ.SSDNetType):
        return get_ssd_weights_reader(net_type, verbose)

    if isinstance(net_type, net_typ.MobilenetType):
        return get_mobilenet_weights_reader(net_type, verbose)

    else:
        net_name = net_type.get_name(version_only=True)
        if net_type.style == 'keras.applications':
            name_converter_keras = readers.KerasAppsWeightsNameConverter(net_name, verbose=verbose)
            keras_apps_weights_file = get_keras_apps_weights_file(net_name)
            return readers.H5WeightsFileReader(keras_apps_weights_file, name_converter=name_converter_keras)
        elif net_type.style == 'image-classifiers':
            name_converter_ic  = readers.ImageClassifiersWeightsNameConverter(net_name, verbose=verbose)
            image_classifiers_weights_file = get_image_classifiers_weights_file(net_name)
            return readers.H5WeightsFileReader(image_classifiers_weights_file, name_converter=name_converter_ic)




def get_vgg_ssd_keras_weights_file(model_file=None):

    modelfile_google_drive_ids = {
        'VGG_ILSVRC_16_layers_fc_reduced.h5': '1sBmajn6vOE7qJ8GnxUJt4fGPuffVUZox', # VG
        'VGG_VOC0712_SSD_300x300_iter_120000.h5': '121-kCXaOHOkJE_Kf5lKcJvC_5q1fYb_q',
        'VGG_VOC0712Plus_SSD_300x300_iter_240000.h5': '1M99knPZ4DpY9tI60iZqxXsAxX2bYWDvZ',
        'VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5': '1fyDDUcIOSjeiP08vl1WCndcFdtboFXua',
        'VGG_coco_SSD_300x300_iter_400000.h5': '1vmEF7FUsWfHquXyCqO17UaXOPpRbwsdj',
    }
    if model_file is None:
        model_file = 'VGG_VOC0712_SSD_300x300_iter_120000.h5'
    file_id = modelfile_google_drive_ids[model_file]

    cache_dir = os.path.join(os.path.expanduser('~'), '.keras') + '/models/ssd_keras/'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    dest_path = cache_dir + model_file
    if not os.path.exists(dest_path):
        GoogleDriveDownloader.download_file_from_google_drive(
            file_id=file_id, dest_path=dest_path, showsize=True)

    return dest_path





def get_preprocessor(net_type):
    assert isinstance(net_type, net_typ.NetworkType)

    if isinstance(net_type, net_typ.SSDNetType):
        return lambda x: x

    if isinstance(net_type, net_typ.MobilenetType):
        return preprocess_mobilenet

    elif net_type.style == 'keras.applications':
        assert net_type.name in ['vgg', 'resnet']
        return preprocess_vgg_resnet

    elif net_type.style == 'image-classifiers':
        # all image-classifiers networks have no preprocessing function,
        # except for the inception
        return lambda x : x



def get_sample_image(image_name='panda', imsize=(224, 224)):
    if image_name == 'panda':
        url = 'https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG'
        out_file = utl.repo_root() + 'test_code/panda.jpg'
        if not os.path.exists(out_file):
            wget.download(url, out=out_file)

        img_raw = np.array(PIL.Image.open(out_file).resize(imsize))
        return img_raw


    else:
        raise ValueError('Unrecognized sample image name')

class NetConfig():
    def __init__(self,
                 network_name='',
                 network_tag='',
                 image_size=(224, 224, 3),
                 include_preprocessing=True,
                 resize_inputs=True,
                 net_style='keras.applications',
                 net_type=None,
                 num_classes=1001,

                 conv_padding='valid',
                 stride2_use=None,
                 explicit_zero_prepadding_for_stride2=False,

                 do_batchnorm=True,
                 do_merged_batchnorm=False,
                 batchnorm_share='none',

                 bn_epsilon=None,
                 bn_momentum=None,
                 activation=None,
                 do_dropout=True,
                 do_weight_regularization=True,
                 decouple_weight_decay=True,
                 weight_decay_factor=4e-5,
                 fully_convolutional=None, # for VGG, whether is fully convolutional base (for SSD head)

                 conv_quant='gtc', bn_quant='identity', act_quant='identity',
                 bias_quantizer=None,  # default: None=same as weights quantizer
                 quant_pre_skip=0,
                 quant_post_skip=0,
                 identity_quantization = False,
                 num_quantizers=None,
                 idxs_identity=None,
                 idxs_quant=None,
                 num_bits_linear=None,
                 linear_bits_fixed=False,
                 quantize_input=False,
                 quantize_features=False,
                 quantize_logits_output=False,
                 ssd_extra_quant=True,
                 ssd_box_quant=False,
                 ssd_box_share_quant=True,
                 ssd_background_bias=False,
                 freeze_classification_layers=False,

                 layer_name_prefix=None,
                 hp_trainable=True,
                 **kwargs):

        self.network_name = network_name
        self.network_tag = network_tag
        self.image_size = image_size
        self.resize_inputs = resize_inputs
        self.include_preprocessing = include_preprocessing
        assert net_style in ['tensorflow', 'keras.applications', 'image-classifiers']
        if net_type is not None:
            assert isinstance(net_type, net_typ.NetworkType)
            net_style = net_type.style
        self.net_style = net_style
        self.net_type = net_type
        self.num_classes = num_classes

        self.conv_padding = conv_padding
        self.stride2_use = stride2_use
        self.explicit_zero_prepadding_for_stride2 = explicit_zero_prepadding_for_stride2

        self.do_batchnorm = do_batchnorm
        self.do_merged_batchnorm = do_merged_batchnorm
        self.batchnorm_share=batchnorm_share

        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.activation = activation

        self.do_dropout = do_dropout
        self.do_weight_regularization = do_weight_regularization
        self.decouple_weight_decay = decouple_weight_decay
        self.weight_decay_factor = weight_decay_factor
        self.fully_convolutional = fully_convolutional

        self.conv_quant = conv_quant
        self.bn_quant = bn_quant
        self.act_quant = act_quant
        self.quant_pre_skip = quant_pre_skip
        self.quant_post_skip = quant_post_skip

        self.bias_quantizer = bias_quantizer
        self.num_quantizers = num_quantizers
        self.identity_quantization = identity_quantization
        self.idxs_identity = idxs_identity
        self.idxs_quant = idxs_quant
        self.num_bits_linear = num_bits_linear
        self.linear_bits_fixed = linear_bits_fixed
        self.quantize_input = quantize_input
        self.quantize_features = quantize_features
        self.quantize_logits_output = quantize_logits_output
        self.ssd_extra_quant = ssd_extra_quant
        self.ssd_box_quant = ssd_box_quant
        self.ssd_box_share_quant = ssd_box_share_quant
        self.ssd_background_bias = ssd_background_bias
        self.freeze_classification_layers = freeze_classification_layers

        self.hp_trainable = hp_trainable
        self.layer_name_prefix = layer_name_prefix

        raise_error_for_unrecognized_args = True
        found_unrecognized_args = False
        for k,v in kwargs.items():
            print('Warning: unrecognized argument: "%s"' % k)
            setattr(self, k, v)
            found_unrecognized_args = True

        if found_unrecognized_args and raise_error_for_unrecognized_args:
            raise ValueError('Found unrecognized arguments.')


    def copy(self):
        new_config = NetConfig()
        for k,v in self.__dict__.items():
            setattr(new_config, k, v)
        return new_config


def get_network_full_name_with_tag(config, default='GTCModel'):
    assert isinstance(config, NetConfig)
    network_name = default

    if bool(config.network_name):
        network_name = config.network_name

    if bool(config.network_tag):
        if network_name.endswith('/'):
            network_name = network_name[:-1] # remove '/' before adding tag.
        network_name += '_' + config.network_tag

    # Add '/' if not already there
    if not network_name.endswith('/'):
        network_name += '/'

    network_name_orig = network_name
    network_name = get_unique_tf_prefix(network_name_orig)

    return network_name



def save_activations_of_keras_model(model, image, filename, save_weights=True):
    hkl_filename = filename
    if not hkl_filename.endswith('.hkl'):
        hkl_filename = hkl_filename + '.hkl'

    if image.ndim < 4:
        image = np.expand_dims(image, axis=0)

    layers = model.layers[1:]  # skip input layer
    layer_names = [x.name for x in layers]

    S_outputs = OrderedDict()
    for i,layer_i in tqdm(enumerate(layers), total=len(layers)):
        layer_name = layer_names[i]
        partial_model = keras.models.Model(inputs=model.input, outputs=layer_i.output)
        #layer_output = model._forward_prop_outputs[layer_name]
        partial_output = partial_model.predict(image)
        S_outputs[layer_name] = partial_output

        if save_weights:
            weights = layer_i.weights
            if len(weights) > 0:
                weight_names = [w.name for w in weights]
                weight_vals = layer_i.get_weights()
                S_outputs[layer_name + '_weights'] = OrderedDict(zip(weight_names, weight_vals))

    hkl.dump(S_outputs, filename)
    print('Saved network outputs to %s' % filename)
    hasher = lambda x: xxhash.xxh64(np.array(x)).hexdigest()
    np.set_printoptions(linewidth=220)
    save_txt_copy = True
    if save_txt_copy:
        txt_filename = hkl_filename.replace('.hkl', '.txt')
        with open(txt_filename, 'w') as f:
            for i, layer_name in enumerate(layer_names):
                t = S_outputs[layer_name]
                hash_str = hasher(t)
                f.write(' %d/%d : %s {%s} [shape=%s].\n' % (
                    i + 1, len(layer_names), layer_name, hash_str, str(t.shape)))
                f.write(get_tensor_str(t) + '\n\n')

                if save_weights:
                    weight_key = layer_name + '_weights'
                    if weight_key in S_outputs:
                        weights = S_outputs[layer_name + '_weights']
                        for j, (wgt_name, wgt_val) in enumerate(weights.items()):
                            f.write('  [%d/%d %s]  *WEIGHT* %d/%d %s {%s} [shape=%s].\n' % (
                                i + 1, len(layer_names), layer_name,
                                j + 1, len(weights.keys()), wgt_name, hasher(wgt_val),
                                str(wgt_val.shape)))
                            f.write(get_tensor_str(wgt_val) + '\n\n')

        print('Saved network outputs to %s' % txt_filename)
    a = 1

    pass


def get_tensor_str(t):
    if t.size <= 10:
        return str(t)
    else:
        ts = np.squeeze(t)
        if ts.ndim == 1:
            return str(ts[:10])
        elif ts.ndim == 2:
            return str(ts[:5, :5])
        elif ts.ndim == 3:
            return str(ts[:5, :5, 0])
        elif ts.ndim == 4:
            return str(ts[:5, :5, 0, 0])

    return '?'

def save_activations_of_gtc_model(model, sample_image, filename, save_weights=True):
    hkl_filename = filename
    if not hkl_filename.endswith('.hkl'):
        hkl_filename = hkl_filename + '.hkl'
    sess = tf.get_default_session()
    layer_names = model._layer_names
    placeholder = model.get_placeholder()
    if sample_image.ndim == 4:
        sample_image_use = sample_image
    else:
        sample_image_use = np.expand_dims(sample_image, axis=0)
    data_dict = {placeholder: sample_image_use}
    S_outputs = OrderedDict()

    all_names = []
    all_ops = []

    for layer_name in tqdm(layer_names):
        layer_output = model._forward_prop_outputs[layer_name]
        #output_hp, output_lp = sess.run([layer_output['hp'], layer_output['lp'], feed_dict=data_dict)
        #S_outputs[layer_name] = OrderedDict([('hp', output_hp), ('lp', output_lp)])
        all_names.extend([layer_name + '_hp', layer_name + '_lp'])
        all_ops.extend([layer_output['hp'], layer_output['lp']])
        if save_weights:
            layer_obj = model._layers_objects[layer_name]['layer_obj']
            if hasattr(layer_obj, 'get_hp_weights'):
                hp_weights = layer_obj.get_hp_weights()
                hp_weight_names = [w.name for w in hp_weights]
                #hp_weight_vals = sess.run(hp_weights)
                lp_weights = layer_obj.get_lp_weights()
                lp_weight_names = [w.name for w in lp_weights]
                #lp_weight_vals = sess.run(lp_weights)
                all_names.extend(hp_weight_names + lp_weight_names)
                all_ops.extend(hp_weights + lp_weights)
                #S_outputs[layer_name + '_weights'] = OrderedDict([
                #    ('hp', OrderedDict(zip(hp_weight_names, hp_weight_vals))),
                #    ('lp', OrderedDict(zip(lp_weight_names, lp_weight_vals)))
                #])
    all_vals = sess.run(all_ops, feed_dict=data_dict)
    S_outputs_temp = OrderedDict(zip(all_names, all_vals))


    for layer_name in tqdm(layer_names):
        S_outputs[layer_name] = OrderedDict([('hp', S_outputs_temp[layer_name + '_hp']),
                                             ('lp', S_outputs_temp[layer_name + '_lp'])])
        if save_weights:
            layer_obj = model._layers_objects[layer_name]['layer_obj']
            if hasattr(layer_obj, 'get_hp_weights'):
                hp_weight_names = [w.name for w in layer_obj.get_hp_weights()]
                lp_weight_names = [w.name for w in layer_obj.get_lp_weights()]
                hp_weight_vals = [S_outputs_temp[nm] for nm in hp_weight_names]
                lp_weight_vals = [S_outputs_temp[nm] for nm in lp_weight_names]
                S_outputs[layer_name + '_weights'] = OrderedDict([
                    ('hp', OrderedDict(zip(hp_weight_names, hp_weight_vals))),
                    ('lp', OrderedDict(zip(lp_weight_names, lp_weight_vals)))
                 ])

    hasher = lambda x: xxhash.xxh64(np.array(x)).hexdigest()
    hkl.dump(S_outputs, filename)
    np.set_printoptions(linewidth=220)
    save_txt_copy = True
    if save_txt_copy:
        txt_filename = hkl_filename.replace('.hkl', '.txt')
        with open(txt_filename, 'w') as f:
            for i,layer_name in enumerate(layer_names):
                for prec in ['hp', 'lp']:
                    t = S_outputs[layer_name][prec]
                    hash_str = hasher(t)
                    f.write(' %d/%d [%s] : %s {%s} [shape=%s].\n'  % (
                        i+1, len(layer_names), prec.upper(), layer_name, hash_str, str(t.shape) ) )
                    f.write(get_tensor_str(t) + '\n\n')

                    if save_weights:
                        weight_key = layer_name + '_weights'
                        if weight_key in S_outputs:
                            weights = S_outputs[layer_name + '_weights'][prec]
                            for j, (wgt_name, wgt_val) in enumerate(weights.items()):
                                f.write('  [%d/%d %s]  *WEIGHT* %d/%d %s {%s} [shape=%s]. %s\n'  % (
                                    i+1, len(layer_names), layer_name,
                                    j+1, len(weights.keys()), wgt_name, hasher(wgt_val),
                                    str(wgt_val.shape), prec.upper()  ))
                                f.write(get_tensor_str(wgt_val) + '\n\n')

                f.write('\n')
    a = 1



def compare_model_activations(model_gtc, model_pub=None, activations_file=None,
                              net_type=None, config=None, img_raw=None,
                              gtc_name_func=None, gtc_value_func=None,
                              fast=False):

    preprocess_func = get_preprocessor(net_type)
    if img_raw is None:
        if isinstance(net_type, net_typ.MobilenetType) and net_type.style == 'tensorflow':
            preprocess_func = lambda x: (x.astype(np.float32) / 128.0) - 1.0
        imsize = (224, 224)
        im_filename = 'panda_224.jpg'
        img_raw = np.array(PIL.Image.open(im_filename).resize(imsize))

        if isinstance(net_type, net_typ.SSDNetType):
            imsize = (300, 300)
            img_raw = get_sample_image('panda', imsize)

        # Get test image.

    if img_raw.ndim < 4:
        img_raw = np.expand_dims(img_raw, axis=0)

    img = preprocess_func(img_raw)  #  img_raw.astype(np.float32) / 128.0 - 1.0

    placeholder = model_gtc.get_placeholder()
    if config.include_preprocessing:
        data_dict_sample = {placeholder: img_raw}
    else:
        data_dict_sample = {placeholder: img}


    net_name = net_type.get_name(version_only=True)
    layers_chk_use = act_names.get_activation_layer_names(net_type, config)
    contains_any = lambda x, strs: any([str in x for str in strs])

    if fast:
        layers_chk_use = layers_chk_use[:5] + layers_chk_use[-5:] # first 5 and last 5

    skip_n = 2

    n_layers = len(layers_chk_use)

    have_model_pub = model_pub is not None
    have_activations_file = activations_file is not None
    assert (have_model_pub != have_activations_file), "Please specify either model_pub or activations_file"

    print('Now comparing all hp-layers activations in GTC model with pretrained model.')
    utl.cprint('Layers that match (hopefully all) will be displayed in GREEN', color=utl.Fore.GREEN)
    utl.cprint('Layers that do not match (hopefully none) will be displayed in RED\n', color=utl.Fore.RED)
    n_matched, n_checked = 0, 0

    if have_activations_file:
        # e.g. Tensorflow checkpoints: We compare real-time outputs from the gtcModel
        # with the saved activations from the tensorflow checkpoint
        S_tf = hkl.load(activations_file)

        for layer_idx, layer_chk in enumerate(layers_chk_use):
            gtc_layer_name, tf_layer_name = layer_chk
            if tf_layer_name not in S_tf:
                if gtc_layer_name in S_tf:
                    tf_layer_name = gtc_layer_name
                else:
                    utl.cprint('No tf layer %s' % tf_layer_name, color=utl.Fore.RED)
                    continue

            if gtc_name_func is not None:
                gtc_layer_name = gtc_name_func(gtc_layer_name)
                if gtc_layer_name is None:
                    continue

            name_proposal1, name_proposal2 = '', ''
            if gtc_layer_name not in model_gtc._forward_prop_outputs:
                prefix = gtc_layer_name.split('/')[0] + '/'
                name_proposal1 = gtc_layer_name.replace(prefix, model_gtc._name )
                name_proposal2 = model_gtc._name + gtc_layer_name
                if name_proposal1 in model_gtc._forward_prop_outputs:
                    gtc_layer_name = name_proposal1
                elif name_proposal2 in model_gtc._forward_prop_outputs:
                    gtc_layer_name = name_proposal2


            if gtc_layer_name not in model_gtc._forward_prop_outputs:
                utl.cprint('No gtc layer %s (or %s or %s)' % (gtc_layer_name, name_proposal1, name_proposal2), color=utl.Fore.RED)
                continue

            layer_out_pub = S_tf[tf_layer_name]
            layer_out_gtc = model_gtc._forward_prop_outputs[gtc_layer_name]['hp'].eval(data_dict_sample)

            if gtc_value_func is not None:
                layer_out_gtc = gtc_value_func(gtc_layer_name, layer_out_gtc)

            elif contains_any(gtc_layer_name, ['Logits', 'predictions', 'fc1000']) and not 'SSD' in gtc_layer_name:
                layer_out_gtc = tf.nn.softmax(layer_out_gtc).eval()  # keras version has softmax built-in to logits

            if layer_out_gtc.shape != layer_out_pub.shape:
                layer_out_pub = np.squeeze(layer_out_pub)
                layer_out_gtc = np.squeeze(layer_out_gtc)

            if layer_out_gtc.shape != layer_out_pub.shape:
                utl.cprint('Shape mismatch %s (%s) vs %s(%s)' % (
                    gtc_layer_name, str(layer_out_gtc.shape), tf_layer_name, str(layer_out_pub.shape)),
                           color=utl.Fore.RED)
                continue



            mean_diff = np.mean(np.abs(layer_out_pub - layer_out_gtc)).item()
            max_diff = np.max(np.abs(layer_out_pub - layer_out_gtc)).item()
            do_match = np.allclose(layer_out_pub, layer_out_gtc, atol=1e-3)
            color_use = Fore.GREEN if do_match else Fore.RED
            n_checked += 1
            n_matched += int(do_match)
            if not do_match:
                debug = False
                if debug:
                    utl.cprint('(%d/%d) ("%s"/"%s") equal = %s [mean diff = %g]. [max_diff= %g]' % (
                        layer_idx + 1, n_layers, gtc_layer_name, tf_layer_name, do_match, mean_diff, max_diff),
                               color=color_use)

                    gtc_layer_name_prev, tf_layer_name_prev = layers_chk_use[layer_idx-1]
                    layer_out_prev = S_tf[tf_layer_name_prev]
                    wgts = model_gtc._layers_objects[gtc_layer_name]['layer_obj'].get_weights()
                    wgts0 = wgts[0]




            utl.cprint('(%d/%d) ("%s"/"%s") equal = %s [mean diff = %g]. [max_diff= %g]' % (
                layer_idx + 1, n_layers, gtc_layer_name, tf_layer_name, do_match, mean_diff, max_diff), color=color_use)

        print('Completed comparison with model saved activations model.')

    elif have_model_pub:
        assert config.net_style in ['keras.applications', 'image-classifiers']
        # Keras.applications reference model:
        # We compare real-time outputs from the two models

        if model_pub is None:
            model_pub = get_keras_model(net_type, pretrained=True)

        # gtc_precision = 'lp'
        gtc_precision = 'hp'
        model_name = model_gtc._name
        model_default_name = net_type.get_name(version_only=True).title().replace('_', '') + '/'

        for layer_idx, layer_chk in enumerate(layers_chk_use):
            gtc_layer_name_raw, keras_layer_name = layer_chk
            try:
                model_pub_partial_i = keras.models.Model(inputs=model_pub.input,
                                                         outputs=model_pub.get_layer(keras_layer_name).output)
            except:
                utl.cprint('error with keras layer %s' % keras_layer_name)
                continue

            if model_default_name in gtc_layer_name_raw:
                gtc_layer_name = gtc_layer_name_raw.replace(model_default_name, model_name)
            else:
                gtc_layer_name = model_name + gtc_layer_name_raw
            layer_out_pub = model_pub_partial_i.predict(img)

            try:
                layer_out_gtc = model_gtc._forward_prop_outputs[gtc_layer_name][gtc_precision].eval(data_dict_sample)
            except:
                utl.cprint('error with gtc layer %s' % gtc_layer_name)
                continue

            if layer_out_gtc.shape != layer_out_pub.shape:
                layer_out_pub = np.squeeze(layer_out_pub)
                layer_out_gtc = np.squeeze(layer_out_gtc)

            if layer_out_gtc.shape != layer_out_pub.shape:
                utl.cprint('Shape mismatch %s (%s) vs %s(%s)' % (
                gtc_layer_name, str(layer_out_gtc.shape), keras_layer_name, str(layer_out_pub.shape)), color=utl.Fore.RED)
                continue
            if contains_any(gtc_layer_name, ['Logits', 'predictions', 'fc1000']) and\
                    keras_layer_name in ['predictions', 'fc1000', 'Logits']:
                layer_out_gtc = tf.nn.softmax(layer_out_gtc).eval()  # keras version has softmax built-in to logits

            mean_diff = np.mean(np.abs(layer_out_pub - layer_out_gtc)).item()
            max_diff = np.max(np.abs(layer_out_pub - layer_out_gtc)).item()
            do_match = np.allclose(layer_out_pub, layer_out_gtc, atol=1e-3)
            color_use = Fore.GREEN if do_match else Fore.RED
            n_checked += 1
            n_matched += int(do_match)
            if not do_match:
                aaa = 1
            utl.cprint('(%d/%d) ("%s"/"%s") equal = %s [mean diff = %g]. [max_diff= %g]' % (
                layer_idx + 1, n_layers, gtc_layer_name, keras_layer_name, do_match, mean_diff, max_diff), color=color_use)

        print('Completed comparison with model.')

    print('')
    if n_matched == n_checked:
        utl.cprint('All %d layers matched!!' % n_matched , color=utl.Fore.GREEN)
    else:
        utl.cprint('%d layers matched, but %d layers did not match!!' % (n_matched, n_checked-n_matched), color=utl.Fore.RED)




def eval_classification_model_on_dataset(model, dataLoader, hp=True, lp=True,
                                         keras_model=None,
                                         gtc_preprocessing_function=None,
                                         keras_preprocessing_function=None,
                                         label_offset=0,
                                         loader_args=None):
    if loader_args is None:
        loader_args = {}

    batch_size = loader_args.get('batch_size', 32)
    loader_args['batch_size'] = batch_size


    do_hp_test = hp
    do_lp_test = lp
    do_pure_keras_model_test = keras_model is not None

    model_is_gtcModel = isinstance(model, gtc.GTCModel)
    model_is_pureKerasModel = isinstance(model, keras.Model) and not hasattr(model, 'gtcModel')
    model_is_gtcKerasModel = isinstance(model, keras.Model) and hasattr(model, 'gtcModel')
    assert model_is_gtcModel or model_is_pureKerasModel or model_is_gtcKerasModel

    if model_is_pureKerasModel:
        # if a pure keras model, then there is obviously has one precision type.
        # However, I use the hp/lp flags to indicate whether to print out and store
        # the results as either coming from a HP network or an LP network
        do_lp_test = not do_hp_test  # set HP to True for an HP network, otherwise we assume it is an LP network

    top_k = 5
    if loader_args is None:
        loader_args = {}
    all_preds_hp_list, all_preds_lp_list, all_preds_pub_list, all_gt_list = [], [], [], []
    count = 0

    try:
        dataset_it = dataLoader.flow(**loader_args)
        nBatches = len(dataLoader) // batch_size
    except:
        dataset_it = iter(dataLoader)
        nBatches = len(dataLoader)

    for data_j in tqdm(dataset_it, total=nBatches):
        data_im_raw, data_labels = data_j
        data_im_use_gtc = data_im_raw
        if isinstance(data_labels, list):
            data_labels = data_labels[0]

        if gtc_preprocessing_function is not None:
            data_im_use_gtc = gtc_preprocessing_function(data_im_use_gtc)

        if data_labels.ndim > 1 and data_labels.shape[1] > 1: # sparse categorical data
            data_labels = np.argmax(data_labels, axis=1)

        all_gt_list.append(data_labels)

        preds_hp, preds_lp = None, None

        if model_is_pureKerasModel:
            softmax_vals = model.predict(data_im_use_gtc)
            preds = tf.nn.top_k(softmax_vals, k=top_k).indices.eval()
            if do_hp_test:
                preds_hp = preds
            else:
                preds_lp = preds

        elif model_is_gtcKerasModel:
            keras_preds = model.predict(data_im_use_gtc)
            softmax_hp, softmax_lp = keras_preds[0], keras_preds[1]
            preds_hp = tf.nn.top_k(softmax_hp, k=top_k).indices.eval()
            preds_lp = tf.nn.top_k(softmax_lp, k=top_k).indices.eval()

        else:
            assert model_is_gtcModel
            if do_hp_test:
                preds_hp = model.predict_class(data_im_use_gtc, hp=True, top_k=top_k)

            if do_lp_test:
                preds_lp = model.predict_class(data_im_use_gtc, hp=False, top_k=top_k)

        all_preds_hp_list.append(np.squeeze(preds_hp))
        all_preds_lp_list.append(np.squeeze(preds_lp))

        if do_pure_keras_model_test:
            if keras_preprocessing_function is not None:
                data_im_preprocessed = keras_preprocessing_function(data_im_raw)
            else:
                data_im_preprocessed = data_im_raw
            preds_pub_softmax = keras_model.predict(data_im_preprocessed)
            preds_pub_top_k = tf.nn.top_k(preds_pub_softmax, k=top_k).indices.eval()
            all_preds_pub_list.append(np.squeeze(preds_pub_top_k))

        count += 1
    a = 1

    all_gt = np.concatenate(all_gt_list)
    all_gt = all_gt.reshape((len(all_gt), 1))
    n = len(all_gt)
    print('')

    def acc(preds, gt):
        acc_top1 = np.count_nonzero(preds[:, :1] == gt + label_offset) / n * 100
        acc_topk = np.count_nonzero(np.any(preds == gt + label_offset, axis=1)) / n * 100
        return acc_top1, acc_topk

    out_dict = {}
    if do_hp_test:
        all_preds_hp = np.concatenate(all_preds_hp_list)

        hp_accuracy_top_1, hp_accuracy_top_k = acc(all_preds_hp, all_gt)
        print(' --- HP accuracy          : top-1 : %.3f.  top-%d : %.3f' % (
            hp_accuracy_top_1, top_k, hp_accuracy_top_k))

        out_dict['hp_accuracy_top_1'] = hp_accuracy_top_1
        out_dict['hp_accuracy_top_k'] = hp_accuracy_top_k

    if do_lp_test:
        all_preds_lp = np.concatenate(all_preds_lp_list)
        lp_accuracy_top_1, lp_accuracy_top_k = acc(all_preds_lp, all_gt)

        print(' --- LP accuracy          : top-1 : %.3f.  top-%d : %.3f' % (
            lp_accuracy_top_1, top_k, lp_accuracy_top_k))

        out_dict['lp_accuracy_top_1'] = lp_accuracy_top_1
        out_dict['lp_accuracy_top_k'] = lp_accuracy_top_k

    if do_pure_keras_model_test:
        all_preds_pub = np.concatenate(all_preds_pub_list)
        pub_accuracy_top_1, pub_accuracy_top_k = acc(all_preds_pub, all_gt)
        print(' --- Keras model accuracy : top-1 : %.3f.  top-%d : %.3f' % (
            pub_accuracy_top_1, top_k, pub_accuracy_top_k))

        out_dict['pub_accuracy_top_1'] = pub_accuracy_top_1
        out_dict['pub_accuracy_top_1'] = pub_accuracy_top_1

    return out_dict


def get_keras_model(network_type, pretrained=True, input_shape=(224,224,3), classes=1000):

    net_str = network_type.get_name(version_only=True)
    if network_type.style == 'keras.applications':

        model = get_keras_applications_network(network_type, pretrained, input_shape, classes)

    else: # use image_classifiers library
        assert network_type.style == 'image-classifiers'
        model_builder, preprocess_input = \
            classification_models.keras.Classifiers.get(net_str)

        weights = 'imagenet' if pretrained else None
        model = model_builder(input_shape=input_shape, weights=weights, classes=classes)

    return model


def get_keras_applications_network(network_type, pretrained=True, input_shape=(224,224,3), classes=1000):
    assert isinstance(network_type, net_typ.NetworkType)
    if network_type.name == 'mobilenet':
        assert isinstance(network_type, net_typ.MobilenetType)
        if network_type.version == 1:
            model = keras.applications.MobileNet
        elif network_type.version == 2:
            model = keras.applications.MobileNetV2
        else:
            raise ValueError('Invalid Mobilenet version %d' % network_type.version)

    elif network_type.name == 'vgg':
        if network_type.num_layers == 16:
            model = keras.applications.VGG16
        elif network_type.num_layers == 19:
            model = keras.applications.VGG19
        else:
            raise ValueError('Invalid number of layers for VGG : %d' %
                             network_type.num_layers)

    elif network_type.name == 'resnet':
        if network_type.num_layers == 50:
            model = keras.applications.ResNet50
        elif network_type.num_layers == 101:
            model = keras.applications.ResNet101
        elif network_type.num_layers == 152:
            model = keras.applications.ResNet152
        else:
            raise ValueError('Invalid number of layer for ResNet : %d' %
                             network_type.num_layers)

    else:
        raise ValueError('Invalid network name : %s' % network_type.name)

    weights = 'imagenet' if pretrained else None
    keras_model = model(input_shape=input_shape, weights=weights, classes=classes)
    return keras_model

def eval_detection_model_on_dataset(model, dataLoader, hp=True, lp=True, keras_model=None, keras_model_preprocess=None, label_offset=0):
    nBatches = len(dataLoader)

    do_hp_test = hp
    do_lp_test = lp
    do_keras_model_test = keras_model is not None

    model_is_gtcKeras = isinstance(model, gtc.GTCModelKerasLayer)

    top_k = 5

    all_preds_hp_list, all_preds_lp_list, all_preds_pub_list, all_gt_list = [], [], [], []
    count = 0
    for data_j in tqdm(dataLoader):
        data_im_raw, data_labels = data_j
        data_im_use_gtc = data_im_raw

        all_gt_list.append(data_labels)

        preds_hp, preds_lp = None, None
        if model_is_gtcKeras:
            keras_preds = model.predict(data_im_use_gtc)
            logits_hp, logits_lp = keras_preds[0], keras_preds[1]
            preds_hp = tf.nn.top_k(logits_hp, k=top_k).indices.eval()
            preds_lp = tf.nn.top_k(logits_hp, k=top_k).indices.eval()

        else:
            if do_hp_test:
                preds_hp = model.predict_class(data_im_use_gtc, hp=True, top_k=top_k)

            if do_lp_test:
                preds_lp = model.predict_class(data_im_use_gtc, hp=False, top_k=top_k)

        all_preds_hp_list.append(np.squeeze(preds_hp))
        all_preds_lp_list.append(np.squeeze(preds_lp))

        if do_keras_model_test:
            data_im_preprocessed = keras_model_preprocess(data_im_raw)
            preds_pub_softmax = keras_model.predict(data_im_preprocessed)
            preds_pub_top_k = tf.nn.top_k(preds_pub_softmax, k=top_k).indices.eval()
            all_preds_pub_list.append(np.squeeze(preds_pub_top_k))

        count += 1
    a = 1

    all_gt = np.concatenate(all_gt_list)
    all_gt = all_gt.reshape((len(all_gt), 1))
    n = len(all_gt)
    print('')

    def acc(preds, gt):
        acc_top1 = np.count_nonzero(preds[:, :1] == gt + label_offset) / n * 100
        acc_topk = np.count_nonzero(np.any(preds == gt + label_offset, axis=1)) / n * 100
        return acc_top1, acc_topk

    out_dict = {}
    if do_hp_test:
        all_preds_hp = np.concatenate(all_preds_hp_list)

        hp_accuracy_top_1, hp_accuracy_top_k = acc(all_preds_hp, all_gt)
        print(' --- HP accuracy          : top-1 : %.3f.  top-%d : %.3f' % (
            hp_accuracy_top_1, top_k, hp_accuracy_top_k))

        out_dict['hp_accuracy_top_1'] = hp_accuracy_top_1
        out_dict['hp_accuracy_top_k'] = hp_accuracy_top_k

    if do_lp_test:
        all_preds_lp = np.concatenate(all_preds_lp_list)
        lp_accuracy_top_1, lp_accuracy_top_k = acc(all_preds_lp, all_gt)

        print(' --- LP accuracy          : top-1 : %.3f.  top-%d : %.3f' % (
            lp_accuracy_top_1, top_k, lp_accuracy_top_k))

        out_dict['lp_accuracy_top_1'] = lp_accuracy_top_1
        out_dict['lp_accuracy_top_k'] = lp_accuracy_top_k

    if do_keras_model_test:
        all_preds_pub = np.concatenate(all_preds_pub_list)
        pub_accuracy_top_1, pub_accuracy_top_k = acc(all_preds_pub, all_gt)
        print(' --- Keras model accuracy : top-1 : %.3f.  top-%d : %.3f' % (
            pub_accuracy_top_1, top_k, pub_accuracy_top_k))

        out_dict['pub_accuracy_top_1'] = pub_accuracy_top_1
        out_dict['pub_accuracy_top_1'] = pub_accuracy_top_1

    return out_dict


def get_model_path(model_id):
    # model_id should be of the form "[model_name]/[date]
    # In the 'models_root' folder, I keep copies of important model checkpoints
    # in a sub-folder called 'saved'
    if model_id.endswith('.h5'):
        model_name, model_date, base_name = model_id.split('/')
        # add default checkpoint filename:
    else:
        model_name, model_date = model_id.split('/')
        base_name = model_date + '_checkpoint.h5'

    model_ckpt_file = utl.models_root() + '/saved/' + os.path.join(model_name, model_date, base_name)

    if not os.path.exists(model_ckpt_file):
        raise ValueError('Model path %s does not exist' % model_ckpt_file)
    return model_ckpt_file




def acc(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true, y_pred) * 100




def copy_keras_model(keras_model, quantizer_set=None):
    model = gtc.GTCModel()
    #if quantizer_set is None:

    for lyr in keras_model.layers:
        c = lyr.get_config()
        if isinstance(lyr, krs.Conv2D ):
            model.add( gtc.Conv2D.from_config(c) )
        elif isinstance(lyr, krs.BatchNormalization ):
            model.add( gtc.BatchNormalization.from_config(c) )
        elif isinstance(lyr, krs.Activation):
            model.add( gtc.Activation.from_config(c))
        elif isinstance(lyr, krs.Dense):
            model.add( gtc.Dense.from_config(c))
        elif isinstance(lyr, krs.Dropout):
            model.add( gtc.Dropout.from_config(c))
        #elif isinstance(lyr, (krs.Add, krs.Subtract, krs.Multiply, krs.Maximum, krs.Minimum, krs.Concatenate, krs.Dot ):
        #    model.add( gtc.Merge.from_config(c))

    return model


def readTrainingLogFile(training_log_file):
    # Reads a log file output by the TrainingHistoryLogger callback (above)
    # Returns an OrderedDict with
    #   key=header of each column, value=all data in the corresponding column

    training_log_file = training_log_file.replace('//', '/')
    print('Reading %s' % training_log_file)

    with open(training_log_file, 'r') as f:
        all_lines = f.read().splitlines()

    data = np.loadtxt(training_log_file, skiprows=1)

    header_line = all_lines[0]
    all_headers = header_line.split()
    log_data = OrderedDict([(k, data[:,i]) for i,k in enumerate(all_headers)])

    epoch_ids = log_data['epoch']
    val_epoch_ids = None
    for k in log_data:
        if k.startswith('VAL_'):
            v = log_data[k]
            idx_new_vals = np.nonzero( np.diff(v) != 0 )[0] + 1
            if val_epoch_ids is not None:
                assert np.array_equal(val_epoch_ids, epoch_ids[idx_new_vals])
            else:
                val_epoch_ids = epoch_ids[idx_new_vals]
            v_keep = v[idx_new_vals]
            log_data[k] = v_keep

    log_data['VAL_epoch'] = val_epoch_ids
    return log_data


def num_gpus_available():
    if utl.onLaptop():
        return 1

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len(visible_devices) == 0:
        return 0
    visible_devices_list = visible_devices.split(',')
    return len(visible_devices_list)


def print_training_history(history, keys_print, skip_missing=True):
    # warning: untested
    do_fancy = False
    if do_fancy:
        h_dict = history.history
        h_dict_keys = list(h_dict.keys())
        epoch_ids = history.epoch

        heading_keys  = []
        vals_templates = []
        for k in keys_print:
            k_matches = []
            if k in h_dict:
                k_matches = [k]
            elif '*' in k:
                k_pattern = k.replace('*', '\\w*')
                k_matches = [ kk for kk in h_dict_keys if re.search(k_pattern, kk) is not None]

            for k_match in k_matches:
                heading_keys.append(k_match)
                vals_templates.append('%' + str(len(k_match)+3) + '.3f')

        heading_str   = 'Epoch' + ' '.join(heading_keys)
        vals_template = ' %3d ' + ' '.join(vals_templates)

        #print(" Epoch.  Training Loss. Validation Loss")
        print(heading_str)
        for i, epoch_id in enumerate(epoch_ids):
            vals = [epoch_id] +  [h_dict[k][i] for k in heading_keys]
            str_i = vals_template % vals
            print(str_i)


# Copied from keras losses.py from version 2.3.0.  Duplicated here to replace
# version from 2.2.5, which doesn't support the label_smoothing
def categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0.0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
    return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)


def smoothed_crossentropy(label_smoothing=0.0):
    def categorical_crossentropy_smoothed(y_true, y_pred):
        return categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)

    return categorical_crossentropy_smoothed



# Modification of keras model save_weights() method.
# Includes code to retry a few times on failure
def keras_save_weights_robust(self, filepath, overwrite=True):
    """Dumps all layer weights to a HDF5 file.

    The weight file has:
        - `layer_names` (attribute), a list of strings
            (ordered names of model layers).
        - For every layer, a `group` named `layer.name`
            - For every such layer group, a group attribute `weight_names`,
                a list of strings
                (ordered names of weights tensor of the layer).
            - For every weight in the layer, a dataset
                storing the weight value, named after the weight tensor.

    # Arguments
        filepath: String, path to the file to save the weights to.
        overwrite: Whether to silently overwrite any existing file at the
            target location, or provide the user with a manual prompt.

    # Raises
        ImportError: If h5py is not available.
    """

    # Sometimes the file creation fails, breaking the training loop.
    # To make it more robust, we allow a few tries before failing
    #  'seconds_wait_between_attempts'  doubles after each save attempt.
    num_save_attempts = 10
    seconds_wait_between_attempts = 1

    for attempt_id in range(num_save_attempts):

        save_succeeded = False
        try:

            with h5py.File(filepath, 'w') as f:
                keras.engine.saving.save_weights_to_hdf5_group(f, self.layers)
                f.flush()

            save_succeeded = True

        except OSError as err:
            if attempt_id > 0:
                print("OS error: {0}".format(err))

        except Exception as err:
            if attempt_id > 0:
                print("Unexpected error: %s" % str(err))
                # print("Unknown error : %s :", sys.exc_info()[0])

        if save_succeeded:
            break

        else:
            if attempt_id > 0:
                print('Encountered error during save attempt %d / %d. Trying again in %d seconds... ' % (
                    attempt_id + 1, num_save_attempts, seconds_wait_between_attempts))
            time.sleep(seconds_wait_between_attempts)
            seconds_wait_between_attempts *= 2




def print_torchmodel(torch_filename):
    resnet_tfile = utl.pretrained_weights_root() + 'resnet-18.t7'
    import hickle as hkl
    import torchfile

    resnet18_torch_variables = OrderedDict([
        ('conv0', (7, 7, 3, 64)),
        ('bn0', (64,)),

        ('stage1_unit1_conv1', (3, 3, 64, 64)),
        ('stage1_unit1_bn1', (64,)),
        ('stage1_unit1_conv2', (3, 3, 64, 64)),
        ('stage1_unit1_bn2', (64,)),
        ('stage1_unit2_conv1', (3, 3, 64, 64)),
        ('stage1_unit2_bn1', (64,)),
        ('stage1_unit2_conv2', (3, 3, 64, 64)),
        ('stage1_unit2_bn2', (64,)),

        ('stage2_unit1_conv1', (3, 3, 64, 128)),
        ('stage2_unit1_bn1', (128,)),
        ('stage2_unit1_conv2', (3, 3, 128, 128)),
        ('stage2_unit1_bn2', (128,)),
        ('stage2_unit1_sc', (1, 1, 64, 128)),
        ('stage2_unit2_conv1', (3, 3, 128, 128)),
        ('stage2_unit2_bn1', (128,)),
        ('stage2_unit2_conv2', (3, 3, 128, 128)),
        ('stage2_unit2_bn2', (128,)),

        ('stage3_unit1_conv1', (3, 3, 128, 256)),
        ('stage3_unit1_bn1', (256,)),
        ('stage3_unit1_conv2', (3, 3, 256, 256)),
        ('stage3_unit1_bn2', (256,)),
        ('stage3_unit1_sc', (1, 1, 128, 256)),
        ('stage3_unit2_conv1', (3, 3, 256, 256)),
        ('stage3_unit2_bn1', (256,)),
        ('stage3_unit2_conv2', (3, 3, 256, 256)),
        ('stage3_unit2_bn2', (256,)),

        ('stage4_unit1_conv1', (3, 3, 256, 512)),
        ('stage4_unit1_bn1', (512,)),
        ('stage4_unit1_conv2', (3, 3, 512, 512)),
        ('stage4_unit1_bn2', (512,)),
        ('stage4_unit1_sc', (1, 1, 256, 512)),
        ('stage4_unit2_conv1', (3, 3, 512, 512)),
        ('stage4_unit2_bn1', (512,)),
        ('stage4_unit2_conv2', (3, 3, 512, 512)),
        ('stage4_unit2_bn2', (512,)),

        ('Logits', (512, 1000))])

    var_idx = 0
    variable_shapes = resnet18_torch_variables
    keys = list(variable_shapes.keys())

    all_tensors = OrderedDict()

    def print_torch_module(m, base_str=''):
        torch_name = m.torch_typename()
        global var_idx
        is_conv = b'Convolution' in torch_name
        is_bn = b'BatchNormalization' in torch_name
        is_lin = b'Linear' in torch_name
        if is_conv or is_bn or is_lin:
            var_name = keys[var_idx]
            var_idx += 1
            expected_shape = variable_shapes[var_name]
            if is_conv:
                assert 'conv' in var_name or 'sc' in var_name, "Torchname = %s. Expected conv in name %s" % (torch_name, var_name)
            if is_bn:
                assert 'bn' in var_name, "Torchname = %s. Expected bn in name %s" % (torch_name, var_name)
            if is_lin:
                assert 'Logits' in var_name, "Torchname = %s. Expected Logits in name %s" % (torch_name, var_name)

            main_attrs = ['weight', 'bias', 'running_mean', 'running_var',
                         ]
            extra_attrs = ['momentum', 'eps',
                         'padW', 'padH', 'scaleT', 'kW', 'kH', 'dW', 'dH', 'ceil_mode' ]

            out_str = base_str
            for param in main_attrs:
                if hasattr(m, param) and getattr(m, param) is not None:
                    out_str += '  %s: %s' % (param, str( getattr(m, param).shape) )
                    x = getattr(m, param).transpose()

                    expected_shape_i = expected_shape
                    if 'bias' in param and (is_conv or is_lin):
                        expected_shape_i = (expected_shape[-1],)
                    print( " %s / %s (%s). Expected %s. Actual: %s" % (var_name, param, torch_name, str(expected_shape_i), str(x.shape) ) )
                    assert x.shape == expected_shape_i
                    all_tensors[var_name + '/' + param] = x

            print(out_str)


        if hasattr(m, 'modules') and m.modules is not None:

            for j, m_sub_i in enumerate(m.modules):
                base_str_j = base_str + '   --> %d : %s: ' % (j+1, m_sub_i.torch_typename() )
                print_torch_module(m_sub_i, base_str_j)




                        #if m.bias is not None:
            #    out_str += '  bias: %s' % str(m.bias.shape)
            #if m.running_mean is not None:
            #    out_str += '  running_mean: %s' % str(m.running_mean.shape)
            #if m.running_var is not None:
            #    out_str += '  running_var: %s' % str(m.running_var.shape)


    #print(model)
    dst_file = 'resnet-18-torch.hkl'
    hkl.dump( all_tensors, dst_file )
    model = torchfile.load(resnet_tfile, force_8bytes_long=True)




