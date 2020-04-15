# Core utilities needed by both GTC models and networks.
import tensorflow.compat.v1 as tf
import keras
import warnings
import numpy as np
import re
from collections import OrderedDict

from networks.net_types import *
from utility_scripts import misc_utl as utl

######### 1. Lambda Weights ############

class LambdaWeights():
    def __init__(self, hp=1, lp=None, distillation_loss=None, bit_loss=None, bits_per_layer=None):
        self.hp = hp
        self.lp = lp
        self.distillation_loss = distillation_loss
        self.bit_loss = bit_loss
        if bits_per_layer is None and bit_loss is not None:
            bits_per_layer = 0
        self.bits_per_layer = bits_per_layer
        self.keys = ['hp', 'lp', 'distillation_loss', 'bit_loss', 'bits_per_layer']

    def need_lp_network(self):
        return self.lp is not None or \
               self.distillation_loss is not None or \
               self.bit_loss is not None

    def __str__(self):

        wgt_strs = []
        hp_default = 1.0
        lp_default = 0.0
        distill_default = 0.0
        bit_default = 0.0

        if self.hp != hp_default:
            wgt_strs.append('Hp_%g' % self.hp)

        if self.lp is not None and self.lp != lp_default:
            wgt_strs.append('Lp_%s' % utl.get_small_num_str(self.lp))

        if self.distillation_loss is not None and self.distillation_loss != distill_default:
            wgt_strs.append('Ds_%s' % utl.get_small_num_str(self.distillation_loss))

        if self.bit_loss is not None and self.bit_loss != bit_default:
            wgt_strs.append('Bt_%s' % utl.get_small_num_str(self.bit_loss))

        weight_str = '-'.join(wgt_strs)

        if len(weight_str) > 0:
            weight_str = 'Lam' + weight_str

        return weight_str

    def get_dict(self):
        d = OrderedDict( [('hp', self.hp), ('lp', self.lp),
                          ('distillation_loss', self.distillation_loss),
                          ('bit_loss', self.bit_loss),
                          ('bits_per_layer', self.bits_per_layer)])
        return d

    def get_keys(self):
        return self.keys

    def copy(self):
        return LambdaWeights(self.hp, self.lp,
                             self.distillation_loss,
                             self.bit_loss,
                             self.bits_per_layer)
    def __copy__(self):
        return self.copy()

    def get_abbrev(self):
        abbrev = dict(hp='hp',
                      lp='lp',
                      distillation_loss='distil',
                      bit_loss='bit',
                      bits_per_layer='bits_per_layer')
        return abbrev
        #loss_weights_dict = {abbrev[k]:v for k,v in loss_weights_dict0.items()}


    def AugmentOutput(self, y):
        remove_none_losses = True
        keys = self.get_keys()
        nTot = y.shape[0]
        y_augmented = []
        for k in keys:
            v = getattr(self, k)
            if v is not None or not remove_none_losses:
                if k in ['hp', 'lp']:
                    y_augmented.append(y)
                elif k in ['distillation_loss', 'bit_loss', 'bits_per_layer']:
                    y_augmented.append(np.zeros(nTot))
                else:
                    raise ValueError('invalid key : %s' % k)

        return y_augmented

    def AugmentLossesAndMetrics(self, hp_lp_loss=None, hp_lp_metric=None,
                                distil_loss='ident', distil_metric=None,
                                bit_loss='ident', bit_metric='ident'):
        remove_none_losses = True
        keys = self.get_keys()
        abbrev = self.get_abbrev()
        losses_augmented = OrderedDict()
        metrics_augmented = OrderedDict()
        loss_weights = OrderedDict()

        for k in keys:
            v = getattr(self, k)
            if v is not None or not remove_none_losses:
                k_abbrev = abbrev[k]

                loss_weights[k_abbrev] = v
                if k in ['hp', 'lp']:
                    # For HP and LP, we want the losses & accuracy metrics.
                    if hp_lp_loss is not None:
                        losses_augmented[k_abbrev] = hp_lp_loss
                    if hp_lp_metric is not None:
                        metrics_augmented[k_abbrev] = hp_lp_metric
                elif k in ['distillation_loss']:
                    if distil_loss == 'ident':
                        distil_loss = ident_sum
                    losses_augmented[k_abbrev] = distil_loss

                    # For distillation loss, we can already see the value in the loss
                    # function (distil_loss), so we don't need yet another copy
                    # in the metrics. So should have a distil_metric
                    assert distil_metric is None, "Dont need a distillation metric"

                elif k in ['bit_loss']:
                    # For bit loss, we can already see the value in the loss
                    # function, so we don't need yet another copy in the metrics:
                    if bit_loss == 'ident':
                        bit_loss = ident_sum

                    #bit_losses.append(bit_loss)
                    losses_augmented[k_abbrev] = bit_loss


                elif k in ['bits_per_layer']:
                    # A useful bit metric is the average num bits per layer:
                    # (this is actually a distinct output from the bit-loss
                    # in the the gtcKerasModel)
                    if bit_metric is not None:
                        # Ideally, this would be a simple metric that is included
                        # in the keras metrics. But keras seems to ignore metrics that are not
                        # included in the loss function as well.
                        # So we instead put this bits-per-layer metric in the loss function
                        # (but assign it zero weight so it doesn't actually affect the fit)
                        # It then gets logged and displayed, but doesn't affect the loss
                        # (as if it were a metric)
                        losses_augmented[k_abbrev] = ident_sum
                        bits_per_layer_val = self.bits_per_layer
                        if isinstance(self.bits_per_layer, tf.Variable):
                            bits_per_layer_val = keras.backend.get_value(self.bits_per_layer)
                        assert bits_per_layer_val == 0, "Should be zero"

                        # Keras will actually ignore this metric unless it also
                        # contributes to the loss function. We add a link to a

                else:
                    raise ValueError('invalid key : %s' % k)

        return losses_augmented, metrics_augmented, loss_weights



def ident(y_true, y_pred):
    # A dummy 'loss' function for the distillation loss and bit-loss
    # "y-true" is just the target of 0. The 'loss' is just the value of y-pred
    return y_pred

def ident_sum(y_true, y_pred):
    # A dummy 'loss' function for the distillation loss and bit-loss
    # "y-true" is just the target of zero(s). The 'loss' is just the value of y-pred
    return keras.backend.sum(y_pred)


class NetStylesType():
    def __init__(self, style):
        self.tensorflow = 1
        self.keras_applications = 2
        self.image_classifiers = 3

tensorflow_style = 1
keras_applications_style = 2
image_classifiers_style = 3

#Styles = NetStylesType()






def get_unique_tf_prefix(prefix):
    global_var_names = [x.name for x in tf.compat.v1.global_variables()]
    global_prefixes = set([name.split('/')[0] + '/' for name in global_var_names if '/' in name])

    orig_prefix = prefix
    if prefix in global_prefixes:
        id = 1
        while orig_prefix.replace('/', '_%d/' % id) in global_prefixes:
            id += 1
        prefix = orig_prefix.replace('/', '_%d/' % id)

    return prefix




def clean_type_str(x):

    s = str(x)
    # e.g.   "<dtype: 'float32'>"   ->    "float32"
    # e.g.   "<class 'gtc.dense.Dense'>" -> "gtc.dense.Dense"
    #s_clean = re.search("<[a-z:]* '(\w*)'>", s).groups()[0]
    # Note: I subsequently found that a much easier way to do this
    # is to simply use type(obj).__name__  or type(obj).name
    brackets_match = re.search("<[a-z:]* '([\w.]*)'>", s)
    if brackets_match:
        s_clean = brackets_match.groups()[0]
        return s_clean
    else:
        return s

    #re.search('<[a-z]*: '(\w*)' >', s)

# for help in serializing/deserializing gtc model definitions
def to_str(obj):

    params = None
    if isinstance(obj, (str, float, int, bool, tuple, list, dict)) or obj is None:
        return obj
    elif isinstance(obj, NetworkType):
        return str(obj)

    obj_type = type(obj)
    if hasattr(obj_type, '__name__'):
        s_type = obj_type.__name__
    elif hasattr(obj_type, 'name'):
        s_type = obj_type.name
    else:
        s_type = clean_type_str(str(obj_type))

    s_type_last = s_type
    if '.' in s_type:
        s_type_last = s_type.split('.')[-1]

    if s_type == 'TruncatedNormal':  # keras.initializers.TruncatedNormal
        params = dict(mean=obj.mean, stddev=obj.stddev, seed=obj.seed)
    elif s_type == 'L1L2':           # keras.regularizers.L1L2
        params = dict(l1=float(obj.l1), l2=float(obj.l2))
    elif s_type == 'SSDConfig':      # ie. 'ssd.ssd_utils.SSDConfig'
        params = obj.get_config()
    elif s_type == 'LambdaWeights':  # ie. 'networks.net_utils_core.LambdaWeights'
        params = obj.get_dict()
    elif s_type == 'ndarray':        # ie. numpy.ndarray
        params = dict(object=obj.tolist(), dtype=str(obj.dtype))
    else:
        raise ValueError('Unhandled object type : %s' % s_type)

    if params is None:
        return obj
    else:
        return ('obj', s_type, params)


def from_str(obj):

    if isinstance(obj, (tuple, list)) and len(obj) == 3 and obj[0] == 'obj':
        _, obj_name, params = obj


        obj_dict = {
            'TruncatedNormal': keras.initializers.TruncatedNormal,
            'L1L2': keras.regularizers.L1L2,
            # 'SSDConfig': SSDConfig,
            'LambdaWeights': LambdaWeights,
            'ndarray': np.array,

        }
        if '.' in obj_name:
            obj_name = obj_name.split('.')[-1]

        if obj_name == 'SSDConfig':
            from ssd.ssd_utils import SSDConfig
            obj_ref = SSDConfig
        else:
            obj_ref = obj_dict[obj_name]

        return obj_ref(**params)

    else:
        return obj


def stringify_model_specs(cfg_list):
    for cfg_spec in cfg_list:
        if isinstance(cfg_spec, tuple) and len(cfg_spec) == 2 and \
                isinstance(cfg_spec[1], dict) and 'config' in cfg_spec[1]:
            cfg = cfg_spec[1]['config']
            for k in cfg.keys():
                cfg[k] = to_str(cfg[k])

def destringify_model_specs(cfg_list):
    for cfg_spec in cfg_list:
        if isinstance(cfg_spec, (tuple, list)) and len(cfg_spec) == 2 and \
                isinstance(cfg_spec[1], dict) and 'config' in cfg_spec[1]:
            cfg = cfg_spec[1]['config']
            for k in cfg.keys():
                cfg[k] = from_str(cfg[k])


