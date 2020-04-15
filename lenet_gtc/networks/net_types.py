import keras
import tensorflow.compat.v1 as tf
import numpy as np
import re
from collections import OrderedDict
from utility_scripts import callbacks as net_callbacks

import utility_scripts.misc_utl as utl

if keras.__version__ == '2.3.0':
    from networks.optimizers import optimizers as opt
elif keras.__version__ == '2.2.5':
    from networks.optimizers import optimizers_225 as opt


######### 2. Network / Mobilenet Types  ############

default_imsize_class = 224
default_imsize_detection = 300

class NetworkType():
    def __init__(self, name, num_layers=None, style='keras.applications', imsize=224):
        self.name = name.lower()
        self.num_layers = num_layers
        self.set_style(style)
        self.imsize = imsize


    def set_style(self, style):
        if style in ['tensorflow', 'tf']:
            self.style = 'tensorflow'
            self.style_abbrev = 'tf'
        elif style in ['keras.applications', 'keras']:
            self.style = 'keras.applications'
            self.style_abbrev = 'keras'
        elif style in ['image-classifiers', 'ic']:
            self.style = 'image-classifiers'
            self.style_abbrev = 'ic'
        else:
            raise ValueError('Unrecongnized network style : %s' % style)


    def get_name(self, version_only=None, skip_size=False):
        # name = self.name + '-' + str(self.num_layers)
        name = self.name
        if self.num_layers is not None:
            name += str(self.num_layers)

        if not version_only:
            name += '-' + self.style_abbrev

        imsize_str = ''
        if self.imsize is not None and self.imsize != default_imsize_class and not skip_size:
            imsize_str = '-%d' % self.imsize
            name += imsize_str

        return name

    def __str__(self):
        return self.get_name()

    def copy(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


    @classmethod
    def from_str(cls, net_str):
        # acceptable formats: 'resnet18', 'resnet18_ic',
        # 'mobilenet_v1', 'mobilenet-v1', '

        #TODO: Support for depth-multiplier, ema, in network string,
        if net_str is None:
            return None

        style = 'keras'  # default style
        if 'tf' in net_str:
            style = 'tf'
        elif 'keras' in net_str:
            style = 'keras'
        elif 'ic' in net_str:
            style = 'ic'

        #contains_any = lambda x, strs: any([str in x for str in strs])
        net_str = net_str.lower()

        is_ssd_network = 'ssd' in net_str
        if is_ssd_network:
            default_imsize = default_imsize_detection
        else:
            default_imsize = default_imsize_class

        imsize = default_imsize  # default for SSD
        possible_sizes = [224, 300, 320]
        for sz in possible_sizes:
            if str(sz) in net_str:
                # imsize = (sz, sz)
                imsize = sz
                break

        imsize_base = None if is_ssd_network else imsize

        version = None
        if 'mobilenet' in net_str:
            version = None
            if 'v1' in net_str:
                version = 1
            elif 'v2' in net_str:
                version = 2
            elif 'v3' in net_str:
                version = 3

            net_size = None
            if 'large' in net_str.lower():
                net_size = 'Large'
            elif 'small' in net_str.lower():
                net_size = 'Small'

            depth_multiplier = None
            depth_multipliers_chk = ['0.25', '0.35', '0.5', '0.50', '0.75', '1.0', '1.3', '1.4']
            for depth_multiplier_str in depth_multipliers_chk:
                if depth_multiplier_str in net_str:
                    assert depth_multiplier is None
                    depth_multiplier = float(depth_multiplier_str)

            if depth_multiplier is None:
                depth_multiplier = 1.0


            ema = True
            if style == 'tf' and '_p' in net_str:
                ema = False

            net_type = MobilenetType(version=version, style=style,
                                     imsize=imsize_base, size=net_size, ema=ema,
                                     depth_multiplier=depth_multiplier)

        elif 'vgg' in net_str:
            num_layers_str = re.search('vgg(\d+)', net_str).groups()[0]
            num_layers = int(num_layers_str)

            net_type = NetworkType('vgg', num_layers, style=style, imsize=imsize_base)

        elif 'resnet' in net_str:
            num_layers_str = re.search('resnet(\d+)', net_str).groups()[0]
            num_layers = int(num_layers_str)

            net_type = NetworkType('resnet', num_layers, style=style, imsize=imsize_base)


        elif 'lenet' in net_str:
            net_type = NetworkType('lenet', 5, style=style, imsize=imsize_base)

        else:
            raise ValueError('unrecognized network string : %s' % net_str)


        if is_ssd_network:
            ssd_lite = 'lite' in net_str
            if net_type.name == 'vgg':
                return VGGSSDType(ssd_lite=ssd_lite, style=style, imsize=imsize)
            elif net_type.name == 'mobilenet':
                return MobilenetSSDType(mobilenet_version=version, ssd_lite=ssd_lite, style=style, imsize=imsize)
            else:
                return SSDNetType(base_net_type=net_type, imsize=imsize,
                                  ssd_lite=ssd_lite, style=style)
        else:
            return net_type



    @classmethod
    def equalto(cls, net1, net2):
        if sorted(net1.__dict__.keys()) != sorted(net2.__dict__.keys()):
            return False

        all_keys = net1.__dict__.keys()
        for k in all_keys:
            if isinstance(net1.__dict__[k], NetworkType):
                # For MobilenetSSD, one of the attributes is a 'base_net_type'
                # which is another NeworkType. We recurse the equalto method
                # to check these two base_net_types to see if they are the same
                if not NetworkType.equalto(net1.__dict__[k], net2.__dict__[k]):
                    return False
            else:
                if net1.__dict__[k] != net2.__dict__[k]:
                    return False

        return True




class MobilenetType(NetworkType):
    def __init__(self, name='mobilenet', version=1, style='tf', depth_multiplier=1.0,
                 imsize=224, ema=True, size='Large'):
        self.version = version # mobilenet version (1, 2 or 3)
        self.depth_multiplier = depth_multiplier
        self.imsize = imsize
        self.ema = ema
        if version == 3:
            size = size.title()
            assert size in ['Large', 'Small']
            self.size = size
        else:
            self.size = None

        super(MobilenetType, self).__init__(name=name, num_layers=None, style=style)

    def get_name(self, version_only=False, suffix_only=False):

        version_str = 'v%d' % self.version

        size_str = ''
        if self.version == 3:
            size_str = '_' + self.size.lower()

        style_str = '-' + self.style_abbrev
        depth_mult_str = '_%g' % self.depth_multiplier if self.depth_multiplier != 1 else ''

        imsize_str = ''
        if self.imsize is not None and self.imsize != 224:
            imsize_str = ('-%d' % self.imsize)

        ema_str = ''
        if self.style == 'tensorflow' and not self.ema:
            ema_str = '_p' # 'pristine' (ie. not ema)


        if version_only:
            details_str = version_str
        else:
            details_str = version_str + size_str + style_str + ema_str + depth_mult_str + imsize_str

        if suffix_only:
            return details_str
        else:
            return self.name + '_' + details_str

    def __str__(self):
        return self.get_name(version_only=False, suffix_only=False)



class SSDNetType(NetworkType):
    def __init__(self, base_net_type, ssd_lite=False, style=None, imsize=None, inherited=False):
        base_net_type.imsize = None
        self.base_net_type = base_net_type

        if style is None:
            style = base_net_type.style
        assert inherited == True, "Initialize using one of the child classes"

        self.ssd_lite = ssd_lite

        if isinstance(imsize, tuple):
            assert imsize[0] == imsize[1]
            imsize = imsize[0]

        self.imsize = imsize

        base_net_name = base_net_type.get_name(version_only=True)
        #if 'mobilenet' in base_net_name:
        ssd_name = 'ssd_' + base_net_name
        #else:
        #    ssd_name = 'ssd'

        super(SSDNetType, self).__init__(
            name=ssd_name, num_layers=None,
            style=style, imsize=self.imsize)


    def get_name(self, version_only=None, suffix_only=False):
        # e.g. "ssd_mobilenet_v2"
        name = super(SSDNetType, self).get_name(
            version_only=version_only, skip_size=True)

        if self.ssd_lite:
            # eg. "ssdlite_mobilenet_v2"
            name = name.replace('ssd', 'ssdlite')

        if self.imsize is not None and self.imsize != default_imsize_detection:
            name += '-' + str(self.imsize)

        return name



# Convenience class MobilenetSSDType (the SSD with Mobilenet backbone)
class MobilenetSSDType(SSDNetType):
    def __init__(self, mobilenet_version=1, ssd_lite=False, style='tf', size='Large', imsize=300):
        base_net = MobilenetType(version=mobilenet_version, style=style, size=size)

        super(MobilenetSSDType, self).__init__(
            base_net, ssd_lite=ssd_lite, style=style, imsize=imsize, inherited=True)

# Convenience class VGGSSDType (the original SSD with VGG backbone)
class VGGSSDType(SSDNetType):
    def __init__(self, ssd_lite=False, style='keras', imsize=300):
        base_net = NetworkType('vgg', num_layers=16, style='keras')

        super(VGGSSDType, self).__init__(
            base_net, ssd_lite=ssd_lite, style=style, imsize=imsize, inherited=True)



######### 3. Optimizer types ############

class OptimizerType():

    def get_name(self):
        raise NotImplementedError

    def get_optimizer(self, **kwargs):
        raise NotImplementedError

    def set_mini_epochs_per_epoch(self, n):
        raise NotImplementedError

    def __str__(self):
        return self.get_name()


    @classmethod
    def get_init_learning_rate(cls, learning_rate):
        if isinstance(learning_rate, float):
            return learning_rate
        elif isinstance(learning_rate, LearningRateScheduleType):
            return learning_rate.get_initial_learning_rate()



class SGDType(OptimizerType):
    def __init__(self, learning_rate=None, momentum=0., decay=0.0, nesterov=False,
                 weight_decay=False):
        if learning_rate is None:
            learning_rate = 0.01 # default SGD learning rate
        assert isinstance(learning_rate, (float, LearningRateScheduleType))
        self.learning_rate = learning_rate
        self.lr = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.decay = decay

        self.weight_decay=weight_decay

    def get_name(self):

        sgd_momentum_str = ''
        if self.momentum != 0:
            sgd_momentum_str = '_m' + ('%g' % self.momentum).replace('0.', '')

        sgd_nesterov_str = 'n' if self.nesterov else ''

        learning_rate_str = ''
        if isinstance(self.learning_rate, float):
            learning_rate_str = utl.get_small_num_str(self.learning_rate)
            if self.decay != 0:
                decay_str = '-d%g' % self.decay
                learning_rate_str += decay_str

        elif isinstance(self.learning_rate, LearningRateScheduleType):
            learning_rate_str = self.learning_rate.get_name()
            assert self.decay == 0

        weight_decay_str =''
        if self.weight_decay:
            weight_decay_str = 'W'

        opt_name = 'sgd' + weight_decay_str + sgd_momentum_str + sgd_nesterov_str + '-' + learning_rate_str
        return opt_name

    def get_optimizer(self, weight_decays=None, **kwargs):
        lr_init = SGDType.get_init_learning_rate(self.learning_rate)
        if not self.weight_decay:
            assert weight_decays is None, "Don't need weight decays for plain SGD"
            optimizer = keras.optimizers.SGD(
                lr=lr_init,
                momentum=self.momentum,
                nesterov=self.nesterov,
                decay=self.decay)
        else:
            assert weight_decays is not None, "Need weight decay dict for SGDW"
            optimizer = opt.SGDW(
                learning_rate=lr_init,
                momentum=self.momentum,
                nesterov=self.nesterov,
                decay=self.decay,
                weight_decays=weight_decays,
                # for now, bypass the weight decay normalization factor
                batch_size=1, total_iterations=1, init_verbose=False,
                **kwargs)

        return optimizer


class AdamType(OptimizerType):
    def __init__(self, learning_rate=None, beta_1=0.9, beta_2 = 0.999, epsilon=1e-7,
                 amsgrad=False, decay=0, weight_decay=False):
        if learning_rate is None:
            learning_rate = 0.001  # default Adam learning rate
        assert isinstance(learning_rate, (float, LearningRateScheduleType))
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.amsgrad = amsgrad
        self.decay = decay
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def get_name(self):
        opt_name = 'adam'

        if self.weight_decay:
            opt_name += 'W'

        if self.amsgrad:
            opt_name += '-ams'

        if isinstance(self.learning_rate, float):
            learning_rate_str = utl.get_small_num_str(self.learning_rate)
            if self.decay != 0:
                decay_str = '-d%g' % self.decay
                learning_rate_str += decay_str
            opt_name += '_' + learning_rate_str

        elif isinstance(self.learning_rate, LearningRateScheduleType):
            learning_rate_str = self.learning_rate.get_name()
            assert self.decay == 0
            opt_name += '_' + learning_rate_str

        return opt_name

    def get_optimizer(self, weight_decays=None, **kwargs):
        lr_init = SGDType.get_init_learning_rate(self.learning_rate)
        if not self.weight_decay:
            assert weight_decays is None, "Don't need weight decays for standard Adam"
            optimizer = keras.optimizers.Adam(
                lr=lr_init,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                amsgrad=self.amsgrad,
                epsilon=self.epsilon,
                decay=self.decay)
        else:
            assert weight_decays is not None, "Need weight decay dict for AdamW"
            optimizer = opt.AdamW(
                lr=lr_init, beta_1=self.beta_1, beta_2=self.beta_2,
                amsgrad=self.amsgrad, weight_decays=weight_decays,
                # for now, bypass the weight decay normalization factor
                batch_size=1, total_iterations=1, init_verbose=False,
                **kwargs,
                #batch_size=32,
                #total_iterations_wd=None, use_cosine_annealing=False,

                # lr_multipliers=None,
                # warm restart parameters:
                # total_iterations = 0, eta_min=0, eta_max=1, t_cur=0)
            )
        return optimizer


class RMSPropType(OptimizerType):
    def __init__(self, learning_rate=0.001, rho=0.9, decay=0.0,
                 weight_decay=False):
        self.learning_rate = learning_rate
        self.rho = rho
        self.decay = decay
        assert weight_decay == False

    def get_name(self):
        opt_name = 'rmsprop'

        learning_rate_str = ''
        if isinstance(self.learning_rate, float):
            learning_rate_str = utl.get_small_num_str(self.learning_rate)
            if self.decay != 0:
                decay_str = '-d%g' % self.decay
                learning_rate_str += decay_str

        elif isinstance(self.learning_rate, LearningRateScheduleType):
            learning_rate_str = self.learning_rate.get_name()
            assert self.decay == 0

        opt_name += '-' + learning_rate_str

        return opt_name

    def get_optimizer(self, weight_decays=None, ):
        lr_init = RMSPropType.get_init_learning_rate(self.learning_rate)
        assert weight_decays is None, "Weight decay not supported yet for rmsprop"
        return keras.optimizers.RMSprop(
            lr=lr_init,
            rho=self.rho,
            decay=self.decay)



######### 4. Learning Rate Schedule Types  ############


class LearningRateScheduleType():

    def __init__(self, lr_init):
        self.lr_init = lr_init

    def get_name(self, long=False):
        raise NotImplementedError

    def get_learning_rates(self):
        return None

    def get_callback(self):
        return None

    def set_mini_epochs_per_epoch(self, mini_epochs_per_epoch):
        assert isinstance(mini_epochs_per_epoch, int)
        self.mini_epochs_per_epoch = mini_epochs_per_epoch

    def __str__(self):
        return self.get_name(long=False)

    def __repr__(self):
        return self.get_name(long=True)

    def get_initial_learning_rate(self):
        return self.lr_init



class ConstantLearningRateType(LearningRateScheduleType):
    def __init__(self, learning_rate, num_epochs=10000):
        self.learning_rate = learning_rate
        self.num_epochs=num_epochs
        self.mini_epochs_per_epoch = 1
        super(ConstantLearningRateType, self).__init__(learning_rate)

    def get_learning_rates(self):
        return [self.learning_rate] * (self.num_epochs * self.mini_epochs_per_epoch)

    def get_name(self, long=False):
        if long:
            name = 'Constant Learning rate: %g' % self.learning_rate
        else:
            name = 'lr%g' % self.learning_rate
        return name


def get_monitor_mode(monitor):
    # for keras model_checkpoints and reduce_lr_on_plataeu, we want to know
    # whether to expect a monitored parameter to increase (mode='max') or
    # decrease (mode='min')
    if 'acc' in monitor:
        mode = 'max'
    elif 'loss' in monitor:
        mode = 'min'
    elif 'f1' in monitor:  # SSD monitor: 2 * prec * recall / (prec + recall)
        mode = 'max'
    elif 'map' in monitor.lower():
        mode = 'max'
    else:
        raise ValueError('Unrecognized monitor variable : %s' % monitor)

    return mode


class OnlineLearningRateAdapter(LearningRateScheduleType):
    # Use the keras callback ReduceLROnPlateu
    def __init__(self, lr_init, monitor=None, factor=0.1, patience=5,
                 min_delta = None, cooldown=0, min_lr = 0, num_epochs=None,
                 verbose=True):

        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.verbose = verbose
        self.num_epochs = num_epochs

        # `min` mode:  lr will be reduced when the quantity monitored has stopped decreasing;
        # `max` mode:  lr will be reduced when the quantity monitored has stopped increasing;

        if 'acc' in monitor:
            self.min_delta_default = 0.25  # assume accuracy in percentage
        elif 'loss' in monitor:
            self.min_delta_default = 1e-4
        elif 'f1' in monitor:  #SSD monitor: 2 * prec * recall / (prec + recall)
            self.min_delta_default = 0.25  # assume F1 in percentage
        else:
            raise ValueError('Unrecognized monitor variable : %s' % self.monitor)

        mode = get_monitor_mode(monitor)

        self.mode = mode
        if min_delta is None:
            self.min_delta = self.min_delta_default

        super(OnlineLearningRateAdapter, self).__init__(lr_init)

    def get_name(self, long=False):

        if long:
            name = 'Adaptive Learning rate (ReduceLROnPlateau): Init=%g. ' \
                   'Monitor %s [mode=%s], and reduce by %.g after %d epochs of ' \
                   'change < %.g, but with min=%.g. cooldown=%d epochs' % (
                       self.lr_init, self.monitor, self.mode, self.factor, self.patience,
                       self.min_delta, self.min_lr, self.cooldown)
        else:
            lr_init_str = utl.get_small_num_str(self.lr_init)

            name = 'lr' + lr_init_str + '-M%s' % self.monitor.replace('val_', '')
            if self.factor != 0.1:
                name += '-f%.2g' % self.factor
            if self.patience != 5:
                name += '-p%d' % self.patience
            if self.cooldown != 1:
                name += '-c%d' % self.cooldown
            if self.min_delta != self.min_delta_default:
                name += '-d%g' % self.min_delta

        return name

    def get_learning_rates(self):
        return None

    def get_callback(self):
        assert self.monitor is not None, "Define parameter to monitor for ReduceLROnPlateau"
        cb = keras.callbacks.ReduceLROnPlateau(
            monitor=self.monitor, factor=self.factor, patience=self.patience,
            verbose=self.verbose, mode=self.mode, min_delta=self.min_delta,
            cooldown=self.cooldown, min_lr=self.min_lr)
        return cb


class DecayedLearningRateType(LearningRateScheduleType):
    # Learning rate schedule where training starts with learnign rate =`lr_init`
    # After every epoch, it is multiplied by `decay_rate`, until it reaches
    # `lr_final` after `num_epochs` have been completed. Note that this is over-specified
    # eg. from `lr_init`, `decay_rate`, and `num_epochs` you can calculate `lr_final`.
    # So you can leave out either `lr_final`, `decay_rate` or `num_epochs` and it will
    # be calculated using the other parameters.

    def __init__(self, lr_init, lr_final=None, decay_rate=None, num_epochs=None):
        num_args_specified = sum([bool(lr_final), bool(decay_rate), bool(num_epochs)])
        if num_args_specified < 2:
            raise ValueError('Need at least 2 out of the 3 parameters '
                             '(lr_final, decay rate, num_epochs) to be provided')

        self.lr_final = lr_final
        self.decay_rate = decay_rate
        self.num_epochs = num_epochs
        self.mini_epochs_per_epoch = 1
        super(DecayedLearningRateType, self).__init__(lr_init)

    def fill_in_missing_data(self):

        if self.lr_final is None:
            assert self.decay_rate is not None and self.num_epochs is not None
            self.lr_final = self.lr_init * (self.decay_rate ** self.num_epochs)

        elif self.decay_rate is None:
            assert self.lr_final is not None and self.num_epochs is not None
            # Calculate decay rate
            self.decay_rate = np.exp(np.log(self.lr_final / self.lr_init) / self.num_epochs)

        elif self.num_epochs is None:
            # Calculate number of epochs needed to get to final_lr
            assert self.decay_rate is not None and self.lr_final is not None
            self.num_epochs = np.round(np.log(self.lr_final / self.lr_init) / np.log(self.decay_rate))

    def get_learning_rates(self):
        self.fill_in_missing_data()

        num_epochs = self.num_epochs * self.mini_epochs_per_epoch
        decay_rate = self.decay_rate ** (1/self.mini_epochs_per_epoch)

        #num_epochs = self.num_epochs
        #decay_rate = self.decay_rate

        all_learning_rates = self.lr_init * (decay_rate ** np.arange(num_epochs))
        assert np.abs(all_learning_rates[-1] - self.lr_final) < np.abs(self.lr_final)

        self.all_learning_rates = all_learning_rates
        return self.all_learning_rates

    def get_name(self, long=False):
        self.fill_in_missing_data()
        if long:
            name = 'Decayed Learning rate: Init=%g. Decay of %g each epoch. Final=%g (after %d epochs)' % (
                self.lr_init, self.decay_rate, self.lr_final, self.num_epochs)
        else:
            lr_init_str = utl.get_small_num_str(self.lr_init)
            decay_str = '-decay' + '%.3g' % self.decay_rate
            name = 'lr' + lr_init_str + decay_str
        return name



class StepLearningRateType(LearningRateScheduleType):
    # Learning rate schedule where a list of learning rates are provided
    # (`learning rates`), each of which is used for `num_epochs_each` epochs.
    # `num_epochs_each` can be a list of integers (the same length as `learning rates`)
    # or a single integer, which is then used for all learning rates.
    # This continues until a total of `num_epochs` have been completed.

    def __init__(self, learning_rates, num_epochs_each=10, num_epochs=None, warmup=None ):
        if isinstance(num_epochs_each, int):
            num_epochs_each = [num_epochs_each] * len(learning_rates)

        assert isinstance(num_epochs_each, list)
        if warmup:
            learning_rates.insert(0, warmup)
            num_epochs_each.insert(0, 1)
        self.warmup = warmup

        if num_epochs is not None:
            expected_num_epochs = sum(num_epochs_each)
            assert num_epochs == expected_num_epochs, \
                "Adding up all epochs : %s gives total of %d, not %d" % (
                    str(num_epochs_each), expected_num_epochs, num_epochs)

        n_tot = sum(num_epochs_each)
        self.num_epochs = sum(num_epochs_each)

        assert len(learning_rates) == len(num_epochs_each)
        self.learning_rates = learning_rates
        self.num_epochs_each = np.array(num_epochs_each)
        self.mini_epochs_per_epoch = 1

        super(StepLearningRateType, self).__init__(learning_rates[0])

    def get_learning_rates(self):
        all_learning_rates = []
        for i in range(len(self.learning_rates)):
            all_learning_rates.extend(
                [self.learning_rates[i]] * (self.num_epochs_each[i] * self.mini_epochs_per_epoch)  )

        assert len(all_learning_rates) == self.num_epochs
        return all_learning_rates

    def get_name(self, long=False):
        names = []
        i_start = 0 if not self.warmup else 1
        if long:
            for i in range(i_start, len(self.learning_rates)):
                names.append('%g [%d epochs]' % (self.learning_rates[i], self.num_epochs_each[i] ) )

            if bool(self.warmup):
                names.insert(0, '[1 warmup epoch of %g]' % self.warmup)

            name = 'Stepwise learning rate: ' + '; '.join(names)

        else:
            for i in range(i_start, len(self.learning_rates)):
                names.append('%gx%d' % (self.learning_rates[i], self.num_epochs_each[i] ) )
            if bool(self.warmup):
                names.insert(0, 'w%g' % self.warmup)
            name = 'lr' + '-'.join(names)
        return name



class DropLearningRateType(LearningRateScheduleType):
    # Learning rate schedule that starts at a certain initial value (`lr_init`),
    # then reduces by a factor (`drop_rate`) after every `drop_num_epochs` epochs.
    # This continues until a total of `num_epochs` have been completed.

    def __init__(self, lr_init, num_epochs=None, drop_rate=0.5, drop_num_epochs=10  ):


        if isinstance(drop_num_epochs, int):
            num_drops = int( np.ceil(num_epochs / drop_num_epochs) )
            drop_num_epochs = [drop_num_epochs] * num_drops
            drop_num_epochs_with_final = drop_num_epochs
            assert num_epochs is not None

        elif isinstance(drop_num_epochs, list):
            tot_epochs = sum(drop_num_epochs)
            if num_epochs is not None:
                assert num_epochs > tot_epochs, "Mismatch in total number of epochs"
            else:
                num_epochs = tot_epochs
            drop_num_epochs_with_final = drop_num_epochs.copy()
            drop_num_epochs_with_final.append(num_epochs - tot_epochs)
        else:
            raise ValueError('Invalid input')


        self.num_epochs = num_epochs
        self.drop_rate = drop_rate
        self.drop_num_epochs = drop_num_epochs
        self.drop_num_epochs_with_final = drop_num_epochs_with_final
        self.mini_epochs_per_epoch = 1
        super(DropLearningRateType, self).__init__(lr_init)

    def get_learning_rates(self):

        all_learning_rates = []
        current_lr = self.lr_init

        for i in range(len(self.drop_num_epochs_with_final)):
            all_learning_rates.extend(
                [current_lr] * (self.drop_num_epochs_with_final[i] * self.mini_epochs_per_epoch)  )
            current_lr *= self.drop_rate

        #num_epochs = self.num_epochs * self.mini_epochs_per_epoch
        #drop_num_epochs = self.drop_num_epochs * self.mini_epochs_per_epoch
        #all_learning_rates = self.lr_init * self.drop_rate ** \
        #                     np.floor( np.arange(num_epochs) / drop_num_epochs)
        return all_learning_rates

    def get_name(self, long=False):
        uniform_drops = all(self.drop_num_epochs[0] == drop_n for drop_n in self.drop_num_epochs)

        if uniform_drops:
            freq_str_long = ' every %d epochs' % self.drop_num_epochs[0]
            freq_str = '%dep' % self.drop_num_epochs[0]
        else:
            freq_str_long = ' on epochs %s' % (','.join([str(n) for n in self.drop_num_epochs]))
            freq_str = 'ep%s' % '-'.join([str(n) for n in self.drop_num_epochs])

        if long:
            name = 'Drop-based learning schedule: Init=%g. Drop by a factor of %g %s' % (
                self.lr_init, self.drop_rate, freq_str_long)
        else:
            name = 'lr%g-drop%.2g-%s' % (self.lr_init, self.drop_rate, freq_str)
        return name



class DatasetAugmentationsType():

    def get_name(self):
        raise NotImplementedError

    def __str__(self):
        return self.get_name()



class KerasDatasetAugmentationsType(DatasetAugmentationsType):
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format='channels_last',
                 validation_split=0.0,
                 interpolation_order=1,
                 dtype='float32'):

        self.featurewise_center=featurewise_center
        self.samplewise_center=samplewise_center
        self.featurewise_std_normalization=featurewise_std_normalization
        self.samplewise_std_normalization=samplewise_std_normalization
        self.zca_whitening=zca_whitening
        self.zca_epsilon=zca_epsilon
        self.rotation_range=rotation_range
        self.width_shift_range=width_shift_range
        self.height_shift_range=height_shift_range
        self.brightness_range=brightness_range
        self.shear_range=shear_range
        self.zoom_range=zoom_range
        self.channel_shift_range=channel_shift_range
        self.fill_mode=fill_mode
        self.cval=cval
        self.horizontal_flip=horizontal_flip
        self.vertical_flip=vertical_flip
        self.rescale=rescale
        self.preprocessing_function=preprocessing_function
        self.data_format=data_format
        self.validation_split=validation_split
        self.interpolation_order=interpolation_order
        self.dtype=dtype

    def get_name(self):
        names = []
        if self.featurewise_center:
            names.append('ms')
        if self.featurewise_std_normalization:
            names.append('std')
        if self.zca_whitening:
            names.append('zca-%g' % self.zca_whitening)

        if self.rotation_range != 0:
            names.append('rot%g' % self.rotation_range)

        if self.width_shift_range != 0 and self.height_shift_range != 0 and \
            self.width_shift_range == self.height_shift_range:
            names.append('hws%g' % self.width_shift_range)
        else:
            if self.width_shift_range != 0:
                names.append('ws%g' % self.width_shift_range)
            if self.height_shift_range != 0:
                names.append('hs%g' % self.height_shift_range)

        if self.brightness_range is not None:
            names.append('br%g' % self.brightness_range)
        if self.channel_shift_range != 0:
            names.append('chsh%g' % self.channel_shift_range)

        if self.shear_range != 0:
            names.append('sh%g' % self.shear_range)
        if self.zoom_range != 0:
            names.append('zm%g' % self.zoom_range)

        if self.horizontal_flip:
            names.append('hFlp')
        if self.vertical_flip:
            names.append('vFlp')

        name = '-'.join(names)
        return name



class SSDDatasetAugmentationsType(DatasetAugmentationsType):
    def __init__(self,
                 img_height=None,
                 img_width=None,

                 subtract_mean=None,  # usually not defined, since these
                 divide_by_stddev=None, # centering/scaling steps are done
                 swap_channels=False,   # in the network

                 photometric=False,
                 expand=False,
                 crop=False,
                 flip=False,

                 resize=False,
                 ):

            self.img_height = img_height
            self.img_width = img_width

            self.subtract_mean=subtract_mean
            self.divide_by_stddev=divide_by_stddev
            self.swap_channels=swap_channels

            self.photometric=photometric
            self.expand=expand
            self.crop=crop
            self.flip=flip

    def build_kwargs(self):
        kwargs = dict(photometric=self.photometric,
                      expand=self.expand,
                      crop=self.crop,
                      flip=self.flip)
        return kwargs

    def get_name(self):
        names = []
        if self.subtract_mean:
            names.append('ms')
        if self.divide_by_stddev:
            names.append('std')
        if self.swap_channels:
            names.append('bgr')

        if self.photometric:
            names.append('p')
        if self.expand:
            names.append('x')
        if self.crop:
            names.append('c')
        if self.flip:
            names.append('f')


        '''
        def min_max_prob_str(param_name, x_vals, default):
            default_range = default[:2]
            default_prob = default[2]
            strs = [param_name]
            if x_vals[:2] != default_range:
                strs.extend(['%f' % x for x in x_vals[:2]])
            if x_vals[2] != default_prob:
                strs.append('p%f'% x_vals[2])
            str_final = '-'.join(strs)
            return str_final

        def x12_y12_str(param_name, x1, y1, x2, y2):

            if x1 == x2 == y1 == y2:
                str_final = param_name + '%d' % x1
            else:
                if x1 == x2:
                    x_str = 'x%d' % x1
                else:
                    x_str = 'x%d-%d' % (x1, x2)

                if y1 == y2:
                    y_str = 'y%d' % y1
                else:
                    y_str = 'y%d-%d' % (y1, y2)
                str_final = param_name + '-' + x_str + '-' + y_str

            return str_final

        def hw_min_trial_str(height, width, min_1_object, max_n_trials, mix_ratio=None):
            strs = []
            if width != self.img_width or height != self.img_height:
                strs.append( '%dx%d' % (width, height) )
            if min_1_object != 1:
                strs.append( 'min%d' % min_1_object)
            if max_n_trials != 3:
                strs.append('tr%d' % max_n_trials)
            if mix_ratio is not None and mix_ratio != 0.5:
                strs.append('mx%f' % mix_ratio)

            str_final = '-'.join(strs)
            if bool(str_final):
                str_final = '-' + str_final
            return str_final

        if bool(self.brightness): # tuple of form (min, max, prob)
            brightness_min_max_prob_default = (0.5, 2, 0.5)
            brightness_str = min_max_prob_str('br',
                self.brightness, brightness_min_max_prob_default)
            names.append(brightness_str)

        if self.flip:
            flip_prob_default = 0.5
            flip_str = ''
            if self.flip != flip_prob_default:
                flip_str = '%f' % self.flip
            names.append('hFlp' + flip_str)

        if self.translate:
            (xmin, xmax), (ymin, ymax), prob = self.translate
            tr_str = x12_y12_str('tr', xmin, xmax, ymin, ymax)
            if prob != 1:
                tr_str += '-p%f' % prob

            names.append(tr_str)

        if self.scale: # tuple of form (min, max, prob)
            scale_min_max_prob_default = (0.5, 2, 0.5)
            scale_str = min_max_prob_str('scl',
                                         self.scale, scale_min_max_prob_default)
            names.append(scale_str)

        if self.max_crop_and_resize:
            height, width, min_1_object, max_n_trials = self.max_crop_and_resize
            mcr_str = hw_min_trial_str(height, width, min_1_object, max_n_trials)
            names.append('mcr' + mcr_str)

        if self.random_pad_and_resize:
            height, width, min_1_object, max_n_trials, mix_ratio = self.random_pad_and_resize
            rpr_str = hw_min_trial_str(height, width, min_1_object, max_n_trials, mix_ratio)
            names.append('rpr' + rpr_str)
            self.random_crop = False
            self.crop = False


        if self.random_crop:
            height, width, min_1_object, max_n_trials = self.random_crop
            rc_str = hw_min_trial_str(height, width, min_1_object, max_n_trials)
            names.append('rc' + rc_str)

        if self.crop:
            (xl, xr), (yl, yr) = self.crop
            crop_str = x12_y12_str('crop', xl, xr, yl, yr)
            names.append(crop_str)

        if self.resize:
            height, width = self.resize
            size_str = ''
            if width != self.img_width or height != self.img_height:
                size_str = '%dx%d' % (width, height)

            names.append('rsz' + size_str)

        if self.gray:
            names.append('gray')
        '''

        name = ''.join(names)
        return name





default_patience=3
default_cooldown=3
default_factor = 2.0
default_max_lambda = 1.0

'''
adjust_distillation_cb = AdjustLambdaCallback(
    monitor='lp_acc', target='hp_acc', threshold=2.0
lambda_name = 'distillation_loss', factor = 2.0, max_lambda = 0.1,
                                                              patience = 3, cooldown = 3)

adjust_bitloss_cb = AdjustLambdaCallback(
    monitor='bits_per_layer', target=4.0, threshold=0.1,
    lambda_name='bit_loss', factor=1.4, max_lambda=0.1,
    patience=3, cooldown=3)
'''

class LambdaAdaptType():
    def __init__(self, monitor='lp_acc', target='hp_acc', min_delta=2.0,
                 lambda_name='distillation_loss', factor=default_factor,
                 patience=default_patience,
                 max_lambda=default_max_lambda,
                 cooldown=default_cooldown, verbose=True):


        self.monitor = monitor
        self.target = target
        self.min_delta = min_delta

        self.lambda_name = lambda_name
        self.factor = factor
        self.patience = patience
        self.max_lambda=max_lambda
        self.cooldown=cooldown
        self.verbose=verbose

    def get_name(self, long=False):

        name = 'Ad'
        if self.lambda_name.lower() == 'distillation_loss':
            lambda_name_str = 'Ds'
        elif self.lambda_name.lower() == 'lp':
            lambda_name_str = 'Lp'
        elif self.lambda_name.lower() == 'bit_loss':
            lambda_name_str = 'Bt'
        else:
            raise ValueError('unhandled lambda ')
        name += lambda_name_str



        if (self.monitor == 'lp_acc' and self.target == 'hp_acc') or \
                (self.monitor == 'val_lp_acc' and self.target == 'val_hp_acc'):

            monitor_str='Acc'
            if self.lambda_name.lower() == 'distillation_loss':
                monitor_str = '' # this is the default

            if 'val' in self.monitor:
                monitor_str = 'V' + monitor_str

        elif 'bits' in self.monitor and isinstance(self.target, (float, int, np.number)):
            monitor_str = 'Bt'
            if self.lambda_name.lower() == 'bit_loss':
                monitor_str = '' # this is the default

            monitor_str += '%g' % float(self.target)

        else:
            monitor_str = '-' + self.monitor + '-' + str( self.target )

        name += monitor_str

        min_delta_default = None
        if self.monitor == 'acc':
            min_delta_default = 2.0
        elif self.monitor == 'loss':
            min_delta_default = 0.1
        else:
            pass
            #raise ValueError('Unhandled monitor')


        #if self.lambda_name == 'distillation_loss':
        if self.min_delta != min_delta_default or True:
            name += '-d%g' % self.min_delta
        if self.factor != default_factor or True:
            name += '-f%.2g' % self.factor
        if self.patience != default_patience:
            name += '-p%d' % self.patience
        if self.cooldown != default_cooldown:
            name += '-c%d' % self.cooldown
        if self.max_lambda != default_max_lambda:
            name += '-mx%g' % self.max_lambda

        return name

    def __str__(self):
        return self.get_name()

    def get_callback(self):
        callback = net_callbacks.AdjustLambdaCallback(
            monitor=self.monitor, target=self.target,
            factor=self.factor, lambda_name=self.lambda_name,
            min_delta=self.min_delta, patience=self.patience,
            max_lambda=self.max_lambda,
            cooldown=self.cooldown, verbose=self.verbose)
        return callback


########### Network defaults
class NetworkDefaults():
    pass


class MobilenetDefaults(NetworkDefaults):
    def __init__(self, net_type, config):
        assert isinstance(net_type, MobilenetType)

        self.defaults = net_type.style
        kernel_regularizer = None
        if config.do_weight_regularization and not config.decouple_weight_decay:
            weight_decay = config.weight_decay_factor
            kernel_regularizer = keras.regularizers.l2(weight_decay)

        self.dropout_rate = None
        if self.defaults == 'tensorflow':
            # defaults in tensorflow/slim
            self.conv_args = dict(
                kernel_initializer=keras.initializers.truncated_normal(stddev=0.09),
                kernel_regularizer=kernel_regularizer,
                use_bias=False)

            self.bn_args = dict(
                momentum= 0.997,
                epsilon=1e-3)

            if net_type.version == 1:
                self.dropout_rate = 1e-3
            elif net_type.version == 2:
                self.dropout_rate = 0.2
            elif net_type.version == 3:
                self.dropout_rate = 0.2
                self.conv_args.pop('use_bias')  # use_bias is provided explicitly for each layer.

        elif self.defaults == 'keras.applications':
            # defaults in keras.applications
            if net_type.version == 1:
                kernel_initializer = "glorot_uniform"  # keras default
            else:
                kernel_initializer = "he_normal"

            self.conv_args = dict(
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                use_bias=False)
            self.bn_args = dict(
                momentum= 0.999,
                epsilon=1e-3)


            if net_type.version == 1:
                self.dropout_rate = 1e-3
            elif net_type.version == 2:
                self.dropout_rate = 0.25

        else:
            raise ValueError('unknown model defaults.')


        if config.bn_epsilon is not None:
            self.bn_args['epsilon'] = config.bn_epsilon
        if config.bn_momentum is not None:
            self.bn_args['momentum'] = config.bn_momentum

        if net_type.version in [1, 2]:
            # default activation in Mobilenet-v1 paper is relu6
            # default activation in Mobilenet-v2 paper is relu6 (keras/ tensorflow)
            self.conv_args['activation'] = 'relu6'
        if config.activation is not None:
            # mobilenet-SSD (keras version) uses relu instead of relu6,
            # For those SSD networks, we put this activation in config file
            # to allow to override defaults
            self.conv_args['activation'] = config.activation

        if net_type.version in [3]:
            # in Mobilenet-v3, activation is defined separately for each layer
            # don't define here in the defaults.
            self.conv_args.pop('activation', None)


class VGGDefaults(NetworkDefaults):

    def __init__(self, net_type, config):
        self.defaults = net_type.style
        assert 'keras' in self.defaults

        # The original mobilenet-SSD network in caffe expects images in BGR format.
        # Need the 'swap_channels' if not loading with opencv2 imread
        #self.preprocess_op_list = [('subtract', [127.5, 127.5, 127.5]), ('divide', 127.5), ('swap_channels')]
        # self.preprocess_op_list = [('subtract', [127.5, 127.5, 127.5]), ('divide', 127.5)]
        #self.explicit_zero_prepadding_for_stride2 = True

        self.conv_kwargs = dict(
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(0.0005)
        )

        self.bn_kwargs = dict(epsilon=0)
        self.do_batchnorm = False

        if config.fully_convolutional:  # is the base network for SSD
            self.fc_units = 1024
            self.fc_kernel_size = (3, 3)
            self.fc_dilation_rate = (6, 6) # ("atrous" convolution) makes SSD faster
            self.pool_sizes   = [2, 2, 2, 2, 3]
            self.pool_strides = [2, 2, 2, 2, 1]
            #self.base_for_ssd = True

        else:
            self.fc_units = 4096
            self.fc_kernel_size = (1, 1)
            self.fc_dilation_rate = (1, 1)
            self.pool_sizes   = [2, 2, 2, 2, 2]
            self.pool_strides = [2, 2, 2, 2, 2]
            #self.base_for_ssd = False

        #else:
        #    raise ValueError('Unknown VGG style : %s' % net_type.style)


class SSDDefaults(NetworkDefaults):
    def __init__(self, net_type, config):
        assert isinstance(net_type, SSDNetType)
        base_net = net_type.base_net_type
        mobilenet_version = None
        if isinstance(base_net, MobilenetType):
            mobilenet_version = base_net.version
            extra_layer_filters = [256, 512,  128, 256,  128, 256,  64, 128]
            extra_layer_strides = [2, 2, 2, 2]
            extra_layer_paddings = ['same', 'same', 'same', 'same']
            feat_extractor_strides = [16, 32]
            do_L2_norm_on_first_feature_map = False
            extra_layers_use_bias =False
        else:
            assert base_net.name.lower() == 'vgg'
            extra_layer_filters = [256, 512,  128, 256,  128, 256,  128, 256]
            extra_layer_strides = [2, 2, 1, 1]
            extra_layer_paddings = ['same', 'same', 'valid', 'valid']
            feat_extractor_strides = [8, 16]
            do_L2_norm_on_first_feature_map = True
            extra_layers_use_bias = True
            #assert isinstance(config, net_c)

        self.defaults = net_type.style
        self.extra_layer_filters = extra_layer_filters
        self.extra_layer_strides = extra_layer_strides
        self.extra_layer_paddings = extra_layer_paddings
        self.extra_layers_use_bias = extra_layers_use_bias
        self.feat_extractor_strides = feat_extractor_strides

        self.do_L2_norm_on_first_feature_map = do_L2_norm_on_first_feature_map




        if 'keras' in self.defaults:
            # The original mobilenet-SSD network in caffe expects images in BGR format.
            # Need the 'swap_channels' if not loading with opencv2 imread
            if isinstance(base_net, MobilenetType):
                self.preprocess_op_list = [('subtract', [127.5, 127.5, 127.5]), ('divide', 127.5), ('swap_channels')]
            else:
                self.preprocess_op_list = [('subtract', [123, 117, 104]),  ('swap_channels', )]


            #self.preprocess_op_list = [('subtract', [127.5, 127.5, 127.5]), ('divide', 127.5)]
            self.explicit_zero_prepadding_for_stride2 = True
            self.conv_kwargs = dict(
                kernel_initializer = 'he_normal',
                kernel_regularizer = keras.regularizers.l2(0.0005),
                )

            self.bn_kwargs = dict(momentum = 0.99, epsilon = 1e-5)
            self.activation = 'relu'

            self.feat_extractor_activation_1 = 'relu'
            self.feat_extractor_activation_2 = 'relu'
            self.do_batchnorm = base_net.name.lower() != 'vgg'

            #self.n_classes = 20
            #self.add_variances_to_anchors = True
            #self.variances = (0.1, 0.1, 0.2, 0.2)

            #self.mimic_tf = False
            #self.imsize = (300, 300)


        elif self.defaults == 'tensorflow':  # defaults for all tensorflow models:
            self.preprocess_op_list = [('divide', 127.5), ('subtract', 1.0)]

            self.explicit_zero_prepadding_for_stride2 = False
            self.conv_kwargs = dict(
                kernel_initializer = keras.initializers.truncated_normal(mean=0.0, stddev=0.03),
                kernel_regularizer=keras.regularizers.l2(0.00004)
            )
            self.bn_kwargs = dict(momentum=0.9997, epsilon=1e-3)
            self.activation = 'relu6'
            self.do_batchnorm = True

            self.feat_extractor_activation_1 = 'relu' if mobilenet_version == 1 else 'relu6'
            self.feat_extractor_activation_2 = 'relu6'

            #self.n_classes = 90
            #self.add_variances_to_anchors = False
            #self.variances = (1.0, 1.0, 1.0, 1.0)

            # use_tensoadd_variances_to_anchorsrflow_models = True
            #self.imsize = (300, 300) if net_type.mobilenet_version in [1, 2] else (320, 320)
            #self.feature_extractor = net_type.get_name().replace('lite', '')
            self.add_nms_to_network = True

            #self.depthwise_feature_extractor = net_type.mobilenet_version in [2, 3]
        else:
            raise ValueError('Unsupported style for SSD : %s' % self.defaults)

        self.depthwise_feature_extractor = mobilenet_version in [2, 3]
        self.depthwise_box_predictors = net_type.ssd_lite or mobilenet_version in [3]


        do_default_regularization = True
        weight_decay_factor = None
        if not config.do_weight_regularization or \
                config.do_weight_regularization and config.decouple_weight_decay:
            # No weight decay.  Or will be added later in optimizer
            do_default_regularization = False
        elif config.do_weight_regularization and not config.decouple_weight_decay and \
                config.weight_decay_factor is not None:
            # Use different weight decay defined in config file
            weight_decay_factor = config.weight_decay_factor

        if not do_default_regularization:
            # Remove weight decay from defaults
            self.conv_kwargs['kernel_regularizer'] = None
        elif weight_decay_factor is not None:
            # Modify weight decay for defaults
            self.conv_kwargs['kernel_regularizer'] = \
                keras.regularizers.l2(weight_decay_factor)


class ResnetDefaults(NetworkDefaults):
    def __init__(self, defaults):
        self._defaults = defaults
        if defaults == 'keras.applications':
            self.do_preprocessing = True
            self.do_input_batchnorm = False

            self.do_conv_before_batchnorm = True
            self.shortcut_after_first_activation = False
            self.stride2_in_first_expansion_block = True
            self.force_conv_before_first_shortcut = False
            self.do_batchnorm_after_shortcuts = True
            self.do_batchnorm_after_last_block = False
            self.final_activation_before_add = False

            self.conv_kwargs = dict(explicit_zero_prepadding=False)
            self.bn_kwargs = {}
        elif defaults == 'image-classifiers':
            self.do_preprocessing = False
            self.do_input_batchnorm = True

            self.do_conv_before_batchnorm = False
            self.shortcut_after_first_activation = True
            self.stride2_in_first_expansion_block = False
            self.force_conv_before_first_shortcut = True
            self.do_batchnorm_after_shortcuts = False
            self.do_batchnorm_after_last_block = True
            self.final_activation_before_add = True

            self.conv_kwargs = dict(
                explicit_zero_prepadding=True,
                kernel_initializer='he_uniform',
                use_bias=False)
            self.bn_kwargs = dict(epsilon=2e-05, momentum=0.99)

        elif defaults == 'torch':
            self.do_preprocessing = False
            self.do_input_batchnorm = False

            self.do_conv_before_batchnorm = False
            self.shortcut_after_first_activation = True
            self.stride2_in_first_expansion_block = False
            self.force_conv_before_first_shortcut = True
            self.do_batchnorm_after_shortcuts = False
            self.do_batchnorm_after_last_block = True
            self.final_activation_before_add = True

            self.conv_kwargs = dict(
                explicit_zero_prepadding=True,
                kernel_initializer='he_uniform',
                use_bias=False)
            self.bn_kwargs = dict(epsilon=2e-05, momentum=0.99)

        else:
            raise ValueError('Unrecongized defaults name : %s' % defaults)


if __name__ == "__main__":
    lr1 = ConstantLearningRateType(learning_rate=0.01)
    lr1_rates = lr1.get_learning_rates()
    lr1_name = lr1.get_name()
    lr1_name2 = lr1.get_name(long=True)

    lr2 = DecayedLearningRateType(lr_init=0.001, decay_rate=0.9, num_epochs=100)
    lr2_rates = lr2.get_learning_rates()
    lr2_name = lr2.get_name()
    lr2_name2 = lr2.get_name(long=True)

    lr3 = StepLearningRateType([0.1, 0.01, 0.001], num_epochs_each=10)
    lr3_rates = lr3.get_learning_rates()
    lr3_name = lr3.get_name()
    lr3_name2 = lr3.get_name(long=True)

    lr4 = DropLearningRateType(lr_init=0.01, num_epochs=100, drop_rate=0.5)
    lr4_rates = lr4.get_learning_rates()
    lr4_name = lr4.get_name()
    lr4_name2 = lr4.get_name(long=True)


    sgd1 = SGDType(learning_rate=lr1, momentum=0.95, nesterov=True)
    sgd1_opt = sgd1.get_optimizer()
    sgd1_name = sgd1.get_name()

    sgd2 = SGDType(learning_rate=lr2, momentum=0.95, nesterov=True)
    sgd2_opt = sgd2.get_optimizer()
    sgd2_name = sgd2.get_name()

    adam1 = AdamType(learning_rate=lr3)
    adam1_opt = adam1.get_optimizer()
    adam1_name = adam1.get_name()

    adam2 = AdamType(learning_rate=lr4)
    adam2_opt = adam2.get_optimizer()
    adam2_name = adam2.get_name()

    a = 1
    # verify all Network types conversion to/from strings:
    all_nets = [NetworkType('VGG', 16, 'keras'),
                NetworkType('VGG', 16, 'ic'),
                NetworkType('VGG', 19, 'keras'),
                NetworkType('ResNet', 50, 'keras'),
                MobilenetType(version=1, style='keras'),
                MobilenetType(version=2, style='keras'),

                MobilenetType(version=1, style='tf', depth_multiplier=1, ema=True),
                MobilenetType(version=2, style='tf', depth_multiplier=1, ema=True),
                MobilenetType(version=2, style='tf', depth_multiplier=1, ema=False),
                MobilenetType(version=2, style='tf', depth_multiplier=0.5, ema=False),
                MobilenetType(version=2, style='tf', depth_multiplier=1.4, ema=False),
                MobilenetType(version=3, style='tf', depth_multiplier=1, ema=True, size='Large'),

                MobilenetSSDType(mobilenet_version=1, style='keras'),
                MobilenetSSDType(mobilenet_version=1, style='keras', imsize=224),
                MobilenetSSDType(mobilenet_version=2, style='tf', ssd_lite=True),
                VGGSSDType(ssd_lite=False),
                VGGSSDType(ssd_lite=False, imsize=224)
                ]
    for net in all_nets:
        net_str = str(net)
        print('%s : ...' % net_str)
        net2 = NetworkType.from_str(net_str)
        assert NetworkType.equalto(net, net2)
        print('%s : OK' % net_str)


