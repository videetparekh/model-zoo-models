import tensorflow.compat.v1 as tf
from tensorflow.python import pywrap_tensorflow
import tensorflow.keras as keras
import scipy.io

import numpy as np
import re
import os
import h5py
import hickle as hkl

import utility_scripts.misc_utl as utl
from collections import OrderedDict

class WeightsReader(): # Parent class for all weights readers
    # currently implemented subclasses:
    #   CheckpointReader()     # for reading tensorflow checkpoints
    #   H5WeightsFileReader()  # for reading keras applications .h5 weights files

    def __init__(self, filename, filetype, all_raw_tensor_names, name_converter=None, load_all_into_memory=True):
        self._filename = filename
        self._filetype = filetype
        self._all_raw_tensor_names = all_raw_tensor_names
        if name_converter is None:
            name_converter = NullNameConverter()
        self._name_converter = name_converter

        self._all_tensor_names = [self._name_converter.convert_name(name) for name in all_raw_tensor_names]
        self._load_all_into_memory = load_all_into_memory

        # make sure all tensor names qre unique:
        unique_names, names_count = np.unique(self._all_tensor_names, return_counts=True)
        if len(unique_names) < len(self._all_tensor_names):
            raise ValueError('Error: The following tensor names were duplicated: %s' %
                             list(unique_names[np.nonzero(names_count > 1)]) )

        self._all_tensors = {}
        self._tensors_used = []

        self._tensors_used = [False] * len(self._all_tensor_names)

        if load_all_into_memory:
            self._all_tensors = self._read_all_tensors()

            #for tensor_name in self._all_tensor_names:
            #    self._all_tensors[tensor_name] = self.read_tensor(tensor_name)
            #print('Loaded %d tensors into memory' % len(self._all_tensor_names))

    def get_tensor_list(self):
        return self._all_tensor_names

    def _read_tensor(self, tensor_name):
        raise NotImplementedError("Overriding classes should implement this.")

    def _read_all_tensors(self):
        all_tensors = {}
        for raw_tensor_name in self._all_raw_tensor_names:
            tensor_name = self._name_converter.convert_name(raw_tensor_name)
            all_tensors[tensor_name] = self._read_tensor(raw_tensor_name)
        return all_tensors

    def matches_tensor_name(self, tensor_name, try_match_to_raw_names=True):
        tf1 = tensor_name in self._all_tensor_names
        idx = None
        if tf1:
            idx = self._all_tensor_names.index(tensor_name)
        if not tf1 and try_match_to_raw_names:
            tf1 = tensor_name in self._all_raw_tensor_names
            if tf1:
                idx = self._all_raw_tensor_names.index(tensor_name)

        return tf1, idx

    def has_tensor(self, tensor_name, try_match_to_raw_names=True, return_match_idx=False):

        tf1, idx = self.matches_tensor_name(
            tensor_name, try_match_to_raw_names=try_match_to_raw_names)

        try_without_prefixes = True
        if not tf1 and try_without_prefixes:
            tensor_name_parts = tensor_name.split('/')
            reader_prefix = self.get_prefix()

            new_tensor_name = tensor_name
            if 'FeatureExtractor' in tensor_name:
                try_remove = [i == 0 or 'FeatureExtractor' in part for i,part in enumerate(tensor_name_parts)]
                new_tensor_name = '/'.join([part for part, tf in zip(tensor_name_parts, try_remove) if not tf ])

                tf1, idx = self.matches_tensor_name(
                    new_tensor_name, try_match_to_raw_names=try_match_to_raw_names)

            if not tf1:
                name_prefix = new_tensor_name[: new_tensor_name.index('/')] + '/'
                new_tensor_name_v2 = new_tensor_name.replace(name_prefix, reader_prefix)
                tf1, idx = self.matches_tensor_name(
                    new_tensor_name_v2, try_match_to_raw_names=try_match_to_raw_names)


        if return_match_idx:
            return tf1, idx
        else:
            return tf1

    def get_tensor(self, tensor_name, mark_as_used=True, try_match_to_raw_names=True, error_if_already_used=False):

        name_match, idx = self.has_tensor(
            tensor_name, try_match_to_raw_names=try_match_to_raw_names, return_match_idx=True)
        #name_match = tensor_name in self._all_tensor_names
        #raw_name_match = not name_match and try_match_to_raw_names and tensor_name in self._all_raw_tensor_names

        tensor_name = self._all_tensor_names[idx]
        #if name_match:
        #    idx = self._all_tensor_names.index(tensor_name)
        #elif raw_name_match:
        #    idx = self._all_raw_tensor_names.index(tensor_name)
        #    tensor_name = self._name_converter.convert_name(tensor_name)
        #else:
        #    raise ValueError('Unrecognized tensor name : %s' % tensor_name)

        if error_if_already_used and self._tensors_used[idx]:
            raise ValueError('Tensor %s has already been used' % tensor_name)

        if mark_as_used:
            self._tensors_used[idx] = True

        if self._load_all_into_memory:
            t = self._all_tensors[tensor_name]
        else:
            t = self._read_tensor(tensor_name)

        return t


    def print_all_tensors(self, only_unused=False, show_raw_names=False, sort_by_name=False):
        unused_str = 'unused ' if only_unused else ''
        print('Displaying all %stensors from %s file %s' % (unused_str, self._filetype, self._filename))
        unused_count = 0
        all_tensors_display = []
        for i, tensor_name in enumerate(self._all_tensor_names):
            if only_unused and self._tensors_used[i]:
                continue

            if only_unused:
                unused_count += 1

            t = self.get_tensor(tensor_name, mark_as_used=False)
            raw_tensor_name_str = ''
            if show_raw_names:
                idx = self._all_tensor_names.index(tensor_name)
                raw_tensor_name = self._all_raw_tensor_names[idx]
                if tensor_name != raw_tensor_name:
                    raw_tensor_name_str = ' <%s>' % raw_tensor_name

            all_tensors_display.append("%s %s (%s) %s" % (tensor_name, raw_tensor_name_str, t.dtype, t.shape))

        if sort_by_name:
            all_tensors_display = sorted(all_tensors_display)

        for i, t_name in enumerate(all_tensors_display):
            print("tensor %3d: %s" % (i+1, t_name))

        if only_unused and unused_count == 0:
            print('  [[ No unused tensors. ]] ')

    def save_to_mat_file(self, matfile):
        S_data = {}
        for i, tensor_name in enumerate(self._all_tensors):
            t = self.get_tensor(tensor_name, mark_as_used=False)
            tensor_name_valid = tensor_name.replace(':0', '').replace('/', '__')
            S_data[tensor_name_valid] = t

        scipy.io.savemat(matfile, S_data)

    def descrip(self):
        return '%s reader (%s)' % (self._filetype, self._filename)

    def get_prefix(self, check_all_strs=True):

        if check_all_strs:
            if len(self._all_tensor_names) > 1:
                prefix0 = utl.factorize_strs(self._all_tensor_names,
                                             return_common_str_only=True)
            else:
                prefix0 = ''

            if prefix0.count('/') > 0 and 'FeatureExtractor' not in prefix0:
                prefix0 = prefix0.split('/')[0]

            if bool(prefix0) and not prefix0.endswith('/'):
                prefix0 += '/'

        else:
            prefix0 = self._all_tensor_names[0].split('/')[0] + '/'

        prefix0_lower = prefix0.lower()
        if prefix0_lower.startswith('logits') or \
                prefix0_lower.startswith('conv') or \
                prefix0_lower.startswith('bn_'):
            prefix0 = ''

        return prefix0




class CheckpointReader(WeightsReader):
    """
    A weights reader for reading the weights from a tensorflow checkpoint
    (Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py)
    """

    def __init__(self, checkpoint_prefix, name_converter='default', exponential_moving_average=True):
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_prefix)
        self._reader = reader
        self._checkpoint_prefix = checkpoint_prefix
        filetype = 'Checkpoint'

        all_tensors_str = self._reader.debug_string().decode("utf-8", errors="ignore")
        all_tensors_list_descrip = all_tensors_str.split('\n')
        all_tensors_names_full = [x.split(' ')[0] for x in all_tensors_list_descrip if len(x) > 0]
        #all_tensors_dtypes = [x.split(' ')[1] for x in all_tensors_list_descrip if len(x) > 0]
        #all_tensors_shapes = [x.split(' ')[2] for x in all_tensors_list_descrip if len(x) > 0]

        suffixes_ignore = ['/ExponentialMovingAverage', '/Momentum', '/RMSProp', '/RMSProp_1', 'global_step']
        ignore_tensor_name = lambda t_name: any([t_name.endswith(suff) for suff in suffixes_ignore])
        all_raw_tensors_names = [tensor_name for tensor_name in all_tensors_names_full if not ignore_tensor_name(tensor_name) ]
        self._exponential_moving_average = exponential_moving_average
        self._all_tensors_names_full = all_tensors_names_full
        self._have_ema_variables = any(['ExponentialMovingAverage' in tensor_name for tensor_name in all_tensors_names_full])
        #self._all_tensors_dtypes = all_tensors_dtypes
        #self._all_tensors_shapes = all_tensors_shapes
        if name_converter == 'default':
            _name_converter = CheckpointNameConverter()
        elif name_converter is None:
            _name_converter = NullNameConverter()
        elif isinstance(name_converter, WeightsNameConverter):
            _name_converter = name_converter
        else:
            raise ValueError('Invalid name converter')

        super(CheckpointReader, self).__init__(
            filename=checkpoint_prefix, filetype=filetype, all_raw_tensor_names=all_raw_tensors_names, name_converter=_name_converter, load_all_into_memory=True)


    def _read_tensor(self, tensor_name):

        tensor_name_use = tensor_name
        names_skip_ema = ['moving_mean', 'moving_variance'] # these variables do not have 'ExponentialMovingAverage' versions
        if self._exponential_moving_average and \
                not any([tensor_name.endswith(name_skip) for name_skip in names_skip_ema]) and \
                self._have_ema_variables:
            tensor_name_use += '/ExponentialMovingAverage'


        if not self._reader.has_tensor(tensor_name_use):
            print("Tensor %s not found in checkpoint" % tensor_name)
            return

        t = self._reader.get_tensor(tensor_name_use)
        return t



class CheckpointHickleWeightsReader(WeightsReader):
    # Deprecated: If a tensorflow checkpoint is loaded and then saved as a .hkl file,
    # it could be read with this reader.
    def __init__(self, checkpoint_prefix, exponential_moving_average=True):
        self._checkpoint_prefix = checkpoint_prefix
        filetype = 'CheckpointHickle'
        ema_str = '_ema' if exponential_moving_average else ''
        checkpoint_weights_file = checkpoint_prefix + '_weights%s.hkl' % ema_str

        S_hkl = hkl.load(checkpoint_weights_file)
        all_tensors_names = list(S_hkl.keys())
        self._S_hkl = S_hkl

        super(CheckpointHickleWeightsReader, self).__init__(
            filename=checkpoint_prefix, filetype=filetype, all_raw_tensor_names=all_tensors_names, load_all_into_memory=True)

    def _read_tensor(self, tensor_name):

        if not tensor_name in self._S_hkl:
            print("Tensor %s not found in checkpoint weights" % tensor_name)
            return

        t = self._S_hkl[tensor_name]
        return t



class ProtobufWeightsFileReader(WeightsReader):
    # Deprecated: For reading a tensorflow graph from a .pb file,

    def __init__(self, pb_filename, load_all=True):

        if not os.path.exists(pb_filename):
            raise ValueError('File not found: %s' % pb_filename)

        filetype = 'Protobuf (.pb) file'
        all_tensors_names = []

        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.
        with tf.gfile.FastGFile(pb_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        #with tf.Session() as sess:
        #    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #    print(len(all_vars))

        #create_graph(GRAPH_DIR)

        constant_values = {}

        with tf.Session() as sess:
            constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
            for constant_op in constant_ops:
                constant_values[constant_op.name] = sess.run(constant_op.outputs[0])

                all_tensors_names.append(constant_op.name)

        self._constant_values = constant_values

        super(ProtobufWeightsFileReader, self).__init__(
            filename=pb_filename, filetype=filetype, all_tensors_names=all_tensors_names, load_all_into_memory=load_all)


    def _read_tensor(self, tensor_name):

        return self._constant_values[tensor_name]






class H5WeightsFileReader(WeightsReader):
    """
    A weights reader for reading the weights from a .h5 weights file from one of
    the keras.applications networks (e.g. mobilenets, vgg, resnet50)
    """

    def __init__(self, h5_weights_filename, name_converter=None, load_all=True, verbose=True):

        if not os.path.exists(h5_weights_filename):
            raise ValueError('File not found: %s' % h5_weights_filename)

        filetype = 'h5 file'
        allowed_extensions = ['.h5']
        file_ext = os.path.splitext(h5_weights_filename)[1]
        if not file_ext in allowed_extensions:
            raise ValueError('Unrecognized file extension : %s' % file_ext)

        f = h5py.File(h5_weights_filename, 'r')
        use_old_method=False
        if use_old_method:
            all_tensors_names = []

            all_file_keys = list(f.keys())
            for k in all_file_keys:
                h5_group = f[k]
                if len(h5_group) == 0:
                    continue

                try:
                    h5_subgroup = f[k][k] #mobilenetv1, mobilenetv2
                    self._double_subscript = True
                except:
                    h5_subgroup = f[k] # VGG
                    self._double_subscript = False

                weight_names = h5_subgroup.keys()
                weight_names_ext = [k + '/' + name for name in weight_names]
                all_tensors_names.extend(weight_names_ext)

        self._f = f
        if name_converter is None:
            _name_converter = NullNameConverter()
        elif isinstance(name_converter, WeightsNameConverter):
            _name_converter = name_converter
        else:
            raise ValueError('Invalid name converter. Should be a class inheriting from WeightsNameConverter or <None> ')

        self.read_all_h5_dataset_variables()
        all_raw_tensor_names = self._all_tensor_names_h5

        super(H5WeightsFileReader, self).__init__(
            filename=h5_weights_filename, filetype=filetype, all_raw_tensor_names=all_raw_tensor_names,
            name_converter=_name_converter, load_all_into_memory=load_all)


    def _read_tensor(self, raw_tensor_name):
        return self._all_tensors_h5[raw_tensor_name]
        #varname, weightname = tensor_name.split('/')
        #if self._double_subscript:
        #    x = np.array(self._f[varname][varname][weightname])
        #else:
        #    x = np.array(self._f[varname][weightname])
        #return x


    def read_all_h5_dataset_variables(self):
        # Initialize attributes to be populated
        self._all_tensor_names_h5 = []
        self._all_tensors_h5 = {}

        # Recursive call to walk through the h5 file variable tree and retreive all variables:
        self.get_all_h5_data_helper(self._f, '')
        return self._all_tensor_names_h5

    def get_all_h5_data_helper(self, f_cur, cur_str_prefix):

        if isinstance(f_cur, h5py._hl.dataset.Dataset):
            tensor_name = cur_str_prefix[1:].replace(':0', '')
            #tensor_name_use = tensor_name
            #if self._name_converter is not None:
            #    tensor_name_use = self._name_converter.convert_name(tensor_name)

            tensor_val = np.array(f_cur)
            assert tensor_val is not None
            self._all_tensor_names_h5.append(tensor_name)
            self._all_tensors_h5[tensor_name] = tensor_val

        elif isinstance(f_cur, h5py._hl.group.Group):
            all_cur_keys = f_cur.keys()
            for k in all_cur_keys:
                self.get_all_h5_data_helper(f_cur[k], cur_str_prefix + '/' + k)



class ModelWeightsReader(WeightsReader):
    # Class to read weights from an existing gtcModel in memory.
    # Allows easy copying of weights from one gtc model to another, using the
    # weights reader API.
    def __init__(self, model, verbose=True, sess=None):

        if sess is None:
            sess = tf.get_default_session()

        if hasattr(model, '_get_weights'):  # for a GTCModel
            all_wgts = model._get_weights(hp=True, lp=True)
            weight_vals = sess.run(all_wgts)
            all_raw_tensor_names = [wgt.name.replace(':0', '') for wgt in all_wgts]

        elif isinstance(model, keras.models.Model):
            weight_vals = model.get_weights()
            all_weights_symbolic = model.weights
            all_raw_tensor_names = [w.name.replace(':0', '') for w in all_weights_symbolic]
        else:
            raise ValueError('Unsupported model type')

        self.weights_dict = OrderedDict( zip(all_raw_tensor_names, weight_vals))
        filename = ''
        if hasattr(model, '_name'):
            filename = model._name

        super(ModelWeightsReader, self).__init__(
            filename=filename, filetype='gtcModel', all_raw_tensor_names=all_raw_tensor_names)


    def _read_tensor(self, raw_tensor_name):
        return self.weights_dict[raw_tensor_name]



################################## Name Converters #############################################
# These classes help convert the names of tensors found with one of the above readers, to make
# the names of tensors found in the reader match the expected weight names in the network
# In most cases, we name the weight names in the network to match the names in the tensorflow
# checkpoints. So for tensorflow checkpoints, only minor adjustments are needed (e.g. removing
# the ":0" at the end of the name). For the keras.applications mobilenet networks, some more
# involved renaming is necessary to get the names to match the tensorflow conventions.
# These Name converters should be initialized and passed to the weights readers, so the readers
# can return the correctly named tensors when requested.


class WeightsNameConverter():   # Parent class
    def __init__(self, verbose=True):
        self._verbose = verbose
        pass

    def convert_name(self, tensor_name):
        raise NotImplementedError('Child class should implement this method')
        pass



class NullNameConverter(WeightsNameConverter):
    # Dummy class that just returns original name.
    def convert_name(self, tensor_name):
        return tensor_name  # Do nothing, return original name



class KerasAppsWeightsNameConverter(WeightsNameConverter):
    # Converts the names of the weights in the keras.applications networks
    # mobilenet_[v1, v2], VGG[16, 19], ResNet_[18, 34, 50, 101, 152]

    def __init__(self, model_name, verbose=True, use_resnet50_fchollet=False):
        if model_name == 'mobilenet_v1':
            self._prefix = 'MobilenetV1/'
            self._converter = self._convert_name_mobilenet_v1
        elif model_name == 'mobilenet_v2':
            self._prefix = 'MobilenetV2/'
            self._converter = self._convert_name_mobilenet_v2
        elif 'resnet' in model_name.lower():
            use_fchollet = False
            self._prefix = model_name.title() + '/'
            if use_resnet50_fchollet:
                self._converter = self._convert_name_resnet_fchollet_weights
            else:
                self._converter = self._convert_name_resnet
        elif 'vgg' in model_name.lower() or 'resnet' in model_name.lower():
            self._prefix = ''
            self._converter = self._convert_name_vgg
        else:
            raise NotImplementedError('converter for model %s is not yet defined' % model_name)

        super(KerasAppsWeightsNameConverter, self).__init__(verbose)


    def convert_name(self, tensor_name):
        return self._converter(tensor_name)

    def _convert_name_mobilenet_v1(self, tensor_name):
        # sample raw tensor names and their converted names:
        #   initial filters:   'conv1/conv1/kernel'                    -> 'MobilenetV1/Conv2d_0/kernel'
        #   initial batchnorm: 'conv1_bn/conv1_bn/beta'                -> 'MobilenetV1/Conv2d_0/BatchNorm/beta'
        #   depthwise conv:    'conv_dw_1/conv_dw_1/depthwise_kernel'  -> 'MobilenetV1/Conv2d_1_depthwise/depthwise_kernel'
        #   pointwise conv:    'conv_pw_1/conv_pw_1/kernel'            -> 'MobilenetV1/Conv2d_1_pointwise/kernel'
        #   logits:            'conv_preds/conv_preds/kernel'          -> 'MobilenetV1/Logits/Conv2d_1c_1x1/kernel'

        prefix = self._prefix

        varname = tensor_name.split('/')[-1]
        init_filters_match = re.search('conv(\d+)[a-zA-Z_]*/', tensor_name)                  # 'conv1/', 'conv1_bn'/  -> initial conv / bn
        separable_filters_match = re.search('conv_([a-zA-Z]+)_(\d+)[a-zA-Z_]*/', tensor_name)  # 'conv_dw_1'/'conv_pw_1_bn'  -> depthwise/pointwise conv/bn
        logits_match = re.search('conv_preds', tensor_name)                                  # 'conv_preds'  -> final logits

        is_batchnorm = '_bn/' in tensor_name

        if init_filters_match is not None:

            src_name = 'conv' + init_filters_match.groups()[0]
            dst_name = 'Conv2d_0'

        elif separable_filters_match is not None:
            conv_type = separable_filters_match.groups()[0]
            conv_id = int(separable_filters_match.groups()[1])
            conv_type_explicit = {'dw':'depthwise', 'pw':'pointwise'}[conv_type]

            src_name = 'conv_%s_%d' % (conv_type, conv_id)  # '/conv3_dw_10/conv3_dw_10//depthwise_kernel:0
            dst_name =  'Conv2d_%d_%s' % (conv_id, conv_type_explicit)

        elif logits_match is not None:
            src_name = 'conv_preds'
            dst_name = 'Logits/Conv2d_1c_1x1'

        else:
            raise ValueError('unrecognized tensor name : %s ' % tensor_name)

        if is_batchnorm:
            src_name += '_bn'
            dst_name += '/BatchNorm'

        # Verify that can reconstruct source name correctly (sanity check) :

        src_final_name = src_name + '/' + src_name + '/' + varname

        assert src_final_name == tensor_name, "Mismatch between : \n      original name (%s) and \n reconstructed name (%s)" % \
                                              (tensor_name, src_final_name)

        # TARGET NAME
        dst_final_name = prefix + dst_name + '/' + varname

        if self._verbose:
            print( '%50s  --> %s ' % (src_final_name, dst_final_name))
        return dst_final_name


    def _convert_name_mobilenet_v2(self, tensor_name):
        # sample tensor names:
        #  initial filters:   'Conv1/Conv1/kernel'                      -> 'MobilenetV2/Conv/kernel'
        #  initial batchnorm: 'bn_Conv1/bn_Conv1/beta'                  -> 'MobilenetV2/Conv/BatchNorm/beta'
        #  final filters:     'Conv_1/Conv_1/kernel'                    -> 'MobilenetV2/Conv_1/kernel'
        #  final batchnorm:   'Conv_1_bn/Conv_1_bn/beta'                -> 'MobilenetV2/Conv_1/BatchNorm/beta'

        #  expand conv:       'mobl1_conv_1_expand/mobl1_conv_1_expand/kernel'                  -> 'MobilenetV2/expanded_conv_1/expand/kernel'
        #  depthwise conv:    'mobl1_conv_1_depthwise/mobl1_conv_1_depthwise/depthwise_kernel', -> 'MobilenetV2/expanded_conv_1/depthwise/depthwise_kernel'
        #  project conv:      'mobl1_conv_1_project/mobl1_conv_1_project/kernel'                -> 'MobilenetV2/expanded_conv_1/project/kernel'
        #  expand bn:         'bn2_conv_2_bn_expand/bn2_conv_2_bn_expand/beta'        -> 'MobilenetV2/expanded_conv_2/expand/BatchNorm/beta'
        #  depthwise bn:      'bn2_conv_2_bn_depthwise/bn2_conv_2_bn_depthwise/beta', -> 'MobilenetV2/expanded_conv_2/depthwise/BatchNorm/beta'
        #  project bn:        'bn2_conv_2_bn_project/bn2_conv_2_bn_project/beta'      -> 'MobilenetV2/expanded_conv_2/project/BatchNorm/beta'

        #  expand conv:       'block_1_expand/block_1_expand/kernel'                  -> 'MobilenetV2/expanded_conv_1/expand/kernel'
        #  depthwise conv:    'block_1_depthwise/block_1_depthwise/depthwise_kernel', -> 'MobilenetV2/expanded_conv_1/depthwise/depthwise_kernel'
        #  project conv:      'block_1_project/block_1_project/kernel'                -> 'MobilenetV2/expanded_conv_1/project/kernel'
        #  expand bn:         'block_1_expand_BN/block_1_expand_BN/beta'              -> 'MobilenetV2/expanded_conv_2/expand/BatchNorm/beta'
        #  depthwise bn:      'block_1_depthwise_BN/block_1_depthwise_BN/beta',       -> 'MobilenetV2/expanded_conv_2/depthwise/BatchNorm/beta'
        #  project bn:        'block_1_project_BN/block_1_project_BN'                 -> 'MobilenetV2/expanded_conv_2/project/BatchNorm/beta'

        #  logits:            'Logits/Logits/kernel'

        repeat_name = True
        prefix = self._prefix

        varname = tensor_name.split('/')[-1]
        init_filters_match = re.search('Conv(\d+)[a-zA-Z_]*/', tensor_name)                # 'Conv1/', 'bn_Conv1'/
        final_filters_match = re.search('Conv_(\d+)[a-zA-Z_]*', tensor_name)               # 'Conv_1/', 'Conv1_bn'/
        res_blocks_conv_match = re.search('mobl(\d+)_conv_\d+_([a-zA-Z]*)/', tensor_name)  # 'mobl12_conv_12_expand/'
        res_blocks_bn_match = re.search('bn(\d+)_conv_\d+_bn_([a-zA-Z]*)/', tensor_name)   # 'bn2_conv_2_bn_expand/'

        res_blocks_conv_match_v2 = re.search('block_(\d+)_([a-zA-Z]*)/', tensor_name)      # 'block_1_expand/'
        res_blocks_bn_match_v2 = re.search('block_(\d+)_([a-zA-Z]*)_BN/', tensor_name)     # 'block_1_expand_BN/'
        res_blocks_conv_match0_v2 = re.search('expanded_conv_([a-zA-Z]*)/', tensor_name)      # 'expanded_conv_project/'
        res_blocks_bn_match0_v2 = re.search('expanded_conv_([a-zA-Z]*)_BN/', tensor_name)  # 'expanded_conv_project_BN/'

        logits_match = re.search('Logits/', tensor_name)                                    # 'conv_preds'  -> final logits
        logits_1x1_match = re.search('Logits/Conv2d_1c_1x1/', tensor_name)
        is_batchnorm = 'bn' in tensor_name

        if init_filters_match is not None:

            src_name = 'Conv' + init_filters_match.groups()[0]
            dst_name = 'Conv'
            if is_batchnorm:
                src_name = 'bn_' + src_name
                dst_name = dst_name + '/BatchNorm'

        elif final_filters_match is not None:
            src_name = 'Conv_' + final_filters_match.groups()[0]
            dst_name = 'Conv_' + final_filters_match.groups()[0]
            if is_batchnorm:
                src_name = src_name + '_bn'
                dst_name = dst_name + '/BatchNorm'

        elif res_blocks_conv_match or res_blocks_conv_match_v2 or res_blocks_conv_match0_v2:

            if res_blocks_conv_match is not None:
                conv_id = int(res_blocks_conv_match.groups()[0])
                conv_type = res_blocks_conv_match.groups()[1]
                src_name = 'mobl%d_conv_%d_%s' % (conv_id, conv_id, conv_type)
            elif res_blocks_conv_match_v2 is not None:
                conv_id = int(res_blocks_conv_match_v2.groups()[0])
                conv_type = res_blocks_conv_match_v2.groups()[1]
                src_name = 'block_%d_%s' % (conv_id, conv_type)
            elif res_blocks_conv_match0_v2 is not None:
                conv_id = 0
                conv_type = res_blocks_conv_match0_v2.groups()[0]
                src_name = 'expanded_conv_%s' % conv_type
            else:
                raise ValueError('error')

            expanded_conv_id = '' if conv_id == 0 else '_%d' % conv_id
            dst_name = 'expanded_conv%s/%s' % (expanded_conv_id, conv_type)

        elif res_blocks_bn_match or res_blocks_bn_match_v2 or res_blocks_bn_match0_v2:

            if res_blocks_bn_match is not None:
                conv_id = int(res_blocks_bn_match.groups()[0])
                conv_type = res_blocks_bn_match.groups()[1]
                src_name = 'bn%d_conv_%d_bn_%s' % (conv_id, conv_id, conv_type)
            elif res_blocks_bn_match_v2 is not None:
                conv_id = int(res_blocks_bn_match_v2.groups()[0])
                conv_type = res_blocks_bn_match_v2.groups()[1]
                src_name = 'block_%d_%s_BN' % (conv_id, conv_type)
            elif res_blocks_bn_match0_v2 is not None:
                conv_id = 0
                conv_type = res_blocks_bn_match0_v2.groups()[0]
                src_name = 'expanded_conv_%s_BN' % conv_type
            else:
                raise ValueError('error')

            expanded_conv_id = '' if conv_id == 0 else '_%d' % conv_id
            dst_name = 'expanded_conv%s/%s/BatchNorm' % (expanded_conv_id, conv_type)

        elif logits_match or logits_1x1_match :
            if logits_1x1_match:
                src_name = 'Logits/Conv2d_1c_1x1'
            elif logits_match:
                src_name = 'Logits'
            else:
                raise ValueError('error!')
            dst_name = 'Logits/Conv2d_1c_1x1'

        else:
            raise ValueError('unrecognized tensor name : %s ' % tensor_name)


        # Verify that source matches up:
        src_final_name = src_name + '/' + src_name + '/' + varname


        # TARGET NAME
        final_name = dst_name

        dst_final_name = prefix + final_name + '/' + varname

        assert src_final_name == tensor_name, "Mismatch between : \n      original name (%s) and \n reconstructed name (%s)" % (tensor_name,src_final_name)

        if self._verbose:
            print( '%50s  --> %s ' % (src_final_name, dst_final_name))
        return dst_final_name

    def _convert_name_resnet_fchollet_weights(self, tensor_name):
        # ResNet:
        # initial conv:             conv1/conv1_W           -> Resnet50/conv0
        # initial batchnorm:     bn_conv1/bn_conv1_beta     -> Resnet50/bn0
        # block conv:   res2a_branch2a/res2a_branch2a_W      -> Resnet50/stage1_unit1_conv1/kernel
        # block bn:     bn2a_branch2a/bn2a_branch2a_beta    -> Resnet50/stage1_unit1_bn1/beta
        # shortcut conv:    res2a_branch1/res2a_branch1_W   -> 'Resnet18/stage1_unit1_sc/kernel'
        # shortcut bn:      bn2a_branch1/bn2a_branch1_beta  -> 'Resnet18/stage1_unit1_sc/kernel' ???
        # logits:                   fc1000/fc1000_W         -> Resnet50/Logits/kernel

        variable_renames = {'W': 'kernel',
                            'b': 'bias',
                            'beta': 'beta',
                            'gamma': 'gamma',
                            'running_mean': 'moving_mean',
                            'running_std': 'moving_variance'}
        for src, dst in variable_renames.items():
            src_name = '_' + src + '[_0-9]*$'  # match to '_W', '_W_1', or '_W_2'
            dst_name = '/' + dst
            tensor_name = re.sub(src_name, dst_name, tensor_name)

        prefix = self._prefix

        varname = tensor_name.split('/')[-1]
        init_filters_match = re.search('conv(\d+)/', tensor_name)                # 'conv1/',
        #init_bn_match = re.search('bn_conv(\d+)/', tensor_name)                # 'bn_conv1/')
        block_conv_match = re.search('res(\d)([a-z])_branch(\d)([a-z]*)', tensor_name)  # 'res2a_branch2a/'
        block_bn_match = re.search('bn(\d)([a-z])_branch(\d)([a-z]*)', tensor_name)   # 'bn2_conv_2_bn_expand/'
        logits_match = re.search('fc1000', tensor_name)
        is_batchnorm = 'bn' in tensor_name

        if init_filters_match is not None:
            conv_id = int(init_filters_match.groups()[0])
            src_name = 'conv%d' % conv_id
            dst_name = 'conv%d' % (conv_id-1)
            if is_batchnorm:
                src_name = 'bn_' + src_name
                dst_name = 'bn%d' % (conv_id-1)

        elif block_conv_match is not None or block_bn_match is not None:
            if block_conv_match:
                src_prefix = 'res'
                dst_prefix = 'conv'
                block_match = block_conv_match
            else:
                src_prefix = 'bn'
                dst_prefix = 'bn'
                block_match = block_bn_match

            stage_str, unit_letter, branch, conv_letter = block_match.groups()
            stage_id = int(stage_str)-1
            unit_id = ord(unit_letter)-ord('a')+1
            branch_id = int(branch)

            if branch_id == 1: # shortcut connection
                dst_suffix = '_bn' if dst_prefix == 'bn' else ''
                src_name = '%s%s%s_branch%s' % (src_prefix, stage_str, unit_letter, branch)
                dst_name = 'stage%d_unit%d_sc%s' % (stage_id, unit_id, dst_suffix)

            elif branch_id == 2: # standard conv blocks
                conv_id = ord(conv_letter)-ord('a')+1 if conv_letter else None

                src_name = '%s%s%s_branch%s%s' % (src_prefix, stage_str, unit_letter, branch, conv_letter)
                dst_name = 'stage%d_unit%d_%s%s' % (stage_id, unit_id, dst_prefix, conv_id)
            else:
                raise ValueError('unexpected branch : %s' % branch)


        elif logits_match:
            src_name = 'fc1000'
            dst_name = 'Logits'

        else:
            raise ValueError('unrecognized tensor name : %s ' % tensor_name)


        # Verify that source matches up:
        src_final_name = src_name + '/' + src_name + '/' + varname


        # TARGET NAME
        final_name = dst_name

        dst_final_name = prefix + final_name + '/' + varname

        assert src_final_name == tensor_name, "Mismatch between : \n      original name (%s) and \n reconstructed name (%s)" % (tensor_name,src_final_name)

        if self._verbose:
            print( '%50s  --> %s ' % (src_final_name, dst_final_name))
        return dst_final_name



    def _convert_name_resnet(self, tensor_name):
        # ResNet:
        # initial conv:             conv1/conv1_W           -> Resnet50/conv0
        # initial batchnorm:     bn_conv1/bn_conv1_beta     -> Resnet50/bn0
        # block conv:   res2a_branch2a/res2a_branch2a_W      -> Resnet50/stage1_unit1_conv1/kernel
        # block bn:     bn2a_branch2a/bn2a_branch2a_beta    -> Resnet50/stage1_unit1_bn1/beta
        # shortcut conv:    res2a_branch1/res2a_branch1_W   -> 'Resnet18/stage1_unit1_sc/kernel'
        # shortcut bn:      bn2a_branch1/bn2a_branch1_beta  -> 'Resnet18/stage1_unit1_sc/kernel' ???
        # logits:                   fc1000/fc1000_W         -> Resnet50/Logits/kernel

        variable_renames = {'W': 'kernel',
                            'b': 'bias',
                            'beta': 'beta',
                            'gamma': 'gamma',
                            'running_mean': 'moving_mean',
                            'running_std': 'moving_variance'}
        for src, dst in variable_renames.items():
            src_name = '_' + src + '[_0-9]*$'  # match to '_W', '_W_1', or '_W_2'
            dst_name = '/' + dst
            tensor_name = re.sub(src_name, dst_name, tensor_name)

        prefix = self._prefix

        varname = tensor_name.split('/')[-1]
        init_filters_match = re.search('conv(\d+)/', tensor_name)                # 'conv1/',
        #init_bn_match = re.search('bn_conv(\d+)/', tensor_name)                # 'bn_conv1/')
        block_conv_match = re.search('res(\d)([a-z])_branch(\d)([a-z]*)', tensor_name)  # 'res2a_branch2a/'
        block_bn_match = re.search('bn(\d)([a-z])_branch(\d)([a-z]*)', tensor_name)   # 'bn2_conv_2_bn_expand/'
        logits_match = re.search('fc1000', tensor_name)
        is_batchnorm = 'bn' in tensor_name

        if init_filters_match is not None:
            conv_id = int(init_filters_match.groups()[0])
            src_name = 'conv%d' % conv_id
            dst_name = 'conv%d' % (conv_id-1)
            if is_batchnorm:
                src_name = 'bn_' + src_name
                dst_name = 'bn%d' % (conv_id-1)

        elif block_conv_match is not None or block_bn_match is not None:
            if block_conv_match:
                src_prefix = 'res'
                dst_prefix = 'conv'
                block_match = block_conv_match
            else:
                src_prefix = 'bn'
                dst_prefix = 'bn'
                block_match = block_bn_match

            stage_str, unit_letter, branch, conv_letter = block_match.groups()
            stage_id = int(stage_str)-1
            unit_id = ord(unit_letter)-ord('a')+1
            branch_id = int(branch)

            if branch_id == 1: # shortcut connection
                dst_suffix = '_bn' if dst_prefix == 'bn' else ''
                src_name = '%s%s%s_branch%s' % (src_prefix, stage_str, unit_letter, branch)
                dst_name = 'stage%d_unit%d_sc%s' % (stage_id, unit_id, dst_suffix)

            elif branch_id == 2: # standard conv blocks
                conv_id = ord(conv_letter)-ord('a')+1 if conv_letter else None

                src_name = '%s%s%s_branch%s%s' % (src_prefix, stage_str, unit_letter, branch, conv_letter)
                dst_name = 'stage%d_unit%d_%s%s' % (stage_id, unit_id, dst_prefix, conv_id)
            else:
                raise ValueError('unexpected branch : %s' % branch)


        elif logits_match:
            src_name = 'fc1000'
            dst_name = 'Logits'

        else:
            raise ValueError('unrecognized tensor name : %s ' % tensor_name)


        # Verify that source matches up:
        src_final_name = src_name + '/' + src_name + '/' + varname


        # TARGET NAME
        final_name = dst_name

        dst_final_name = prefix + final_name + '/' + varname

        assert src_final_name == tensor_name, "Mismatch between : \n      original name (%s) and \n reconstructed name (%s)" % (tensor_name,src_final_name)

        if self._verbose:
            print( '%50s  --> %s ' % (src_final_name, dst_final_name))
        return dst_final_name


    def _convert_name_vgg(self, tensor_name):
        # VGG:
        # block1_conv1/block1_conv1_W_1  ->  block1_conv1/block1_conv1/kernel
        # fc1/fc1_W_1                    ->  fc6/fc6/kernel
        # predictions/predictions_W_1     ->  predictions/predictions/kernel

        variable_renames = {'W': 'kernel',
                            'b': 'bias',
                            'beta': 'beta',
                            'gamma': 'gamma',
                            'running_mean': 'moving_mean',
                            'running_std': 'moving_variance'}

        for src, dst in variable_renames.items():

            src_name = '_' + src + '[_0-9]*$'  # match to '_W', '_W_1', or '_W_2'
            dst_name = '/' + dst

            tensor_name = re.sub(src_name, dst_name, tensor_name)

            fc_match = re.search('fc(\d+)/', tensor_name)    # 'fc1/', 'fc2/'
            if fc_match is not None:
                fc_offset = 5  # fc1 --> fc6.   fc2 --> fc7
                fc_src_id = fc_match.groups()[0]             # e.g. fc1
                fc_dst_id = str(int(fc_src_id) + fc_offset)  # e.g. fc6
                tensor_name = tensor_name.replace('fc' + fc_src_id, 'fc' + fc_dst_id)



        return tensor_name


    def _no_change(self, tensor_name):
        return tensor_name



def rep_with_id(s, id):
    id_str = '_' + str(id) if id is not None else ''
    return s + '/' + s + id_str


class ImageClassifiersWeightsNameConverter(WeightsNameConverter):
    # Converts the names of the weights in the image-classifiers networks
    # So far: implemented Resnets ResNet_[18, 34, 50, 101, 152]
    # Others that might be useful: mobilenet v1,v2, vgg

    def __init__(self, model_name, verbose=True):
        if 'resnet' in model_name:
            self._prefix = model_name.title() + '/'
            self._converter = self._convert_name_resnet
        else:
            raise NotImplementedError('converter for model %s is not yet defined' % model_name)

        super(ImageClassifiersWeightsNameConverter, self).__init__(verbose)


    def convert_name(self, tensor_name):
        return self._converter(tensor_name)

    def _convert_name_resnet(self, tensor_name):
        # sample raw tensor names and their converted names:
        #   input batchnorm:         'bn_data/bn_data_33/beta'                 -> 'Resnet18/bn_data/beta'
        #   initial/final filters:   'conv0/conv0_33/kernel'                 -> 'Resnet18/conv0/kernel'
        #   initial/final batchnorm: 'bn0/bn0_33/beta'                       -> 'Resnet18/bn0/beta'
        #   block conv:        'stage1_unit1_conv1/stage1_unit1_conv1_32/kernel'  -> 'Resnet18/stage1_unit1_conv1/kernel'
        #   block bn:          'stage1_unit1_bn1/stage1_unit1_bn1_32/beta'        -> 'Resnet18/stage1_unit1_bn1/beta'
        #   final bn:          'bn1/bn1_31/beta'                        -> 'Resnet18/bn1/beta'
        #   logits:            'fc1/fc1_26/kernel'                      -> 'Resnet18/logits/kernel'

        prefix = self._prefix

        varname = tensor_name.split('/')[-1]
        bn_data_match = re.search('bn_data/bn_data_(\d+)/', tensor_name)    # 'conv0/', 'conv1/'
        filters_match = re.search('conv(\d+)/conv(\d+)_(\d+)/', tensor_name)    # 'conv0/', 'conv1/'
        bn_match      = re.search('bn(\d+)/bn(\d+)_(\d+)', tensor_name)         # 'bn0/', 'bn1/'
        block_match = re.search('stage(\d+)_unit(\d+)_(\w+)(\d*)/'
                                'stage(\d+)_unit(\d+)_(\w+)(\d*)_(\d+)/', tensor_name)  # 'stage1_unit1_conv1'  -> depthwise/pointwise conv/bn
        logits_match = re.search('fc1/fc1_(\d+)/', tensor_name)                 # 'conv_preds'  -> final logits

        is_batchnorm = '_bn/' in tensor_name

        if bn_data_match is not None:
            dummy_id = bn_data_match.groups()[0]
            src_name = 'bn_data'
            dst_name = src_name

        elif filters_match is not None:

            dummy_id = filters_match.groups()[2]
            src_name = 'conv' + filters_match.groups()[0]
            dst_name = src_name

        elif bn_match is not None:

            dummy_id = bn_match.groups()[2]
            src_name = 'bn' + bn_match.groups()[0]
            dst_name = src_name

        elif block_match is not None:

            stage_id, unit_id, layer_type, layer_id, _, _, _, _, dummy_id = block_match.groups()

            src_name = 'stage%s_unit%s_%s%s' % (stage_id, unit_id, layer_type, layer_id)
            dst_name = src_name

        elif logits_match is not None:
            dummy_id = logits_match.groups()[0]
            src_name = 'fc1'
            dst_name = 'Logits'

        else:
            raise ValueError('unrecognized tensor name : %s ' % tensor_name)

        if is_batchnorm:
            #src_name += '_bn'
            dst_name += '/BatchNorm'

        # Verify that can reconstruct source name correctly (sanity check) :
        src_final_name = rep_with_id(src_name, dummy_id) + '/' + varname

        assert src_final_name == tensor_name, "Mismatch between : \n      original name (%s) and \n reconstructed name (%s)" % \
                                              (tensor_name, src_final_name)

        # TARGET NAME
        dst_final_name = prefix + dst_name + '/' + varname

        if self._verbose:
            print( '%50s  --> %s ' % (src_final_name, dst_final_name))
        return dst_final_name



    def _no_change(self, tensor_name):
        return tensor_name





class MobilenetSSDKerasWeightsNameConverter(WeightsNameConverter):
    # Converts the names for mobilenet_v1_ssd from the weights file used in this repo:
    # https://github.com/ManishSoni1908/Mobilenet-ssd-keras

    def convert_name(self, tensor_name):
        # Initial filters:     conv0_v1/conv0/kernel      --> FeatureExtractor/MobilenetV1/Conv2d_0/kernel
        # Pointwise filters    conv1_v1/conv1/kernel      --> FeatureExtractor/MobilenetV1/Conv2d_1_pointwise/kernel'
        # Initial batchnorm:   conv0/bn_v1/conv0/bn/beta  --> FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/beta
        # Backbone Depthwise filters:   conv1/dw_v1/conv1/dw/depthwise_kernel  --> FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/depthwise_kernel
        # Backbone Depthwise batchnorm: conv1/dw/bn_v1/conv1/dw/bn/beta  --> FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta
        # Additional Filters     'conv14_1_v1/conv14_1/kernel  --> FeatureExtractor/MobilenetV1/Conv2d_14_1_pointwise/kernel
        # Additional Batchnorm   'conv14_1/bn_v1/conv14_1/bn/beta  --> FeatureExtractor/MobilenetV1/Conv2d_14_1_pointwise/BatchNorm/beta '

        # Box ClassPredictors      conv11_mbox_conf_v1/conv11_mbox_conf/bias  -->    BoxPredictor_0/ClassPredictor/bias
        #                          conv14_2_mbox_conf_v1/conv14_2_mbox_conf/bias --> BoxPredictor_2/ClassPredictor/bias
        # Box LocationPredictors   conv11_mbox_loc_v1/conv11_mbox_loc/bias    -->   BoxPredictor_0/BoxEncodingPredictor/bias


        prefix = 'FeatureExtractor/MobilenetV1/'
        varname = tensor_name.split('/')[-1]

        filters_match   = re.search('conv(\d+)_v1/conv(\d+)/', tensor_name)
        batchnorm_match = re.search('conv(\d+)/bn_v1/conv(\d+)/bn/', tensor_name)
        depthwise_filters_match = re.search('conv(\d+)/dw_v1/conv(\d+)/dw/', tensor_name)
        depthwise_batchnorm_match = re.search('conv(\d+)/dw/bn_v1/conv(\d+)/dw/bn/', tensor_name)
        additional_filters_match = re.search('conv(\d+)_(\d+)_v1/conv(\d+)_(\d+)/', tensor_name)
        additional_batchnorm_match = re.search('conv(\d+)_(\d+)/bn_v1/conv(\d+)_(\d+)/bn/', tensor_name)

        additional_filters_box_match = re.search
        box_match = re.search('conv([_\d]+)_mbox_([a-z]*)_v1/conv([\d_]+)_mbox_[a-z]*/', tensor_name)

        is_batchnorm = '/bn' in tensor_name
        bn_str = '/bn' if is_batchnorm else ''
        bn_dst_str = '/BatchNorm' if is_batchnorm else ''
        conv_id_extra_start = 13

        if filters_match is not None or batchnorm_match is not None:
            match = filters_match if filters_match is not None else batchnorm_match
            conv_id = int(match.groups()[0])
            is_pointwise = conv_id > 0

            src_name = 'conv%d%s_v1/conv%d%s' % (conv_id, bn_str, conv_id, bn_str)

            if is_pointwise:
                dst_name = 'Conv2d_%d_pointwise%s' % (conv_id, bn_dst_str)
            else:
                dst_name = 'Conv2d_%d%s' % (conv_id, bn_dst_str)

        elif depthwise_filters_match is not None or depthwise_batchnorm_match is not None:
            match = depthwise_filters_match if depthwise_filters_match is not None else depthwise_batchnorm_match
            conv_id = int(match.groups()[0])
            src_name = 'conv%d/dw%s_v1/conv%d/dw%s' % (conv_id, bn_str, conv_id, bn_str)

            dst_name = 'Conv2d_%d_depthwise%s' % (conv_id, bn_dst_str)

        elif additional_filters_match is not None or additional_batchnorm_match is not None:
            match = additional_filters_match if additional_filters_match is not None else additional_batchnorm_match
            conv_id, stage_id = int(match.groups()[0]), int(match.groups()[1])
            src_name = 'conv%d_%d%s_v1/conv%d_%d%s' % (conv_id, stage_id, bn_str, conv_id, stage_id, bn_str)

            dst_name = 'Conv2d_%d_pointwise_%d_Conv2d_%d%s' % (conv_id_extra_start, stage_id, conv_id-conv_id_extra_start+1, bn_dst_str)

        elif box_match:
            conv_id_str, box_type = box_match.groups()[:2]
            if '_' in conv_id_str:
                conv_id_match = re.search('(\d+)_(\d+)', conv_id_str)
                conv_id, step_id = int(conv_id_match.groups()[0]), int(conv_id_match.groups()[0])
            else:
                conv_id, step_id = int(conv_id_str), None

            src_name = 'conv%s_mbox_%s_v1/conv%s_mbox_%s' % (conv_id_str, box_type, conv_id_str, box_type)

            dst_box_str = {'conf': 'ClassPredictor', 'loc': 'BoxEncodingPredictor'}[box_type]
            prefix = ''
            dst_name = 'BoxPredictor_%d/%s' % (max(conv_id-conv_id_extra_start+1,0), dst_box_str)
        else:
            raise ValueError('Unmatched tensor name %s' % tensor_name)

        src_final_name = src_name + '/' + varname
        assert src_final_name == tensor_name, "Mismatch between : \n      original name (%s) and \n reconstructed name (%s)" % \
                                              (tensor_name, src_final_name)


        dst_final_name = prefix + dst_name + '/' + varname
        if self._verbose:
            print('%40s  --> %s ' % (src_final_name, dst_final_name))
        return dst_final_name


class VGGSSDWeightsNameConverter(WeightsNameConverter):
    def convert_name(self, tensor_name):
        #  initial filters:              'conv1_1/conv_1_1/kernel'  -> 'VGG16/block1_conv1/kernel'
        #  fully_connected layers:       'fc6/fc6/kernel'           -> 'VGG16/block1_conv1/kernel'
        #  L2 normalization layer:       'conv4_3_norm/weights'     -> 'VGG16/block4_conv3/gamma_weights'
        #  box class predictors:         'conv6_2_mbox_conf         -> 'VGG16/BoxPredictor_0/ClassPredictor'
        #  box location predictors:      'conv4_3_norm_mbox_loc     -> 'VGG16/BoxPredictor_0/BoxEncodingPredictor'


        box_ids = ['conv4_3_norm',
                   'fc7',
                   'conv6_2',
                   'conv7_2',
                   'conv8_2',
                   'conv9_2']

        prefix = 'VGG16/'
        varname = tensor_name.split('/')[-1]
        dst_varname = varname
        block_conv_match = re.search('conv(\d+)_(\d+)([a-zA-Z_]*)/', tensor_name)
        fc_conv_match    = re.search('fc(\d+)/', tensor_name)                 # 'fc6/'
        box_pred_match = re.search('([a-zA-Z0-9_]*)_mbox_([a-zA-Z_]*)/', tensor_name)  # 'conv6_2_mbox_conf/' 'conv6_2_mbox_loc/'
        #fc_box_pred_match    = re.search('fc(\d+)_mbox_[a-zA-Z_]*/', tensor_name)  # 'conv6_2_mbox_conf/' 'conv6_2_mbox_loc/'


        if block_conv_match is not None and box_pred_match is None:
            block_id = int(block_conv_match.groups()[0])
            conv_id = int(block_conv_match.groups()[1])
            conv_suffix = block_conv_match.groups()[2] # eg. '_norm'  for 'conv4_3_norm'
            if 'norm' in conv_suffix:
                dst_varname = 'gamma_' + varname.replace('_0', '')  # 'gamma_weights'

            src_name = 'conv%d_%d%s' % (block_id, conv_id, conv_suffix)
            dst_name = 'block%d_conv%d%s' % (block_id, conv_id, conv_suffix)

        elif fc_conv_match and box_pred_match is None:
            fc_id = int(fc_conv_match.groups()[0])
            src_name = 'fc%d' % (fc_id)
            dst_name = 'fc%d' % (fc_id)

        elif box_pred_match is not None:
            box_source_match = box_pred_match.groups()[0]
            box_type = box_pred_match.groups()[1]

            src_name = '%s_mbox_%s' % (box_source_match, box_type)
            box_type_long = {'conf': 'ClassPredictor', 'loc': 'BoxEncodingPredictor'}[box_type]

            box_pred_id = box_ids.index(box_source_match)

            dst_name = 'BoxPredictor_%d/%s' % (box_pred_id, box_type_long)


        else:
            raise ValueError('Unmatched tensor name %s' % tensor_name)

        src_final_name = src_name + '/' + src_name + '/' + varname
        assert src_final_name == tensor_name, "Mismatch between : \n      original name (%s) and \n reconstructed name (%s)" % \
                                              (tensor_name, src_final_name)

        dst_final_name = prefix + dst_name + '/' + dst_varname
        if self._verbose:
            print('%40s  --> %s ' % (src_final_name, dst_final_name))
        return dst_final_name




class MobilenetV1KerasWeightsNameConverter(WeightsNameConverter):
    # Converts the names for mobilenet_v1 from the mobilenet-v1 weights file used in this repo:
    # https://github.com/ManishSoni1908/Mobilenet-ssd-keras

    def convert_name(self, tensor_name):

        #  initial filters:   'Conv1/Conv1/kernel'                      -> 'MobilenetV1/Conv/kernel'
        #  initial batchnorm: 'bn_Conv1/bn_Conv1/beta'                  -> 'MobilenetV1/Conv/BatchNorm/beta'
        #  final filters:     'Conv_1/Conv_1/kernel'                    -> 'MobilenetV1/Conv_1/kernel'
        #  final batchnorm:   'Conv_1_bn/Conv_1_bn/beta'                -> 'MobilenetV1/Conv_1/BatchNorm/beta'

        varname = tensor_name.split('/')[-1]
        init_filters_match = re.search('Conv(\d+)[a-zA-Z_]*/', tensor_name)                # 'Conv1/', 'bn_Conv1'/
        final_filters_match = re.search('Conv_(\d+)[a-zA-Z_]*', tensor_name)               # 'Conv_1/', 'Conv1_bn'/
        logits_match = re.search('Logits', tensor_name)                                    # 'conv_preds'  -> final logits



        #  initial filters:   'Conv1/Conv1/kernel'                      -> 'MobilenetV2/Conv/kernel'
        #  initial batchnorm: 'bn_Conv1/bn_Conv1/beta'                  -> 'MobilenetV2/Conv/BatchNorm/beta'
        #  final filters:     'Conv_1/Conv_1/kernel'                    -> 'MobilenetV2/Conv_1/kernel'
        #  final batchnorm:   'Conv_1_bn/Conv_1_bn/beta'                -> 'MobilenetV2/Conv_1/BatchNorm/beta'
        #  expand conv:       'mobl1_conv_1_expand/mobl1_conv_1_expand/kernel'                  -> 'MobilenetV2/expanded_conv_1/expand/kernel'
        #  depthwise conv:    'mobl1_conv_1_depthwise/mobl1_conv_1_depthwise/depthwise_kernel', -> 'MobilenetV2/expanded_conv_1/depthwise/depthwise_kernel'
        #  project conv:      'mobl1_conv_1_project/mobl1_conv_1_project/kernel'                -> 'MobilenetV2/expanded_conv_1/project/kernel'
        #  expand bn:         'bn2_conv_2_bn_expand/bn2_conv_2_bn_expand/beta'        -> 'MobilenetV2/expanded_conv_2/expand/BatchNorm/beta'
        #  depthwise bn:      'bn2_conv_2_bn_depthwise/bn2_conv_2_bn_depthwise/beta', -> 'MobilenetV2/expanded_conv_2/depthwise/BatchNorm/beta'
        #  project bn:        'bn2_conv_2_bn_project/bn2_conv_2_bn_project/beta'      -> 'MobilenetV2/expanded_conv_2/project/BatchNorm/beta'
        #  logits:            'Logits/Logits/kernel'

        repeat_name = True
        prefix = self._prefix

        varname = tensor_name.split('/')[-1]
        init_filters_match = re.search('Conv(\d+)[a-zA-Z_]*/', tensor_name)                # 'Conv1/', 'bn_Conv1'/
        final_filters_match = re.search('Conv_(\d+)[a-zA-Z_]*', tensor_name)               # 'Conv_1/', 'Conv1_bn'/
        res_blocks_conv_match = re.search('mobl(\d+)_conv_\d+_([a-zA-Z]*)/', tensor_name)  # 'mobl12_conv_12_expand/'
        res_blocks_bn_match = re.search('bn(\d+)_conv_\d+_bn_([a-zA-Z]*)/', tensor_name)   # 'bn2_conv_2_bn_expand/'
        logits_match = re.search('Logits', tensor_name)                                    # 'conv_preds'  -> final logits




        # Initial filters:     conv0_v1/conv0/kernel      --> FeatureExtractor/MobilenetV1/Conv2d_0/kernel
        # Pointwise filters    conv1_v1/conv1/kernel      --> FeatureExtractor/MobilenetV1/Conv2d_1_pointwise/kernel'
        # Initial batchnorm:   conv0/bn_v1/conv0/bn/beta  --> FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/beta
        # Backbone Depthwise filters:   conv1/dw_v1/conv1/dw/depthwise_kernel  --> FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/depthwise_kernel
        # Backbone Depthwise batchnorm: conv1/dw/bn_v1/conv1/dw/bn/beta  --> FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta
        # Additional Filters     'conv14_1_v1/conv14_1/kernel  --> FeatureExtractor/MobilenetV1/Conv2d_14_1_pointwise/kernel
        # Additional Batchnorm   'conv14_1/bn_v1/conv14_1/bn/beta  --> FeatureExtractor/MobilenetV1/Conv2d_14_1_pointwise/BatchNorm/beta '

        # Box ClassPredictors      conv11_mbox_conf_v1/conv11_mbox_conf/bias  -->    BoxPredictor_0/ClassPredictor/bias
        #                          conv14_2_mbox_conf_v1/conv14_2_mbox_conf/bias --> BoxPredictor_2/ClassPredictor/bias
        # Box LocationPredictors   conv11_mbox_loc_v1/conv11_mbox_loc/bias    -->   BoxPredictor_0/BoxEncodingPredictor/bias


        prefix = 'FeatureExtractor/MobilenetV1/'
        varname = tensor_name.split('/')[-1]


        filters_match   = re.search('conv(\d+)_v1/conv(\d+)/', tensor_name)
        batchnorm_match = re.search('conv(\d+)/bn_v1/conv(\d+)/bn/', tensor_name)
        depthwise_filters_match = re.search('conv(\d+)/dw_v1/conv(\d+)/dw/', tensor_name)
        depthwise_batchnorm_match = re.search('conv(\d+)/dw/bn_v1/conv(\d+)/dw/bn/', tensor_name)
        additional_filters_match = re.search('conv(\d+)_(\d+)_v1/conv(\d+)_(\d+)/', tensor_name)
        additional_batchnorm_match = re.search('conv(\d+)_(\d+)/bn_v1/conv(\d+)_(\d+)/bn/', tensor_name)

        additional_filters_box_match = re.search
        box_match = re.search('conv([_\d]+)_mbox_([a-z]*)_v1/conv([\d_]+)_mbox_[a-z]*/', tensor_name)

        is_batchnorm = '/bn' in tensor_name
        bn_str = '/bn' if is_batchnorm else ''
        bn_dst_str = '/BatchNorm' if is_batchnorm else ''
        conv_id_extra_start = 13

        if filters_match is not None or batchnorm_match is not None:
            match = filters_match if filters_match is not None else batchnorm_match
            conv_id = int(match.groups()[0])
            is_pointwise = conv_id > 0

            src_name = 'conv%d%s_v1/conv%d%s' % (conv_id, bn_str, conv_id, bn_str)

            if is_pointwise:
                dst_name = 'Conv2d_%d_pointwise%s' % (conv_id, bn_dst_str)
            else:
                dst_name = 'Conv2d_%d%s' % (conv_id, bn_dst_str)

        elif depthwise_filters_match is not None or depthwise_batchnorm_match is not None:
            match = depthwise_filters_match if depthwise_filters_match is not None else depthwise_batchnorm_match
            conv_id = int(match.groups()[0])
            src_name = 'conv%d/dw%s_v1/conv%d/dw%s' % (conv_id, bn_str, conv_id, bn_str)

            dst_name = 'Conv2d_%d_depthwise%s' % (conv_id, bn_dst_str)

        elif additional_filters_match is not None or additional_batchnorm_match is not None:
            match = additional_filters_match if additional_filters_match is not None else additional_batchnorm_match
            conv_id, stage_id = int(match.groups()[0]), int(match.groups()[1])
            src_name = 'conv%d_%d%s_v1/conv%d_%d%s' % (conv_id, stage_id, bn_str, conv_id, stage_id, bn_str)

            dst_name = 'Conv2d_%d_pointwise_%d_Conv2d_%d%s' % (conv_id_extra_start, stage_id, conv_id-conv_id_extra_start+1, bn_dst_str)

        elif box_match:
            conv_id_str, box_type = box_match.groups()[:2]
            if '_' in conv_id_str:
                conv_id_match = re.search('(\d+)_(\d+)', conv_id_str)
                conv_id, step_id = int(conv_id_match.groups()[0]), int(conv_id_match.groups()[0])
            else:
                conv_id, step_id = int(conv_id_str), None

            src_name = 'conv%s_mbox_%s_v1/conv%s_mbox_%s' % (conv_id_str, box_type, conv_id_str, box_type)

            dst_box_str = {'conf': 'ClassPredictor', 'loc': 'BoxEncodingPredictor'}[box_type]
            prefix = ''
            dst_name = 'BoxPredictor_%d/%s' % (max(conv_id-conv_id_extra_start+1,0), dst_box_str)
        else:
            raise ValueError('Unmatched tensor name %s' % tensor_name)

        src_final_name = src_name + '/' + varname
        assert src_final_name == tensor_name, "Mismatch between : \n      original name (%s) and \n reconstructed name (%s)" % \
                                              (tensor_name, src_final_name)


        dst_final_name = prefix + dst_name + '/' + varname
        if self._verbose:
            print('%40s  --> %s ' % (src_final_name, dst_final_name))
        return dst_final_name






class CheckpointNameConverter(WeightsNameConverter):
    # In initializing keras layers (from which GTC layers inherit), conv layers weights/biases are given the
    # names 'kernel' and 'bias', whereas in tensorflow checkpoints they are typically called 'weights' and 'biases'.
    # This converter switches to the keras convention, and also removes the ':0' that is usually at the end of each
    # tensor name.
    def convert_name(self, tensor_name):
        tensor_name = re.sub(':0$', '', tensor_name)
        tensor_name = re.sub('weights$', 'kernel', tensor_name)
        tensor_name = re.sub('biases$', 'bias', tensor_name)
        return tensor_name




class MobilenetSSD_TF_CheckpointNameConverter(WeightsNameConverter):
    # In addition to the 'weights'->'kernel' and 'biases'->'bias' conversion, the tensorflow mobilenet-SSD models
    # have the number of filters, kernel_sizes and strides encoded into the names of some of the layers. This makes it
    # burdensome to match to other naming systems, and also might be misleading if the details are changed. This removes
    # those details from the names:
    # Eg in Mobilenet-v1-SSD:
    #    'Conv2d_13_pointwise_1_Conv2d_2_1x1_256'     --> 'Conv2d_13_pointwise_1_Conv2d_2'
    #    'Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512'  --> 'Conv2d_13_pointwise_2_Conv2d_2'
    # In mobilenet-v2-SSD:
    #    'layer_19_1_Conv2d_2_1x1_256'      --> 'layer_19_1_Conv2d_2'
    #    'layer_19_2_Conv2d_2_3x3_s2_512    --> 'layer_19_2_Conv2d_2'

    # convert from other naming systems, because that information would have to be incorporated into
    def convert_name(self, tensor_name):
        tensor_name = re.sub(':0$', '', tensor_name)
        tensor_name = re.sub('weights$', 'kernel', tensor_name)
        tensor_name = re.sub('biases$', 'bias', tensor_name)

        mob_ssd_match = None
        if mob_ssd_match is None:
            mob_ssd_match = re.search('/(Conv2d_\d+_pointwise_\d+_Conv2d_\d+)([0-9xs_]*)/', tensor_name) # mobilenet_ssd1

        if 'layer_19' in tensor_name:
            sss = 1
        if mob_ssd_match is None:
            mob_ssd_match = re.search('/(layer_\d+_\d+_Conv2d_\d+)([0-9xs_]*)/', tensor_name) # mobilenet_ssd2 & mobilenet_ssd3

        if mob_ssd_match is None:
            mob_ssd_match = re.search('/(layer_\d+_\d+_Conv2d_\d+)([0-9xs_]*)_depthwise/', tensor_name) # mobilenet_ssd2 & mobilenet_ssd3: depthwise

        if mob_ssd_match is not None:
            str_keep, str_remove = mob_ssd_match.groups()
            tensor_name = tensor_name.replace(str_keep + str_remove, str_keep)

        return tensor_name



if __name__ == "__main__":

    # test a few of the weights readers:
    test_protobuf_reader = False
    if test_protobuf_reader:
        pb_chkpoint_file = 'F:/SRI/bitnet/mobilenet_slim/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_frozen.pb'
        pbReader = ProtobufWeightsFileReader(pb_chkpoint_file)
        pbReader.print_all_tensors()

    test_checkpoint_reader = False
    if test_checkpoint_reader:
        chkpoint_file = utl.pretrained_weights_root() + 'mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt'
        # chkpoint_file = utl.pretrained_weights_root() + 'mobilenet_v2_0.5_224/mobilenet_v2_0.5_224.ckpt'
        # chkpoint_file = utl.pretrained_weights_root() + 'v3-large_224_1.0_float/ema/model-540000'
        # chkpoint_file = utl.pretrained_weights_root() + 'v3-large_224_0.75_float/ema/model-220000'
        # chkpoint_file = utl.pretrained_weights_root() + 'ssd_mobilenet_v1_coco_2018_01_28/model.ckpt'
        # chkpoint_file = utl.pretrained_weights_root() + 'ssdlite_mobilenet_v2_coco_2018_05_09/model.ckpt'
        # chkpoint_file = utl.pretrained_weights_root() + 'ssd_mobilenet_v2_coco_2018_03_29/model.ckpt'
        # chkpoint_file = utl.pretrained_weights_root() + 'ssd_mobilenet_v3_large_coco_2019_08_14/model.ckpt'

        chkReader = CheckpointReader(chkpoint_file)
        chkReader.print_all_tensors(show_raw_names=True)

    test_mobilenet_v1_keras_reader = False
    if test_mobilenet_v1_keras_reader:
        name_converter_m1 = KerasAppsWeightsNameConverter('mobilenet_v1')
        h5_weights_file = 'F:/SRI/bitnet/gtc-tensorflow/pretrained/mobilenet_v1/mobilenet_1_0_224_tf.h5'
        h5Reader_m1 = H5WeightsFileReader(h5_weights_file, name_converter_m1)
        h5Reader_m1.print_all_tensors(show_raw_names=True)

    a = 1
    test_mobilenet_v2_keras_reader = False
    if test_mobilenet_v2_keras_reader:
        name_converter_m2 = KerasAppsWeightsNameConverter('mobilenet_v2')
        h5_weights_file = 'F:/SRI/bitnet/gtc-tensorflow/pretrained/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5'
        h5Reader = H5WeightsFileReader(h5_weights_file, name_converter_m2)
        h5Reader.print_all_tensors(show_raw_names=True)

    test_vgg16_weights_keras_reader = False
    if test_vgg16_weights_keras_reader:
        name_converter_vgg = KerasAppsWeightsNameConverter('vgg16')
        h5_weights_file = 'F:/SRI/bitnet/gtc-tensorflow/pretrained/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
        h5Reader = H5WeightsFileReader(h5_weights_file, name_converter=name_converter_vgg)
        h5Reader.print_all_tensors(show_raw_names=True)

    test_resnet_weights_keras_reader = False
    if test_resnet_weights_keras_reader:
        name_converter_resnet = KerasAppsWeightsNameConverter('resnet50')
        h5_weights_file = 'F:/SRI/bitnet/gtc-tensorflow/pretrained/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        h5Reader = H5WeightsFileReader(h5_weights_file, name_converter=name_converter_resnet)
        h5Reader.print_all_tensors(show_raw_names=True)


    test_mobilenet_v1_ssd_keras_reader = False
    if test_mobilenet_v1_ssd_keras_reader:
        mobilenet_v2_weights_file_local = utl.pretrained_weights_root() + 'ssd_mobilenet_v1_keras/converted_model.h5'
        name_converter_ssd = MobilenetSSDKerasWeightsNameConverter()
        reader = H5WeightsFileReader(mobilenet_v2_weights_file_local, name_converter=name_converter_ssd)
        reader.print_all_tensors(show_raw_names=True)

    test_mobilenet_v1_keras_reader = False
    if test_mobilenet_v1_keras_reader:
        h5_weights_file = 'F:/SRI/bitnet/Mobilenet-ssd-keras/MobilenetWeights/mobilenet.h5'
        h5Reader = H5WeightsFileReader(h5_weights_file)
        h5Reader.print_all_tensors(show_raw_names=True)

    test_resnet_im_classifiers_reader = False
    if test_resnet_im_classifiers_reader:
        num_layers = 18
        name_converter_resnet18 = ImageClassifiersWeightsNameConverter('resnet18')
        h5_weights_file = utl.pretrained_weights_root() + 'resnet%d_imagenet_1000.h5' % num_layers
        h5Reader = H5WeightsFileReader(h5_weights_file, name_converter=name_converter_resnet18)
        h5Reader.print_all_tensors(show_raw_names=True)

    test_ssd300_keras_reader = False
    if test_ssd300_keras_reader:
        base_dir = 'F:/SRI/bitnet/ssd_keras/weights/'
        #h5_weights_file = base_dir + 'VGG_VOC0712_SSD_300x300_iter_120000.h5'
        #h5_weights_file = base_dir + 'VGG_VOC0712Plus_SSD_300x300_iter_240000.h5'
        #h5_weights_file = base_dir + 'VGG_coco_SSD_300x300_iter_400000.h5'
        h5_weights_file = base_dir + 'VGG_ILSVRC_16_layers_fc_reduced.h5'

        name_converter_vgg_ssd = VGGSSDWeightsNameConverter()
        h5Reader = H5WeightsFileReader(h5_weights_file, name_converter=name_converter_vgg_ssd)
        h5Reader.print_all_tensors(show_raw_names=True)


    test_h5_lenet = False
    if test_h5_lenet:
        h5_file = 'F:/SRI/bitnet/test/lenet_keras.h5'
        h5Reader = H5WeightsFileReader(h5_file)
        h5Reader.print_all_tensors()

        h5_file2 = 'F:/SRI/bitnet/test/lenet_hp.h5'
        h5Reader2 = H5WeightsFileReader(h5_file2)
        h5Reader2.print_all_tensors()

        h5_file3 = 'F:/SRI/bitnet/test/lenet_lp.h5'
        h5Reader3 = H5WeightsFileReader(h5_file3)
        h5Reader3.print_all_tensors()




