# this file is in the core branch
import sys 
sys.path.append('../')
import tensorflow.compat.v1 as tf
import gtc
from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict
import quantization as quant
from collections import OrderedDict
import numpy as np
import shutil
import os
import tensorflow.keras as keras
from tabulate import tabulate
from utility_scripts import misc_utl as utl
import h5py
#from networks.net_utils import
#import networks.net_utils as net
import networks.net_utils_core as net
import tensorflow.keras.backend as K
import scipy.stats
from utility_scripts import weights_readers
import time
import json
import warnings

class GTCModel:
    def __init__(
            self,
            layers=None,
            name=None,
            model_file=None,
            weights_file=None,
            verbose=False,
    ):
        print("FROM gtc-tensorflow")
        # self._loss = None
        self._optimizer = None
        self._metrices = {}
        self._total_loss = None
        self._summary = OrderedDict()

        self.reset()
        # Extra metadata attributes
        self.type = None # placeholder for a net_type
        self._name = name
        self.userdata = {}

        # Attributes for building keras models
        self.nettask = 'classification'
        self.dataset = None  # unspecified (specify when training)
        self._x_placeholder = None
        self._x_placeholder_keras = None
        self._updates = None
        self._per_layer_updates = None
        self._exported_methods = []
        self._lambda_weights = None

        # Attributes for SSD models
        self.anchor_box_names = None # placeholder for SSD model
        self.anchor_box_sizes = None # placeholder for SSD model
        self.ssd_config = None
        #learning_phase_name = (name if name is not None else 'net_') + 'learning_phase'
        #self._learning_phase = tf.Variable(True, name=learning_phase_name)
        #self.set_learning_phase(True)
        if model_file is not None:
            print('Loading model architecture from %s' % model_file)
            self.load_model_def(model_file, do_compile=True, verbose=verbose)

        if weights_file is not None:
            print('Loading weights from %s' % weights_file)
            self.load_weights(weights_file, verbose=verbose)
            self.initialize(initialize_weights=True, only_uninitialized=True, verbose=verbose)
        #self._eager_build = True  # build the model continually as new modules are added.

    def reset(self):
        self._latest_layer_name = None
        self._layer_names = []
        self._input_shape = None
        self._output = OrderedDict()
        self._layers_objects = OrderedDict()
        self._layers_strings = OrderedDict()
        self._forward_prop_outputs = OrderedDict()
        self._forward_prop_inputs = OrderedDict()
        self._output_layers = OrderedDict()
        self._input_layers = OrderedDict()
        self._compiled = False
        self._quantizers_initialized = False
        self._keras_model_hp = None
        self._keras_model_lp = None


    def convert_to_object(self):
        for lyr in self._layers_strings:
            # print('lyy', lyr)
            input_objects = [
                self._layers_strings[ip]['layer_obj']
                for ip in self._layers_strings[lyr]['inputs']
            ]
            output_objects = [
                self._layers_strings[op]['layer_obj']
                for op in self._layers_strings[lyr]['outputs']
            ]
            self._layers_objects[lyr] = {
                'layer_obj': self._layers_strings[lyr]['layer_obj'],
                'inputs': input_objects,
                'outputs': output_objects
            }

    def unique(self, x_list): # unique, but preserves order
        y_list = []
        for x in x_list:
            if x not in y_list:
                y_list.append(x)
        return y_list

    def reflex(self):
        [
            self._layers_strings[lyri]['outputs'].append(lyrj)
            for lyri in self._layers_strings for lyrj in self._layers_strings
            if lyri in self._layers_strings[lyrj]['inputs']
        ]
        [
            self._layers_strings[lyri]['inputs'].append(lyrj)
            for lyri in self._layers_strings for lyrj in self._layers_strings
            if lyri in self._layers_strings[lyrj]['outputs']
        ]
        #TODO: This assumes that all inputs are unique, but doesn't deal with cases where inputs might be replicated.
        for lyri in self._layers_strings:
            self._layers_strings[lyri]['inputs'] = self.unique(self._layers_strings[lyri]['inputs']) # make unique
            self._layers_strings[lyri]['outputs'] = self.unique(self._layers_strings[lyri]['outputs']) # make unique
            if isinstance(self._layers_strings[lyri]['layer_obj'], gtc.merge._Merge):
                pass

    def compile_training_info(self, loss, optimizer, metrices):

        # self._loss = loss
        self._optimizer = optimizer
        self._metrices = metrices
        self._summary['total_loss'] = loss


    def compile(self, verbose=True, do_keras=False):

        for lyr in self._layers_strings:
            self._layers_strings[lyr]['outputs'] = [
                val
                for val in self._layers_strings[lyr]['outputs']
                if val is not None
            ]
            self._layers_strings[lyr]['inputs'] = [
                val
                for val in self._layers_strings[lyr]['inputs']
                if val is not None
            ]
            convert_to_set=False
            if convert_to_set: # destroys ordering (needed for some merge layers like Concatenate)
                self._layers_strings[lyr]['outputs'] = set(self._layers_strings[lyr]['outputs'])
                self._layers_strings[lyr]['inputs'] = set(self._layers_strings[lyr]['inputs'])


        # self.print_table()
        self.reflex()
        # self.print_table()
        self.convert_to_object()
        #self.print_table()

        self._ordered_outputs = []
        layers_ready = set([])
        #TODO: check if these are necessary.
        #self._ordered_outputs = [
        #    tuple([lyr]) for lyr in self._layers_objects
        #    if self._layers_strings[lyr]['inputs'] == []
        #]
        #layers_ready = set({
        #    tuple([lyr])
        #    for lyr in self._layers_objects
        #    if self._layers_strings[lyr]['inputs'] == []
        #})

        max_layers = len(self._layers_objects.keys())
        while len(layers_ready) < max_layers:
            # each iteration can process one or more layers
            prev_ready = len(layers_ready)
            iter_ready = [
                lyr for lyr in self._layers_objects
                if (lyr not in layers_ready) and all([
                    ip in layers_ready
                    for ip in self._layers_strings[lyr]['inputs']
                ])
            ]
            layers_ready.update(iter_ready)
            self._ordered_outputs.append(tuple(iter_ready))
            if prev_ready == len(layers_ready):
                raise ValueError('Something is wrong!')

        #print('Ordered layers', self._ordered_outputs, '\n')
    #print(layers_ready)

        #verbose = True
    # forward prop
        self.forward_propagate_inputs(verbose, do_keras)

        # for SSD model, get anchor box sizes
        if self.anchor_box_names is not None:
            anchor_box_sizes = \
                [self._forward_prop_outputs[box_name]['hp']._shape_tuple()[1:3] for box_name in self.anchor_box_names]
            self.anchor_box_sizes = anchor_box_sizes

        print('Successfully compiled model')

    def forward_propagate_inputs(self, verbose=False, do_keras=False):
        #def forward_func(self, input_tensor):
        lyr_ip = {}
        #print('forward prop order', self._ordered_outputs)
        for xi, lyr_tup in enumerate(self._ordered_outputs):

            for lyr in lyr_tup:
                lyr_obj = self._layers_objects[lyr]['layer_obj']
                lyr_ip = self._layers_objects[lyr]['inputs']
                lyr_ip_vars = [
                    self._forward_prop_outputs[lip.name] for lip in lyr_ip
                ]

                if lyr_ip_vars == []:  # placeholder:
                    if do_keras:
                        # Replace tensorflow placeholder with keras input placeholder
                        self._x_placeholder_keras = keras.layers.Input(shape=self._x_placeholder.shape[1:])
                        lyr_obj = self._x_placeholder_keras

                    self._forward_prop_inputs[lyr] = []
                    self._forward_prop_outputs[lyr] = GTCDict(hp=lyr_obj, lp=lyr_obj)
                elif len(lyr_ip_vars) == 1:
                    self._forward_prop_inputs[lyr] = lyr_ip_vars[0]

                    print('---- lyr  : ', lyr)
                    print ('----lyr_obj :  ', lyr_obj)
                    print ('----lyr_obj dictionary   :', lyr_obj.__dict__.keys())
                    print('---in compile input',
                        self._forward_prop_inputs[lyr])
                    #print ('----_keras_layer = ', lyr_obj._keras_layer)
                    print('--all layer ops', 
                        self._forward_prop_outputs)
                    batchnorm_training_kwargs = {}
                    #Using a 'training' flag for dynamic training/testing does not work, just 
                    # like using the keras learning_phase()
                    #if isinstance(lyr_obj, gtc.BatchNormalization):
                    #    batchnorm_training_kwargs = dict(training=self._learning_phase)

                    self._forward_prop_outputs[lyr] = lyr_obj(self._forward_prop_inputs[lyr],**batchnorm_training_kwargs)
                elif len(lyr_ip_vars) > 1:
                    # Handle layers with multiple inputs (e.g. Concatenate(), Add())
                    if all(["gtc_tag" in lip for lip in lyr_ip_vars]):
                        hp_list = [lip['hp'] for lip in lyr_ip_vars]
                        lp_list = [lip['lp'] for lip in lyr_ip_vars]
                        # print('hp_list', hp_list)
                        self._forward_prop_inputs[lyr] = GTCDict(hp=hp_list, lp=lp_list)

                        self._forward_prop_outputs[lyr] = lyr_obj(self._forward_prop_inputs[lyr])
                    elif all(
                        [isinstance(lip, tf.Tensor) for lip in lyr_ip_vars]):
                        self._forward_prop_inputs[lyr] = tf.concat(lyr_ip_vars, axis=-1)
                        self._forward_prop_outputs[lyr] = lyr_obj(tf.concat(lyr_ip_vars, axis=-1))

                if verbose and lyr_ip_vars:

                    in_var = lyr_ip_vars[0]
                    if isinstance(in_var, dict) and 'hp' in in_var:
                        in_var = in_var['hp']

                    out_var = self._forward_prop_outputs[lyr]
                    if isinstance(out_var, dict) and 'hp' in out_var:
                        out_var = out_var['hp']

                    in_shape = in_var.shape
                    out_shape = out_var.shape

                    q = self._layers_objects[lyr]['layer_obj']._quantizer
                    quant_str = '[]' if q is None or q.name is None else '[' + str(q.name) + ']'


                    print('%s %s: in=%s. out=%s ' % (lyr, quant_str, str(in_shape.as_list()), str(out_shape.as_list()) ) )

        self.get_output()
        self.get_input()

        self.layers = self.get_layers()

        self._compiled = True
        self._compiled_keras = do_keras

    def get_config(self):
        model_layers = []
        net_name = self._name
        append_net_name = lambda s: s if s.startswith(net_name) else net_name + s
        for i,lyr_name in enumerate(self._layer_names):

            lyr_obj = self._layers_strings[lyr_name]['layer_obj']
            if isinstance(lyr_obj, tf.Tensor):
                if lyr_obj.op.type == 'Placeholder':
                    lyr_obj_save = dict(type='Placeholder',
                                        name=lyr_obj.name.replace(':0', ''),
                                        dtype=lyr_obj.dtype.name,
                                        shape=lyr_obj._shape_tuple())
                else:
                    raise ValueError('Unsupported Tensor')

                model_layers.append( (lyr_name,  lyr_obj_save) )
            else:
                layer_config = lyr_obj.get_config()
                layer_config['name'] = append_net_name(lyr_name)

                model_layers.append( (lyr_name, dict(type=type(lyr_obj).__name__,
                                                config=layer_config)) )

        return model_layers

    def save_model_def(self, json_file, sort_keys=True):
        #if not self._compiled:
        #    raise ValueError('Model must be compiled before definition can be saved')
        model_layers = self.get_config()

        net.stringify_model_specs(model_layers)

        attributes_save = ['type', '_name', 'userdata', 'nettask', 'dataset', 'ssd_config', '_lambda_weights']
        model_attributes = OrderedDict()
        for attr in attributes_save:
            x = getattr(self, attr)
            x_str = net.to_str(x)
            #if not isinstance(x, (int, bool, float, str, list, tuple, dict)):
            #    x = str(x)
            model_attributes[attr] = x_str

        model_def = OrderedDict()
        model_def['model_attributes']=model_attributes
        model_def['model_config'] =model_layers

        with open(json_file, 'w') as f:
            json.dump(model_def, f, indent=4, ensure_ascii=True, sort_keys=sort_keys)

    def load_model_def(self, json_file, do_compile=True, verbose=False):

        with open(json_file) as f:
            model_def = json.load(f)

        # Assign model attributes
        model_attributes = model_def['model_attributes']
        for attr, val in model_attributes.items():
            val_destr = net.from_str(val)
            setattr(self, attr, val_destr)
        if self.type is not None:
            self.type = net.NetworkType.from_str(self.type)

        orig_name = self._name
        self._name = net.get_unique_tf_prefix(orig_name)

        # Load model layers
        model_layers = model_def['model_config']
        net.destringify_model_specs(model_layers)

        # Keep a dictionary of quantizers to facilitate sharing quantizers
        # between layers
        quantizers_dict = {}

        #layer_names = list(model_config.keys())
        for i, (layer_name, layer_specs) in enumerate(model_layers):

            if layer_specs['type'] == 'Placeholder':
                placeholder = tf.placeholder(
                    layer_specs['dtype'],
                    shape=layer_specs['shape'],
                    name=layer_specs['name'])
                self.add(placeholder)

                if verbose:
                    print('Added Layer %d Placeholder [%s]' %  (i+1, str(layer_specs['shape'])))
            else:
                layer_type = gtc.from_str(layer_specs['type'])
                layer_config = layer_specs['config']

                quantizer_name_scope = self._name
                # if we have renamed the network to avoid clashes with existing network in memory
                # propagate this name change to the layers in the network.
                if (orig_name != self._name):
                    if 'name' in layer_config.keys():
                        if orig_name in layer_config['name']:
                            layer_config['name'] = layer_config['name'].replace(orig_name, self._name)
                        else:
                            layer_config['name'] = self._name + layer_config['name']
                    if bool(layer_config.get('layer_inputs', False)):
                        layer_config['layer_inputs'] = [
                            nm.replace(orig_name, self._name) for nm in layer_config['layer_inputs']]

                # For SSD networks, the name scope for the quantizers of the extra SSD layers
                # is under the main network name:
                # e.g: 'MobileNet_SSD_GTC_1/LinearQuantization/L1/slope:0'
                # but for the earlier layers, they are under an extra 'FeatureExtractor/[NetworkName]' name scope
                # e.g: 'MobileNet_SSD_GTC_1/FeatureExtractor/MobilenetV2/LinearQuantization/L1/slope:0'
                # When reconstructing from a .json file, we want to preserve this structure so that saved weights
                # will match the names of the files in the network.
                extra_ssd_layers_prefixes = ['Conv2d_13_pointwise_', 'layer_19_', 'layer_17_']
                contains_any = lambda x, strs: any([str in x for str in strs])
                if 'FeatureExtractor' in layer_config['name'] and \
                    not contains_any(layer_config['name'], extra_ssd_layers_prefixes):
                    # Use all parts up to last 3 parts of the variable name
                    # to exclude: quantizer type, quantizer name, and variable name,
                    # (e.g. 'LinearQuantization' , 'L1', 'slope').
                    name_spaces = layer_config['name'].split('/')
                    quantizer_name_scope = '/'.join(name_spaces[:3]) + '/'  # e.g: 'MobileNet_SSD_GTC_1/FeatureExtractor/MobilenetV2/'

                if 'quantizer' in layer_config:
                    quantizer_name = layer_config['quantizer']
                    if quantizer_name not in quantizers_dict:
                        quantizer = quant.from_str(quantizer_name,
                                                   kwargs=dict(variable_scope=quantizer_name_scope))
                        quantizers_dict[quantizer_name] = quantizer
                    layer_config['quantizer'] = quantizers_dict[quantizer_name]

                gtc_layer = layer_type.from_config(layer_config)
                self.add(gtc_layer)
                if verbose:
                    print('Added Layer %d [%s]' % (i+1, layer_type))


        print('Model definition loaded from %s' % json_file)

        if do_compile:
            print('Compiling ...')
            self.compile(verbose=verbose)
            self.initialize(verbose=verbose)

    def build_keras_model(self, lambda_weights=None, verbose=False):

        if self._compiled:
            assert self._compiled_keras, "Please do not run model.compile() before building the keras model"
        else:
            self.compile(verbose=verbose, do_keras=True)

        if lambda_weights is None:
            lambda_weights = net.LambdaWeights()

        assert lambda_weights.hp is not None, "hp lambda should usually not be None"
        self._lambda_weights = lambda_weights  # copy for internal use only.

        lambda_keys = lambda_weights.get_keys()
        hp_out_name, lp_out_name, distill_loss_name, bit_loss_name, bits_per_layer_name = \
            lambda_keys

        lambda_dict = lambda_weights.get_dict()

        lambda_abbrev = lambda_weights.get_abbrev()


        keras_input = self._x_placeholder_keras

        all_output_shapes = {hp_out_name: K.int_shape( self.output_hp ),
                             lp_out_name: K.int_shape( self.output_lp ),
                             distill_loss_name: (None,),
                             bit_loss_name: (None,),
                             bits_per_layer_name: (None,) }
        output_names = [lambda_abbrev[k] for k,v in lambda_dict.items() if v is not None]
        output_shapes = [all_output_shapes[k] for k,v in lambda_dict.items() if v is not None]

        #keras_model_outputs = GTCModelKerasLayer(new_graph_func=self.forward_func)(keras_input)
        keras_outputs = gtc.GTCModelKerasLayer(
            get_weights_function=self._get_weights,
            get_layer_updates_function=self._get_layer_updates,
            get_losses_function=self._get_layer_losses,
            call_function=self._forward_func,
            output_shape=output_shapes)(keras_input)

        if not isinstance(keras_outputs, list):
            keras_outputs = [keras_outputs]

        #self.output_hp_keras = keras.layers.Lambda(lambda x: x[0], name='output_hp_keras')(keras_outputs)
        #self.output_lp_keras = keras.layers.Lambda(lambda x: x[1], name='output_lp_keras')(keras_outputs)

        rename = lambda x, name: keras.layers.Lambda(lambda x: x, name=name)(x)

        keras_final_outputs = []
        for i,keras_output in enumerate(keras_outputs):
            if output_names[i] == 'hp':
                if self.nettask == 'classification':
                    keras_output_final = keras.layers.Softmax(name=self._name + 'softmax_hp')(keras_output)
                else:
                    keras_output_final = keras_output

                self.output_hp_keras = keras_output_final
                self._keras_model_hp = keras.models.Model(inputs=keras_input, outputs=self.output_hp_keras)
            elif output_names[i] == 'lp':
                if self.nettask == 'classification':
                    keras_output_final = keras.layers.Softmax(name=self._name + 'softmax_lp')(keras_output)
                else:
                    keras_output_final = keras_output

                self.output_lp_keras = keras_output_final
                self._keras_model_lp = keras.models.Model(inputs=keras_input, outputs=self.output_lp_keras)
            else:
                keras_output_final = keras_output
            keras_final_outputs.append(keras_output_final)


        keras_outputs_named = [rename(x, name) for x, name in zip(keras_final_outputs, output_names)]

        keras_model = keras.models.Model(inputs=keras_input, outputs=keras_outputs_named)

        self.lambda_weights_init = lambda_weights

        self.lambda_weights = lambda_weights.copy()
        all_lambda_vars = []
        for k in lambda_weights.get_keys():
            lambda_val = getattr(lambda_weights, k)
            if lambda_val is not None:
                lambda_var = tf.Variable(
                    initial_value=float(lambda_val), name= self._name + k)
                all_lambda_vars.append(lambda_var)
                setattr(self.lambda_weights, k, lambda_var)

        self.initialize_set_of_variables(all_lambda_vars, description='lambda weights', verbose=False)

        # Copy these methods to the keras model, so that they can be accessed
        # during the callbacks, if needed. (some of them, like 'save_weights'
        # and 'load_weights' overwrite the existing keras methods).
        self._exported_methods = ['save_weights', 'load_weights',
                                  #'_get_weights',
                                  #'lambda_weights', 'lambda_weights_init',
                                  # 'calibrate_bn_statistics',
                                  'anchor_box_sizes']
        for f in self._exported_methods:
            setattr(keras_model, f, getattr(self, f) )

        # Also provide a reference to the original GTC model:
        keras_model.gtcModel = self

        return keras_model


    def get_keras_model(self, hp=True):
        if self._compiled_keras:
            if hp:
                return self._keras_model_hp
            else:
                return self._keras_model_lp


    # Sometimes the save_weights() method fails, breaking the training loop.
    # To make it more robust, we allow a few tries before failing.
    @utl.retry((OSError, Exception), tries=10)
    def save_weights(self, filepath, overwrite=True, sess=None, verbose=False, **kwargs):
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

        """
        # If file exists and should not be overwritten:
        if sess is None:
            sess = tf.get_default_session()

        filedir = os.path.dirname(filepath)
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        if not overwrite and os.path.isfile(filepath):
            proceed = keras.utils.io_utils.ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return


        with h5py.File(filepath, 'w') as f:
            #saving.save_weights_to_hdf5_group(f, self.layers)

            all_wgts = self._get_weights(**kwargs)
            weight_vals = sess.run(all_wgts)

            for idx, wgt in enumerate(all_wgts):
                #g = group.create_group(wgt.name)
                wgt_val = weight_vals[idx]

                if wgt.name in f:
                    raise ValueError('Variable %s already exists' % wgt.name)

                param_dset = f.create_dataset(wgt.name, wgt_val.shape,
                                              dtype=wgt_val.dtype)
                if not wgt_val.shape:
                    # scalar
                    param_dset[()] = wgt_val
                else:
                    param_dset[:] = wgt_val

                if verbose:
                    print('(%d) Saving weight %s, shape=%s' % (idx+1, wgt.name, wgt.shape))

            f.flush()




    def save_weights_for_keras_model(self, filepath, overwrite=True,  hp=True):
        """Dumps all layer weights to a HDF5 file for a keras model to load.
         Adapted directly from the keras model saving function to ensure compatibility when loading
         (Use with singlePrecisionKerasModel )

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

        """
        # If file exists and should not be overwritten:

        filedir = os.path.dirname(filepath)
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        if not overwrite and os.path.isfile(filepath):
            proceed = keras.utils.io_utils.ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return

        # create net_layers: an ordered list of layers with attributes : .name, .weights
        net_layers = []
        net_quantizers = []
        weight_count = 0
        for i, layer_name in enumerate(self._layer_names):
            layer_obj = self._layers_objects[layer_name]['layer_obj']
            if isinstance(layer_obj, tf.Tensor):
                continue
            hp_weights = layer_obj.get_hp_weights()
            weight_names = [w.name.encode('utf8') for w in hp_weights]
            weight_count += len(weight_names)

            # Collect quantizers for LP network
            if hasattr(layer_obj, '_quantizer') and layer_obj._quantizer is not None \
                and not isinstance(layer_obj._quantizer, quant.IdentityQuantization):
                    net_quantizers.append(layer_obj._quantizer)

            if hp:
                weights = hp_weights
            else:
                weights = layer_obj.get_lp_weights()

            if bool(weights):
                # Only add group to h5 file if have some weights
                net_layers.append(dict(name=layer_name,
                                       weights=weights,
                                       weight_names=weight_names))

        # For LP network, also save all the quantizers: some of them may be
        # Activity quantizers, which will be needed when recreating the network

        # Now write to h5 file:
        # (the following code is a slight modification of
        # keras.engine.network.saving.save_weights_to_hdf5_group(f, net_layers)
        # to handle flexible naming of layers: We want to save the LP weights
        # using the HP weight names. so each 'layer' is actually a dict
        # (with `name`, `weights`, and `weight_names`), keys
        # instead of a keras layer (with `name`, `weights` attributes)

        with h5py.File(filepath, 'w') as f:
            keras_version = keras.__version__
            group = f
            keras.engine.network.saving.save_attributes_to_hdf5_group(
                group, 'layer_names', [layer['name'].encode('utf8') for layer in net_layers])
            group.attrs['backend'] = K.backend().encode('utf8')
            group.attrs['keras_version'] = str(keras_version).encode('utf8')

            for ii, layer in enumerate(net_layers):
                g = group.create_group(layer['name'])
                symbolic_weights = layer['weights']
                weight_values = K.batch_get_value(symbolic_weights)
                weight_names = layer['weight_names']

                keras.engine.network.saving.save_attributes_to_hdf5_group(
                    g, 'weight_names', weight_names)
                for name, val in zip(weight_names, weight_values):
                    param_dset = g.create_dataset(name, val.shape,
                                                  dtype=val.dtype)
                    if not val.shape:
                        # scalar
                        param_dset[()] = val
                    else:
                        param_dset[:] = val

            if not hp:
                g = group.create_group('quantizers')

                quant_param_dict = OrderedDict() # for debugging
                quant_count = 0
                for quantizer in net_quantizers:
                    quant_params = quantizer.trainable_parameters()
                    quant_values = K.batch_get_value(quant_params)
                    quant_names = [w.name for w in quant_params]
                    for qi in range(len(quant_names)):
                        # some quantizer names are prefixed with the model name. Remove this
                        # prefix so that the start of the name is GTCQuantization/ or LinearQuantization/ etc
                        # The full name should something like  GTCQuantization/G01/intercept:0
                        name_parts = quant_names[qi].split('/')
                        idx_start = [i for i, part in enumerate(name_parts) if 'Quantization' in part]
                        if len(idx_start) != 0:
                            idx_start = idx_start[0]
                            quant_name = '/'.join(name_parts[idx_start:])
                            quant_names[qi] = quant_name


                    quant_count += len(quant_values)
                    for name, val in zip(quant_names, quant_values):
                        quant_param_dict[name] = val
                        name_utf = name.encode('utf8')
                        param_dset = g.create_dataset(name_utf, val.shape,
                                                      dtype=val.dtype)
                        if not val.shape:
                            # scalar
                            param_dset[()] = val
                        else:
                            param_dset[:] = val


                quant_str = ' [and %d quantizer parameters from %d quantizers]' % (
                    quant_count, len(net_quantizers))
            else:
                quant_str = ''

            f.flush()

        print('Successfully saved %d %s weights from %d layers%s to %s' % (
            weight_count, 'HP' if hp else 'LP', len(self._layer_names), quant_str, filepath))

    def load_weights(self, filepath, sess=None, verbose=False):

        if sess is None:
            sess = tf.get_default_session()

        if isinstance(filepath, weights_readers.WeightsReader):
            self.load_pretrained_weights_from_reader(filepath, sess, quant=True, verbose=verbose)
            return

        assign_ops = []
        with h5py.File(filepath, 'r') as f:
            # saving.save_weights_to_hdf5_group(f, self.layers)

            all_wgts = self._get_weights(hp=True, lp=True, non_assignable=False)
            #weight_vals = sess.run(all_wgts)
            f_roots = list(f.keys())

            for idx, wgt in enumerate(all_wgts):
                # g = group.create_group(wgt.name)

                weight_name = wgt.name
                if not weight_name in f:
                    # be robust to differences in network root name
                    for f_root in f_roots:
                        weight_name_alt = weight_name.replace(self._name, f_root + '/')
                        if weight_name_alt in f:
                            weight_name = weight_name_alt
                            break

                if weight_name in f:
                    wgt_val= f[weight_name]


                    if not wgt_val.shape: # scalar
                        wgt_val = wgt_val[()]
                        val_str = ', value= %g' % wgt_val
                    else:
                        wgt_val = wgt_val[:]
                        val_str = ''

                    if verbose:
                        print('(%d) Loading weight %s, shape=%s%s' % (idx+1, wgt.name, str(wgt.shape), val_str))
                    assign_ops.append(tf.assign(wgt, wgt_val, validate_shape=True))
                else:
                    #if verbose:
                    print('(%d) Warning: No weight %s in file' % (idx+1, wgt.name))

        print('Running assignments for %d saved weights/variables' % len(assign_ops))
        sess.run(assign_ops)




    def _forward_func(self, input_tensor):
        # Deprecated.
        tensors_hp = OrderedDict()
        tensors_lp = OrderedDict()

        output_tensor_hp, output_tensor_lp = None, None

        # 1. Generate the outputs from the input
        for xi, lyr_tup in enumerate(self._ordered_outputs):
            for lyr in lyr_tup:
                lyr_obj = self._layers_objects[lyr]['layer_obj']
                lyr_ip = self._layers_objects[lyr]['inputs']
                lyr_ip_vars = [
                    self._forward_prop_inputs[lip.name] for lip in lyr_ip
                ]
                lyr_op = self._layers_objects[lyr]['outputs']

                #self._keras_tensors[lyr] = layer_obj

                if lyr_ip_vars == []:  # placeholder:
                    tensors_hp[lyr] = input_tensor
                    tensors_lp[lyr] = input_tensor

                elif len(lyr_ip) == 1:
                    hp_input = tensors_hp[lyr_ip[0].name]
                    lp_input = tensors_lp[lyr_ip[0].name]

                    layer_inputs = GTCDict(hp=hp_input, lp=lp_input)

                    #if isinstance(lyr_obj, gtc.BatchNormalization):
                    #    lyr_obj.reset_batch_norm_updates()

                    #cfg = lyr_obj.get_config()
                    #lyr_obj_copy = lyr_obj.from_config(cfg)
                    layer_outputs = lyr_obj(layer_inputs)
                    #layer_outputs = lyr_obj_copy(layer_inputs)

                    tensors_hp[lyr], tensors_lp[lyr] = \
                        unpackGTCDict(layer_outputs)

                elif len(lyr_ip_vars) > 1:

                    assert all(["gtc_tag" in lip for lip in lyr_ip_vars])
                    hp_list = [tensors_hp[lip.name] for lip in lyr_ip]
                    lp_list = [tensors_lp[lip.name] for lip in lyr_ip]

                    batchnorm_training_kwargs = {}
                    #Using a 'training' flag for dynamic training/testing does not work, just 
                    # like using the keras learning_phase()
                    #if isinstance(lyr_obj, gtc.BatchNormalization):
                    #    batchnorm_training_kwargs = dict(training=self._learning_phase)

                    layer_outputs = lyr_obj.call(GTCDict(hp=hp_list, lp=lp_list),
                                                 **batchnorm_training_kwargs)

                    tensors_hp[lyr], tensors_lp[lyr] = \
                        unpackGTCDict(layer_outputs)

                if len(lyr_op) == 0:
                    assert output_tensor_hp is None
                    output_tensor_hp = tensors_hp[lyr]
                    output_tensor_lp = tensors_lp[lyr]

        all_outputs = []
        if self._lambda_weights.hp is not None:
            all_outputs.append(output_tensor_hp)
        if self._lambda_weights.lp is not None:
            all_outputs.append(output_tensor_lp)

        batch_size = tf.shape(output_tensor_hp)[0]
        paddings = [[0, batch_size-1]]

        # 2. Add the extra losses (distillation / bit-loss)
        if self._lambda_weights.distillation_loss is not None:
            distill_loss_val = self.distillation_loss_func(
                    output_tensor_hp, output_tensor_lp)
            distill_loss_val_reshaped = \
                tf.pad( tf.expand_dims( distill_loss_val, axis=0), paddings)
            all_outputs.append( distill_loss_val_reshaped )


        if self._lambda_weights.bit_loss is not None:
            bit_loss_val = self.bit_loss(verbose=False)
            bit_loss_val_reshaped = \
                tf.pad(tf.expand_dims(bit_loss_val, axis=0), paddings)

            all_outputs.append(bit_loss_val_reshaped)

        if self._lambda_weights.bits_per_layer is not None:
            bits_per_layer_val = self.get_mean_bits_per_layer()
            bits_per_layer_val_reshaped = \
                tf.pad(tf.expand_dims(bits_per_layer_val, axis=0), paddings)

            all_outputs.append(bits_per_layer_val_reshaped)

        return all_outputs
        #return output_tensor_hp


    def _get_weights(self, hp=True, lp=True, trainable=True, non_trainable=True, assignable=True, non_assignable=True):
        # trainable weights include hp weights/biases, quantization parameters
        #   all trainable weights are assignable
        # non-trainable weights include :
        #      assignable, e.g:  batchnorm running mean/variance (can assign values to them)
        #      non-assignable, e.g; lp (quantized) weights/biases (can't directly assign values to them)

        if not self._compiled:
            raise ValueError('First compile (with .compile() or .build_keras_model) '
                             'before initializing weights')
        all_weights = []
        for i, layer_name in enumerate(self._layer_names):
            layer_obj = self._layers_objects[layer_name]['layer_obj']
            if isinstance(layer_obj, gtc.GTCLayer):
                if trainable:
                    if hp and lp:
                        all_weights.extend( layer_obj.get_trainable_weights() )
                    elif hp:
                        all_weights.extend(layer_obj.get_hp_trainable_weights())
                    elif lp:
                        all_weights.extend(layer_obj.get_lp_trainable_weights())

                    if layer_obj._quantizer is not None:
                        all_trainable_params = layer_obj._quantizer.trainable_parameters()
                        for weight in all_trainable_params:
                            if weight not in all_weights:
                                all_weights.append( weight )
                            # endif check if weight not already in all_weights
                        # end loop over all trainable params
                    # endif check if have a quantizer (layer_obj._quantizer is not None)
                #end if add trainable weights
                if non_trainable:
                    if hp:
                        all_weights.extend(layer_obj.get_hp_non_trainable_weights())
                    if lp:
                        all_weights.extend(layer_obj.get_lp_non_trainable_weights())
                # endif add non-trainable weights

            # endif check if is GTCLayer
        # end loop over all layers
        weights_assignable = [hasattr(w, 'assign') for w in all_weights]
        weights_keep = [ asgn and assignable or not asgn and non_assignable for asgn in weights_assignable ]

        all_weights_keep = [wgt for wgt, keep in zip(all_weights, weights_keep) if keep]

        return all_weights_keep

    def _get_layer_updates(self, redo=False, hp=True, lp=True):

        if self._updates is not None and not redo:
            return self._updates

        self._updates = []
        self._per_input_updates = {}
        for i, layer_name in enumerate(self._layer_names):
            layer_obj = self._layers_objects[layer_name]['layer_obj']
            if isinstance(layer_obj, gtc.BatchNormalization):
                sub_objs = []  #'_batch_normalizer_hp', '_batch_normalizer_lp']
                if hp:
                    sub_objs.append('_batch_normalizer_hp')
                if lp:
                    sub_objs.append('_batch_normalizer_lp')

                for sub_obj in sub_objs:
                    if hasattr(layer_obj, sub_obj):
                        bn_obj = getattr(layer_obj, sub_obj)
                        if hasattr(bn_obj, '_updates'):
                            self._updates.extend(bn_obj._updates)
                        if hasattr(bn_obj, '_per_input_updates'):
                            self._per_input_updates.update(bn_obj._per_input_updates)

        #return self._updates, self._per_input_updates
        return self._updates

    def _get_layer_losses(self):
        all_losses = []

        for i, layer_name in enumerate(self._layer_names):
            layer_obj = self._layers_objects[layer_name]['layer_obj']
            if isinstance(layer_obj, gtc.GTCLayer):
                losses_i = layer_obj.get_losses()
                all_losses.extend(losses_i)

            # endif check if is GTCLayer
        # end loop over all layers
        return all_losses


    def get_quantization_variables(self, evaluate=False, abbrev=True):
        var_list = []
        for lyr_name in self._layer_names:
            layer_obj = self._layers_objects[lyr_name]['layer_obj']
            if hasattr(layer_obj, '_quantizer'):
                quantizer = layer_obj._quantizer
                params = quantizer.trainable_parameters()
                var_list.extend(params)
                if hasattr(layer_obj, '_bias_quantizer'):
                    bias_quantizer = layer_obj._bias_quantizer
                    if quantizer != bias_quantizer:
                        params = bias_quantizer.trainable_parameters()
                        var_list.extend(params)

        abbrev_dict = {'intercept': 'int',
                       'slope': 'slp',
                       'scale': 'scl'}

        def abbreviate_name(var_name):
            # split by '/' delimiter, keep last two elements, join with '_', remove ':0'
            # e.g 'Mobilenet_V2_GTC/GTCQuantization/G36/intercept:0' --> 'G36_intercept'
            is_gtc_quant = 'GTCQuantization' in var_name
            is_linear_quant = 'LinearQuantization' in var_name
            # Make sure we can correctly infer the quantizer type from the name
            assert is_gtc_quant or is_linear_quant
            var_name = '_'.join(var_name.split('/')[-2:]).replace(':0', '')
            # abbreviate names : e.g intercept -> int

            for k_long, k_short in abbrev_dict.items():
                var_name = var_name.replace(k_long, k_short)

            # differentiate between slope from gtc quant (intercept/slope) and
            # slope from linear quant by adding a prefix of 'L' for linear:
            # eg 'L23_slp' -> 'L23_Lslp'
            quant_prefix = 'L' if is_linear_quant else ''
            var_name = var_name.replace('_', '_' + quant_prefix)
            return var_name

        if evaluate:
            var_names = [v.name for v in var_list]
            if abbrev:
                var_names = [abbreviate_name(v) for v in var_names]


            sess = tf.get_default_session()
            var_values = sess.run(var_list)
            var_dict = OrderedDict(zip(var_names, var_values))
            return var_dict
        else:
            return var_list



    def calibrate_bn_statistics(self, dataset_loader, hp=True, lp=True, max_iter=2000,
                                verbose=True, sess=None, print_freq=None, n_av=None):

        all_stats = []
        print('  *** Calibrating batch-norm statistics of gtc model on dataset *** ')
        if verbose:
            stats_init = self.get_bn_statistics(sess, idxs_print='all')
            stats_init = self.get_bn_statistics(sess)
            stats_init_flat = {k: stats_init[k].flatten() for k in stats_init.keys()}

        model = self
        if isinstance(model, keras.Model):
            model = getattr(model, 'gtcModel')

        all_bn_epsilons = []
        for layer_name in model._layer_names:

            lyr = model._layers_objects[layer_name]['layer_obj']
            if isinstance(lyr, gtc.BatchNormalization):
                all_bn_epsilons.append(lyr._batch_normalizer_hp.epsilon)

        assert len(all_bn_epsilons) > 0, "No batch norm layers to calibrate!!"

        min_bn_epsilon = min(all_bn_epsilons)
        max_bn_epsilon = max(all_bn_epsilons)
        assert min_bn_epsilon > 0, "Can't calibrate bn if epsilon==0"

        if max_iter == 'auto':
            max_iter = int( 10 * (1 / min_bn_epsilon) )
            print('Min batch-norm epsilon: %g. [max: %g] Using max-iter = 1/min_epsilon=%d' % (
                min_bn_epsilon, max_bn_epsilon, max_iter))

        if n_av is None:
            n_av = 100

        if print_freq is None:
            print_freq = 5

        count = 0

        all_updates = model._get_layer_updates()
        # layer updates that contain 'gtc_model_keras_layer
        update_for_this_model = lambda op: isinstance(model, keras.Model) == ('gtc_model_keras_layer' in op.name)


        use_update = lambda op: (('bnorm_hp' in op.name and hp) or ('bnorm_lp' in op.name and lp)) and update_for_this_model(op)
        all_updates_use = [op for op in all_updates if use_update(op)]
        get_op_name = lambda op: op.name.replace('AssignMovingAvg_1:0', 'mov_variance').\
            replace('AssignMovingAvg:0', 'mov_mean').replace('/bnorm_', '_')

        all_stat_names = [get_op_name(op) for op in all_updates if update_for_this_model(op) ]

        # Collect the co-efficient of variation (CV) = stddev/mean  of the activations over the previous n_av iterations
        all_cvs = lambda x: scipy.stats.variation(x[-n_av:] + 1e-5)
        all_means = lambda x: np.mean(x[-n_av:] + 1e-5, axis=0)
        mean_th = 0.1  # CV can be very high for tiny values (small values of mean)
                        # so ignore activations with means less than this threshold

        x_placeholder = model.get_placeholder()

        print('Running batches of images through the network to calibrate the batch-norm statistics.\n'
              'During this process, we display some statistics [mean, median, 75/95th percentiles, max] of the \n'
              'coefficients of variation (=stddev/mean) all batchnorm mean/variance statistics \n'
              '(limited to those with mean > %.1f for stability).\n'
              'The bn-statistics should converge (and coefficients of variation, displayed as percentages, \n'
              'should decrease) during the calibration. As a rule of thumb, typically the mean & median CV end up \n'
              'less than 0.05 (indicating less than 0.05%% variation in value across the last %d batches) and\n'
              ' max ends up < 0.5%%).\n'
              'Tip: the total number of batches should be at least 1/(batch-norm epsilon [%g]), ie. %.0f) \n'
              'to allow batch norm statistics to properly converge. For deep networks like mobilenet, I usually \n'
              ' do about 10x this number. In general, monitor the outputs and make sure the calibration continues\n'
              ' until the CVs have stopped decreasing and have more or less converged.\n '
              'Current num batches selected: (%d)' % (
            mean_th, n_av, max_bn_epsilon, 1/max_bn_epsilon, max_iter))

        t_start = time.time()
        for data_j in dataset_loader:
            count += 1
            data_im_raw, _ = data_j

            data_dict = {x_placeholder: data_im_raw }

            all_update_vals = sess.run(all_updates_use, feed_dict=data_dict)

            stats_i = OrderedDict( zip(all_stat_names, all_update_vals) )
            all_stats.append(stats_i)

            if (count % print_freq == 0)  and (count > 5) : #:  and (count > n_av) :
                #print('update %d' % count)
                stats_summary = {k: np.array([stats_i[k].flatten() for stats_i in all_stats]) for k in all_stat_names}

                stats_cvs = [all_cvs(stats_summary[k]) for k in all_stat_names]
                stats_means = [all_means(stats_summary[k]) for k in all_stat_names]

                #stats_diffs_from_init = [ np.mean( np.abs(stats_init[ki]-stats_i[k])) for ki,k in zip(stats_init.keys(), all_stat_names)]
                #print('mean diff from init : %.6f' % np.mean(stats_diffs_from_init))
                stats_cvs_selected = [c[m > mean_th] * 100 for c, m in zip(stats_cvs, stats_means)]

                est_time_remain = (time.time()-t_start) * ( (max_iter-count)/(count) )
                all_stats_cvs = np.concatenate(stats_cvs_selected)
                # Print out the statistics of the CVs of all activations. These are in percentages, so 1= 1% of mean value
                # when max is < 5%, that means the CV has stabilized pretty well
                print(' (%d/%d) coeff-vars (mean>%.1f): mean = %.4f.    median=%.4f.    75th prctile: %.4f.  95th prctile: %.4f.   max: %.4f [ETA: %.1f sec]' % (
                        count, max_iter, mean_th, all_stats_cvs.mean(), np.median(all_stats_cvs), np.percentile(all_stats_cvs, 75.0),
                        np.percentile(all_stats_cvs, 95.0), all_stats_cvs.max(), est_time_remain))

            plot_results = False
            if plot_results:
                rel = lambda x: x / np.mean(x, axis=0)
                v0_mean = np.array([s['MobilenetV2/Conv/BatchNorm_hp.mean'].flatten()[0:5] for s in all_stats])
                v0_var = np.array([s['MobilenetV2/Conv/BatchNorm_hp.variance'].flatten()[0:5] for s in all_stats])

                v1_mean = np.array([s['MobilenetV2/Conv_1/BatchNorm_hp.mean'].flatten()[0:5] for s in all_stats])
                v1_var = np.array([s['MobilenetV2/Conv_1/BatchNorm_hp.variance'].flatten()[0:5] for s in all_stats])
                # import matplotlib.pyplot as plt
                # plt.plot(rel(v1_var))

            if verbose:
                pass

            if count >= max_iter:
                break


        plot_results = False
        if plot_results:
            rel = lambda x: x / np.mean(x, axis=0)
            v0_mean = np.array([s['MobilenetV2/Conv/BatchNorm_hp.mean'].flatten()[0:5] for s in all_stats ])
            v0_var = np.array([s['MobilenetV2/Conv/BatchNorm_hp.variance'].flatten()[0:5] for s in all_stats])

            v1_mean = np.array([s['MobilenetV2/Conv_1/BatchNorm_hp.mean'].flatten()[0:5] for s in all_stats ])
            v1_var = np.array([s['MobilenetV2/Conv_1/BatchNorm_hp.variance'].flatten()[0:5] for s in all_stats])
            #import matplotlib.pyplot as plt
            #plt.plot(rel(v1_var))


    def get_bn_statistics(self, sess=None, idxs_print=None, verbose=False):
        np.set_printoptions(linewidth=250)
        if sess is None:
            sess = tf.get_default_session()
        if idxs_print is None:
            idxs_print = [0,1,2,3,4,5,   -6,-5,-4,-3,-2,-1]
        all_bn_names = []
        all_bn_tensors = []

        for i, layer_name in enumerate(self._layer_names):
            layer_obj = self._layers_objects[layer_name]['layer_obj']
            if isinstance(layer_obj, gtc.BatchNormalization):
                sub_objs = ['_batch_normalizer_hp', '_batch_normalizer_lp']
                for sub_obj in sub_objs:
                    if hasattr(layer_obj, sub_obj):
                        bn_obj = getattr(layer_obj, sub_obj)
                        bn_name = layer_name + sub_obj[-3:]
                        all_bn_names.append(bn_name + '.mean')
                        all_bn_names.append(bn_name + '.variance')
                        all_bn_tensors.append(bn_obj.moving_mean)
                        all_bn_tensors.append(bn_obj.moving_variance)


        all_bn_tensors_eval = sess.run(all_bn_tensors)
        all_bn_vars = OrderedDict(zip(all_bn_names, all_bn_tensors_eval))
        u_bn_names = [nm[:-5] for nm in all_bn_names if '.mean' in nm]
        if idxs_print == 'all':
            idxs_print_rect = list(range(len(u_bn_names)))
        else:
            idxs_print_rect = [(i if i >= 0 else i + len(u_bn_names)) for i in idxs_print]

        if verbose:
            print('')
            for i,k in enumerate(u_bn_names):
                if i in idxs_print_rect:
                    print('(%d) %10s : mean=%s, var=%s' % (i,
                        k, str(all_bn_vars[k + '.mean'].flatten()[:5]),
                           str(all_bn_vars[k + '.variance'].flatten()[:5]) ) )

        return all_bn_vars



    def fuse_conv_batchnorm_layers(self, sess, verbose=True, remove_bn_modules=False, hp=True, lp=True):

        assign_ops = []
        fuse_count = 0
        all_precisions = ['hp', 'lp']
        precisions_merge = [prec for prec,tf_use in zip(all_precisions, [hp, lp]) if tf_use]
        if not all([hp, lp]):
            assert not remove_bn_modules, "Can only remove bn modules if both hp and lp are fused"

        for i, layer_name in enumerate(self._layer_names):

            layer_obj = self._layers_objects[layer_name]['layer_obj']
            if len(self._layers_objects[layer_name]['outputs']) == 0:
                continue
            next_layer_obj = self._layers_objects[layer_name]['outputs'][0]

            if isinstance(layer_obj, (gtc.Conv2D, gtc.Dense)) and \
                isinstance(next_layer_obj, (gtc.BatchNormalization) ):
                # merge the conv and batchnorm weights.

                fuse_count += 1
                conv_layer = layer_obj
                bn_layer = next_layer_obj
                if verbose:
                    print(' (%d) Fusing  [%s] + [%s]' % (fuse_count, conv_layer.name, bn_layer.name))


                if layer_obj._fused:  # already fused.
                    print('Warning: these layers have already previously been fused')

                for precision in precisions_merge:

                    if precision == 'lp':
                        conv_weights = conv_layer.lp_weights
                        conv_bias = conv_layer.lp_bias
                        bn = bn_layer._batch_normalizer_lp

                    else:
                        assert precision == 'hp'
                        conv_weights = conv_layer.hp_weights
                        conv_bias = conv_layer.hp_bias
                        bn = bn_layer._batch_normalizer_hp


                    bn_weight = tf.divide(bn.gamma,
                                          tf.sqrt(bn.epsilon + bn.moving_variance))  # [c_out] -> [c_out, c_out]

                    fused_weights, nfilters = None, None
                    if isinstance(conv_layer, gtc.Conv2D):
                        nfilters = conv_layer._out_channels

                        if conv_layer._do_depthwise_conv:
                            conv_weights_reshaped = tf.reshape(conv_weights, [-1, nfilters])  # [k , k , c_in , 1] ->  [k*k, c_in]
                            fused_weights = tf.reshape(tf.multiply(conv_weights_reshaped, bn_weight), conv_weights.shape)

                        else:
                            conv_weights_reshaped = tf.transpose(
                                tf.reshape(conv_weights, [-1, nfilters]))  # [k , k , c_in , c_out] ->  [k*k*c_in, c_out]
                            bn_weight_diag = tf.diag(bn_weight)  # [c_out] -> [c_out, c_out]
                            fused_weights = tf.reshape(tf.transpose(tf.matmul(bn_weight_diag, conv_weights_reshaped)),
                                                       conv_weights.shape)

                    elif isinstance(conv_layer, gtc.Dense):
                        nfilters = conv_layer._units

                        dense_weights_reshaped = tf.transpose(
                            tf.reshape(conv_weights, [-1, nfilters]))  # [k , k , c_in , c_out] ->  [k*k*c_in, c_out]
                        bn_weight_diag = tf.diag(bn_weight)  # [c_out] -> [c_out, c_out]
                        fused_weights = tf.reshape(tf.transpose(tf.matmul(bn_weight_diag, dense_weights_reshaped)),
                                                   conv_weights.shape)

                    # 3b. Prepare bias
                    fused_bias = bn.beta - tf.multiply(bn_weight, bn.moving_mean)
                    if conv_layer._use_bias:
                        fused_bias += tf.multiply(bn_weight, conv_bias)  # add to fused_bias if use_bias==True, otherwise stands alone. (assumes center == True)

                    fused_weights_val = sess.run(fused_weights)
                    fused_bias_val = sess.run(fused_bias)
                    # copy over the newly fused weights:
                    if precision == 'hp': # overwrite current weights
                        assign_ops.append( tf.assign(conv_weights, fused_weights_val, validate_shape=True) )
                        if conv_layer.hp_bias is None:
                            conv_layer.hp_bias = tf.constant(fused_bias_val)
                        else:
                            assign_ops.append(tf.assign(conv_bias, fused_bias_val, validate_shape=True))
                    elif precision == 'lp': # create new low-precision weight tensors
                        conv_layer.lp_weights = tf.constant(fused_weights_val)
                        conv_layer.lp_bias = tf.constant(fused_bias_val)
                        #assign_ops.append( fused_weights )
                        #assign_ops.append( fused_bias )


                            # overwrite the batchnorm values with values to make it do nothing
                    assign_ops.append( tf.assign(bn.beta, np.zeros(nfilters), validate_shape=True ))
                    assign_ops.append( tf.assign(bn.gamma, np.ones(nfilters), validate_shape=True ))
                    assign_ops.append( tf.assign(bn.moving_mean, np.zeros(nfilters), validate_shape=True ))
                    assign_ops.append( tf.assign(bn.moving_variance, np.ones(nfilters)-bn.epsilon, validate_shape=True ))

                layer_obj.fused = True
                conv_layer._use_bias = True  # whether or not it had a bias term before, the conv layer has one now
                                             # only set this value after doing completing the merge for this layer

                #if verbose:
                #    print(' (%d) Fusing  [%s] + [%s]  ... complete' % (fuse_count, conv_layer.name, bn_layer.name))

        print('Fused %d batchnorm layers' % fuse_count)
        sess.run([assign_ops])
        if remove_bn_modules:
            self.remove_batchnorm_modules()


    def remove_batchnorm_modules(self, verbose=True):

        rm_count = 0
        new_layer_names = []
        for i, layer_name in enumerate(self._layer_names):

            layer_obj = self._layers_objects[layer_name]['layer_obj']
            if len(self._layers_objects[layer_name]['outputs']) == 0:
                continue
            if len(self._layers_objects[layer_name]['inputs']) == 0:
                continue
            prev_layer_objs = self._layers_objects[layer_name]['inputs']
            next_layer_objs = self._layers_objects[layer_name]['outputs']

            cur_layer_str = self._layers_strings[layer_name]

            prev_layer_names = self._layers_strings[layer_name]['inputs']
            next_layer_names = self._layers_strings[layer_name]['outputs']


            if isinstance(layer_obj, gtc.BatchNormalization):
                assert isinstance(prev_layer_objs[0], (gtc.Conv2D, gtc.Dense))
                # merge the conv and batchnorm weights.

                # change the graph so that
                #  [prev-layer] ==> batch-norm layer ==> [next-layer]
                # becomes
                #  [prev-layer]            ==>           [next-layer]



                if len(prev_layer_names) > 1 or len(next_layer_names) > 1:
                    a = 1
                bn_layer_name = layer_name
                assert len(prev_layer_names) == 1, "Batch norm layers should only receive 1 input"

                # modify outputs:
                # For the upstream layer that produced an output that feeds INTO the batchnorm layer,
                # its output will have to be directed to the layer that the batchnorm currently outputs to
                prev_layer_name = prev_layer_names[0]
                prev_layer_outputs = self._layers_strings[prev_layer_name]['outputs']
                idx_batchnorm_output = prev_layer_outputs.index(bn_layer_name)

                if verbose:
                    print(' Removing %s ' % bn_layer_name)

                # Make the change for all downstream layers (could be more than 1)
                for next_layer_name in next_layer_names:
                    # modify inputs:
                    # For each downstream layer that receives inputs FROM the batchnorm layer,
                    # their new inputs will have to be from the layer that precedes the current batchnorm layer
                    next_layer_inputs = self._layers_strings[next_layer_name]['inputs']
                    idx_batchnorm_input = next_layer_inputs.index(bn_layer_name)
                    if verbose:
                        print('   %s --> [[%s]] --> %s ' % (prev_layer_name, bn_layer_name, next_layer_name) )

                    # modify output for previous layer:
                    prev_layer_outputs[idx_batchnorm_output] = next_layer_name

                    # modify input for next layer
                    next_layer_inputs[idx_batchnorm_input] = prev_layer_name




                rm_count += 1
                # end loop over downstream layers
            #end if is batchnorm layer
        # end loop over all layers.
        print('Removed %d batchnorm layers' % rm_count)
        # now recompile:
        self.compile(verbose=verbose)



    def print_model(self, print_input=True, print_output=True, print_shape=True, print_quantizers=True, sess=None):
        headers = ['Layer Name', 'Quantizer', 'Input size', 'Output size', 'Weights']
        tf_use =  [True, print_input, print_quantizers, print_output, print_shape]
        headers_use = [ h for h,t in zip(headers, tf_use) if t ]

        get_shape = lambda X: [x._shape_tuple() for x in X] if isinstance(X, list) else X._shape_tuple()
        table = []
        #try:
        for intd_layers in self._ordered_outputs:
            for intd_layer in intd_layers:
                # Input name
                layer_obj = self._layers_objects[intd_layer]['layer_obj']
                layer_is_frozen = hasattr(layer_obj, 'trainable') and not layer_obj.trainable
                layer_has_weights = hasattr(layer_obj, 'get_weights') and len(layer_obj.get_hp_weights()) > 0
                if layer_is_frozen and layer_has_weights:
                    layer_name_str = '{' + intd_layer + '}'
                else:
                    layer_name_str = '-' + intd_layer


                quant_name_strs = []
                if isinstance(layer_obj, gtc.GTCLayer):
                    quantizer_names = ['_quantizer']
                    if hasattr(layer_obj, '_bias_quantizer_type') and \
                            layer_obj._bias_quantizer_type != 'same' and \
                            layer_obj._use_bias == True:
                        quantizer_names.append('_bias_quantizer')

                    for quantizer_name in quantizer_names:
                        if not hasattr(layer_obj, quantizer_name):
                            continue

                        q = getattr(layer_obj, quantizer_name)
                        if q is not None and q.name is not None:
                            q_params_str = ''
                            if self._quantizers_initialized and sess is not None:
                                params = q.trainable_parameters()
                                params_vals = sess.run( q.trainable_parameters() )
                                params_names = [p.name.split('/')[-1][:1] for p in params]
                                q_params_str = ':' + ', '.join(
                                    [nm + '=' + str(val) for nm,val in zip(params_names, params_vals) ] )
                            quant_name_strs.append(q.name + q_params_str)

                quant_name_str = '<' + ','.join(quant_name_strs) + '>'

                # Input shapes
                if self._forward_prop_inputs[intd_layer] == []:
                    input_shape_str = '[]'
                elif isinstance(self._forward_prop_inputs[intd_layer],
                                tf.Tensor):
                    input_shape_str = self._forward_prop_inputs[intd_layer].shape
                else:
                    input_shapes = get_shape( self._forward_prop_inputs[intd_layer]['hp'] )

                    if isinstance(input_shapes, list):
                        input_shape_strs = [str(shp) for shp in input_shapes]
                        input_shape_str = utl.concat_detect_repeats(input_shape_strs)

                    else:
                        input_shape_str = str(input_shapes)


                # Output shapes
                if isinstance(self._forward_prop_outputs[intd_layer],
                              tf.Tensor):
                    output_shape_str = self._forward_prop_outputs[intd_layer].shape
                else:
                    output_shape_str = get_shape( self._forward_prop_outputs[intd_layer]['hp'])

                # Kernel sizes

                try:
                    layer_obj = self._layers_objects[intd_layer]['layer_obj']
                    hp_weights = layer_obj.get_hp_weights()
                    #var3 = layer_obj.filter_shape
                    #var3_w = hp_weights[0]
                    #var3 = str(var3_w.shape.as_list())

                    try:
                        wgt_details_each = [w.name + ':' + str(w.shape.as_list()) for w in hp_weights ]
                        weights_str = utl.factorize_strs(wgt_details_each)
                    except:
                        weights_str = '<Error creating Shape>'


                except:
                    weights_str  = ' '

                line_strs = [layer_name_str, quant_name_str, input_shape_str, output_shape_str, weights_str]
                line_strs_use = [v for v, t in zip(line_strs, tf_use) if t]
                table.append(line_strs_use)
        #except:
        #    print('Error when creating table. Here is what we got so far...:')

        print(
            tabulate(
                tabular_data=table,
                headers=headers_use ) )


    def add_list(self, layers_list=None, inputs=None, outputs=None, names=None):

        for i,layer in enumerate(layers_list):
            inputs_i = None
            outputs_i = None
            name_i=None
            if i == 0:
                inputs_i = inputs
            if i == len(layers_list)-1:
                outputs_i = outputs
            if names is not None:
                name_i=names[i]

            self.add(layer, inputs=inputs_i, outputs=outputs_i, name=name_i)

    def add(self, layer, inputs=None, outputs=None, name=None):

        # inputs:
        if inputs is None and not isinstance(layer, tf.Tensor):
            inputs = self._latest_layer_name

        if isinstance(layer, GTCLayer) and layer._layer_inputs is not None: # layer inputs can be stored in the layer itself
            inputs = layer._layer_inputs

        if not isinstance(inputs, list):
            inputs = [inputs]

        # outputs:
        if isinstance(layer, GTCLayer) and layer._layer_outputs is not None: # layer inputs can be stored in the layer itself
            outputs = layer._layer_outputs

        if not isinstance(outputs, list):
            outputs = [outputs]

        for ip in inputs:
            # print('inputs', ip)
            assert isinstance(ip, (GTCLayer, str)) or (ip is None)


        for ip_idx in range(len(inputs)):
            if isinstance(inputs[ip_idx], GTCLayer):
                inputs[ip_idx] = inputs[ip_idx].name

        for op in outputs:
            assert isinstance(op, str) or (op is None)

        if layer.name in self._layers_strings:
            raise ValueError('Already have layer "%s" [existing layers : %s]' % (layer.name, ', '.join(self._layers_strings.keys())) )
        assert layer.name not in self._layers_strings
        self._layers_strings[layer.name] = {
            'layer_obj': layer,
            #'inputs': set(inputs),
            #'outputs': set(outputs)
            'inputs': inputs,  # for Concatenate, order is important. For merge layers in general, don't mess with specific order/duplication.
            'outputs': outputs
        }
        self._latest_layer_name = layer.name

        self._layer_names.append(layer.name)

        if isinstance(layer, tf.Tensor):
            assert inputs == [None]
            #new_layer = gtc.Activation(
            #    quantizer=quant.IdentityQuantization(),
            #    activation='linear',
            #    name=name)
            #self.add(new_layer, inputs=[layer.name])
        #print('table from add')
        #print('\n')
            if 'Placeholder' in layer.op.name:
                self._x_placeholder = layer
                self._input_shape = layer._shape_tuple()



    def add_linear_quantizer(self, num_bits, quant_name='LQ', layer_name='activity_quantized'):

        linear_quantizer = quant.LinearQuantization(
            num_bits=num_bits, variable_scope=self._name, name=quant_name)

        candidate_layer_name = self._name + layer_name
        id = 0
        while candidate_layer_name in self._layer_names:
            id += 1
            candidate_layer_name = self._name + layer_name + '_%d' % id

        self.add(gtc.Activation('linear',
                                name=candidate_layer_name,
                                quantizer=linear_quantizer)) # all_quantizers.next('activation')))


    def get_layers(self):
        all_layers = [ self._layers_strings[layer_name]['layer_obj'] for layer_name in self._layer_names ]
        return all_layers

    def get_layer_names(self):
        all_layer_names = [ x.name for x in self.get_layers() ]
        return all_layer_names

    def last_layer_name(self):
        return self.get_layer_names()[-1]

    def get_output(self):
        for k, op in self._layers_strings.items():
            if self._layers_strings[k]["outputs"] in [ set(), [] ]:
                self._output_layers[k] = self._forward_prop_outputs[k]

        if len(self._output_layers) == 1:
            k = list(self._output_layers.keys())[0]
            self.output = self._output_layers[k]
            self.output_hp = self.output['hp']
            self.output_lp = self.output['lp']

        return self._output_layers

    def get_input(self):
        for k, op in self._layers_strings.items():
            if self._layers_strings[k]["inputs"] in [ set(), [] ]:
                self._input_layers[k] = self._layers_objects[k]['layer_obj']

        if len(self._input_layers) == 1:
            k = list(self._input_layers.keys())[0]
            self.input = self._input_layers[k]

        return self._input_layers


    def print_table(self, table=None):
        print("printing summary")
        [print(k, v) for k, v in table.items()]

    def train_on_batch(self, data_dict):
        self._optimizer.run(feed_dict=data_dict)

    def get_summary(self):

        bit_dict = self._bits_per_layer().items()
        qerr_op_dict = self._quantization_loss_outputs().items()
        qerr_weights_dict = self._quantization_loss_weights().items()
        qangle_op_dict = self._quantization_angle_diff_outputs().items()
        qangle_weights_dict = self._quantization_angle_diff_weights().items()
        #slope_dict, intercept_dict = self._get_slope_and_intercept()
        # print('SUMMART in get_summary', self._summary)

        self._summary['weights_bits'] = {}
        self._summary['bias_bits'] = {}
        self._summary['output_bits'] = {}
        self._summary['bits'] = {}
        bits_average = []
        # print('bit dict', bit_dict)
        for k, v in bit_dict:
            #if len(v) == 2:
            #    self._summary['weights_bits'][k] = v['wgbits']
            #    self._summary['bias_bits'][k] = v['biasbits']
            #else:
            #    self._summary['output_bits'][k] = v['opbits']
            self._summary['weights_bits'][k] = v
            bits_average.append(v)
        self._summary['bits']['average'] = tf.reduce_mean(bits_average)

#        qerr_op_average = []
#        self._summary['qerr_op'] = {}
#        for k, v in qerr_op_dict:
#            self._summary['qerr_op'][k] = v[]
#            qerr_op_average.append(v)
#        self._summary['qerr_op']['average'] = tf.reduce_mean(qerr_op_average)
#
#        qerr_weights_average = []
#        self._summary['qerr_weights_bias'] = {}
#        self._summary['qerr_weights_weights'] = {}
#        self._summary['qerr_weights'] = {}
#        for k, v in qerr_weights_dict:
#            self._summary['qerr_weights_weights'][k] = v[0]
#            if len(v) > 1:
#                self._summary['qerr_weights_bias'][k] = v[1]
#            else:
#                self._summary['qerr_weights_bias'][k] = v[0]*0
#
#            qerr_weights_average.append(v)
#        
#        self._summary['qerr_weights']['average'] = tf.reduce_mean(
#            qerr_weights_average)
#
#        qangle_op_average = []
#        self._summary['qangle_op'] = {}
#        for k, v in qangle_op_dict:
#            self._summary['qangle_op'][k] = v[0]
#            qangle_op_average.append(v)
#        self._summary['qangle_op']['average'] = tf.reduce_mean(
#            qangle_op_average)
#
#        qangle_weights_average = []
#        self._summary['qangle_weights_weights'] = {}
#        self._summary['qangle_weights_bias'] = {}
#        self._summary['qangle_weights'] = {}
#
#        for k, v in qangle_weights_dict:
#            self._summary['qangle_weights_weights'][k] = v[0]
#            self._summary['qangle_weights_bias'][k] = v[1]
#            qangle_weights_average.append(v)
#        self._summary['qangle_weights']['average'] = tf.reduce_mean(
#            qangle_weights_average)
#        #self._summary['q_slope'] = {}
#        #self._summary['q_intercept'] = {}

        #for k in slope_dict.keys():
        #    print('keys', k)
        #    self._summary['q_slope'][k] = slope_dict[k]
        #    self._summary['q_intercept'][k] = intercept_dict[k]

        #print('SUMMARY1', self._summary)
        return self._summary

    def print_summary(self, data_dict):
        for k, v in self._summary.items():
            #print('checking ummary param', k, v)
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    if isinstance(v1, tuple):
                        print(k, k1, v1[0].eval(data_dict))
                    else:
                        print(k, k1, v1.eval(data_dict))
            else:
                print(k, v.eval(data_dict))


    def bit_loss(self, verbose=False):

        bits_per_layer = self.get_bits_per_layer(verbose)
        # print('BitsPerLatyer,', bits_per_layer)

        all_num_bits = [2 ** numbits for numbits in bits_per_layer.values()]

        if len(all_num_bits) == 0:
            # in case no layers are quantized, add a dummy value so we don't get nan
            all_num_bits.append( tf.constant([0.0]) )

        bit_loss_val = tf.reduce_sum(all_num_bits)
        # bit_loss_val = tf.reduce_sum(
        #    [2**lb[1] for lb in bits_per_layer.values()])
        self._summary['bit_loss'] = bit_loss_val
        return bit_loss_val


    def get_bits_per_layer(self, evaluate=False, abbrev=False ):

        forward_prop_inputs = self._forward_prop_inputs
        assert self._compiled, "Must compile before getting bits per layer"

        all_keys = []
        all_values = []

        def abbrev_key(k, layer_obj):
            k = k.replace(self._name, '')
            k = k.replace('BatchNorm', 'BN')
            k = k.replace('Activation', 'Act')
            k = k.replace('expanded_conv/', 'blk_0/') # First block
            k = k.replace('expanded_conv_', 'blk_') # subsequent blocks
            k = k.replace('depthwise', 'dw')
            k = k.replace('expand', 'exp')
            k = k.replace('project', 'proj')

            if isinstance(layer_obj, gtc.Conv2D):
                k = k + '/Conv'

            return k

        for k in self._layer_names:
            v = self._layers_objects[k]
            if (v['inputs'] == []):
                continue
            if isinstance(v['layer_obj']._quantizer, quant.IdentityQuantization) or\
                    isinstance(v['inputs'], tf.Tensor):
                continue

            layer_obj = v['layer_obj']
            if abbrev:
                key = abbrev_key(k, layer_obj)
            else:
                key = k
            all_keys.append(key)

            num_bits =  v['layer_obj'].number_of_bits(forward_prop_inputs[k])
            all_values.append(num_bits)


        if evaluate:
            sess = tf.get_default_session()
            all_values = sess.run(all_values)

        bits_per_layer = OrderedDict(zip(all_keys, all_values))

        return bits_per_layer

    def get_mean_bits_per_layer(self, evaluate=False):
        all_bits_per_layer = self.get_bits_per_layer()
        all_num_bits = [numbits for numbits in all_bits_per_layer.values()]
        mean_bits_per_layer = tf.reduce_mean(all_num_bits)

        if evaluate:
            sess = tf.get_default_session()
            mean_bits_per_layer = sess.run(mean_bits_per_layer)

        return mean_bits_per_layer

    def _bits_per_layer(self):
        bits_per_layer = {}
        for k, v in self._layers_objects.items():
            if not (v['inputs'] == []):
                if not isinstance(v['layer_obj']._quantizer,
                                  quant.IdentityQuantization) and not isinstance(
                                      v['inputs'], tf.Tensor):
                    bits_per_layer[k] = v['layer_obj'].number_of_bits(
                        self._forward_prop_inputs[k])
        return bits_per_layer

    def _get_slope_and_intercept(self):
        slope_per_quantized_layer = {}
        intercept_per_quantized_layer = {}
        for k, v in self._layers_objects.items():
            if not (v['inputs'] == []):
                if not isinstance(v['layer_obj']._quantizer,
                                  quant.IdentityQuantization) and not isinstance(
                                      v['inputs'], tf.Tensor):
                    temp_intercept, temp_slope = v[
                        'layer_obj']._quantizer.trainable_parameters()
                    slope_per_quantized_layer[k] = temp_slope
                    intercept_per_quantized_layer[k] = temp_intercept
        return (slope_per_quantized_layer, intercept_per_quantized_layer)


#    def _bits_per_layer(self):
#        bits_per_layer = {}
#        for k, v in self._layers_objects.items():
#            if isinstance(v['layer_obj'], GTCLayer):
#                if ((not isinstance(v['layer_obj']._quantizer, IdentityQuantization)) and (v['layer_obj'].number_of_bits(
#                        self._forward_prop_inputs[k]) is not None)
#                        and (not isinstance(
#                            v['layer_obj'].number_of_bits(
#                                self._forward_prop_inputs[k])[0], int))):
#                    bits_per_layer[k] = v['layer_obj'].number_of_bits(
#                        self._forward_prop_inputs[k])
#        return bits_per_layer

    def get_conv_weights(self):

        #all_weight_names = self._get_layer_losses()
        all_weights = []

        for k in self._layer_names:
            layer_obj = self._layers_objects[k]['layer_obj']
            if isinstance(layer_obj, (gtc.Conv2D, gtc.Dense)):
                wgts = layer_obj.get_hp_weights()
                all_weights.extend(wgts)

        return all_weights




    def L2_regularization(self, list_of_layers=[]):
        if list_of_layers == []:
            [list_of_layers.append(k) for k in self._layers_objects.keys()]

        # regularization parameter
        regularization_term = 0
        for k in list_of_layers:
            if isinstance(self._layers_objects[k]['layer_obj'], (gtc.Conv2D, gtc.Dense)):

                regularization_term = regularization_term + tf.nn.l2_loss(
                    self._layers_objects[k]['layer_obj'].hp_weights)
        return regularization_term

    def hp_cross_entropy(self, y_placeholder):

        # High precision Cross entropy loss
        if isinstance(y_placeholder, tf.Tensor):
            assert len(self._output_layers.items()) == 1
            y_placeholder_dict = {}
            y_placeholder_dict[list(
                self._output_layers.keys())[0]] = y_placeholder
        elif isinstance(y_placeholder, dict):
            y_placeholder_dict = y_placeholder

    # print('SAVING HPCE')
        self._summary['hp_cross_entropy'] = {}
        temp_hp_cross_entropy = {}
        for op_key, op_val in y_placeholder_dict.items():
            ypred = self._forward_prop_outputs[op_key]
            hp_cross_entropy_full = tf.nn.softmax_cross_entropy_with_logits(
                logits=ypred["hp"], labels=y_placeholder_dict[op_key])
            hp_cross_entropy_val = tf.reduce_mean(hp_cross_entropy_full)
            temp_hp_cross_entropy[op_key] = hp_cross_entropy_val
        self._summary['hp_cross_entropy'] = temp_hp_cross_entropy
        return temp_hp_cross_entropy

    def distillation_loss_func(self, logits_hp, logits_lp):
        if self.nettask == 'classification':
            distillation_loss_full = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits_lp, labels=tf.nn.softmax(logits_hp))
            distillation_loss_val = tf.reduce_mean(
                distillation_loss_full, name='distillation_loss')

        elif self.nettask == 'detection':
            from ssd.keras_ssd_loss import SSDLoss
            ssd_loss = SSDLoss(self.ssd_config, soft_labels=True)
            distillation_loss_full = ssd_loss.compute_loss(y_true=logits_hp, y_pred=logits_lp)
            distillation_loss_val = tf.reduce_mean(
                distillation_loss_full, name='distillation_loss')

        else:
            raise ValueError('Unrecognized network task : %s' % self.nettask)

        return distillation_loss_val

    def distillation_loss(self):
        # Cross Entropy between lp and hp
        self._summary['distillation_loss'] = {}
        temp_distillation_loss = {}
        for op_key, op_val in self._output_layers.items():
            ypred = self._forward_prop_outputs[op_key]
            lp_distillation_loss_val = self.distillation_loss_func(ypred['hp'], ypred['lp'])
            temp_distillation_loss[op_key] = lp_distillation_loss_val
            #self._summary['distillation_loss'][
            #   op_key] = lp_distillation_loss_val
        self._summary['distillation_loss'] = temp_distillation_loss
        #print('SUMM in distillation_loss', self._summary)
        return temp_distillation_loss

    def hp_accuracy(self, y_placeholder):
        if isinstance(y_placeholder, tf.Tensor):
            assert len(self._output_layers.items()) == 1
            output_layer_name = list(self._output_layers.keys())[0]
            y_placeholder_dict = {output_layer_name: y_placeholder}
        elif isinstance(y_placeholder, dict):
            y_placeholder_dict = y_placeholder

        self._summary['hp_accuracy'] = {}

        for op_key, op_val in y_placeholder_dict.items():
            ypred = self._forward_prop_outputs[op_key]
            y_true_cls = tf.argmax(y_placeholder_dict[op_key], axis=1)
            y_pred_cls_hp = tf.argmax(ypred["hp"], axis=1)
            correct_prediction_hp = tf.equal(y_true_cls, y_pred_cls_hp)
            hp_accuracy_val = tf.reduce_mean(
                tf.cast(correct_prediction_hp, tf.float32))
            self._summary['hp_accuracy'][op_key] = hp_accuracy_val

    # print('SUMMARY in hp accuracy', self._summary)

    def lp_accuracy(self, y_placeholder):

        if isinstance(y_placeholder, tf.Tensor):
            assert len(self._output_layers.items()) == 1
            output_layer_name = list(self._output_layers.keys())[0]
            y_placeholder_dict = {output_layer_name: y_placeholder}
        elif isinstance(y_placeholder, dict):
            y_placeholder_dict = y_placeholder

        self._summary['lp_accuracy'] = {}

        for op_key, op_val in y_placeholder_dict.items():
            ypred = self._forward_prop_outputs[op_key]
            y_true_cls = tf.argmax(y_placeholder_dict[op_key], axis=1)
            y_pred_cls_lp = tf.argmax(ypred["lp"], axis=1)
            correct_prediction_lp = tf.equal(y_true_cls, y_pred_cls_lp)
            lp_accuracy_val = tf.reduce_mean(
                tf.cast(correct_prediction_lp, tf.float32))
            self._summary['lp_accuracy'][op_key] = lp_accuracy_val

    def save_lp_model(self, path):
        def rep_weights(k):
            try:
                tf.assign(
                    self._layers_objects[k]['layer_obj'].hp_weights,
                    self._layers_objects[k]['layer_obj'].quantized_weights.
                    eval()).eval()
                print('weights replaced', k)
            except:
                pass

        def rep_bias(k):
            try:
                tf.assign(
                    self._layers_objects[k]['layer_obj'].hp_bias,
                    self._layers_objects[k]['layer_obj'].quantized_bias.
                    eval()).eval()
                print('bias replaced', k)
            except:
                pass

        for k in self._layers_objects.keys():
            rep_weights(k)
            rep_bias(k)

        sess = tf.get_default_session()
        saver = tf.train.Saver()
        saver.save(sess, path + 'model/lp.ckpt')

    def save_model_v2(self, path, int_model=False):
        # check for input placeholder
        # check for #outputs
        input_dict = {}
        for k, v in self._layers_objects.items():
            if v['inputs'] == []:
                input_dict[k] = v['layer_obj']

        output_dict = {}
        for k, v in self._output_layers.items():
            output_dict[k + '_hp'] = v['hp']
            output_dict[k + '_lp'] = v['lp']

        # print('Input dict', input_dict)
        # print('Output dict', output_dict)
        if int_model:
            for k in self._layers_objects.keys():
                layer_obj = self._layers_objects[k]['layer_obj']
                if isinstance(layer_obj, (gtc.Conv2D, gtc.Dense)):
                    minimum_val = tf.reduce_min(
                        tf.abs(
                            tf.squeeze(layer_obj.quantized_weights)))
                    layer_obj.quantized_weights /=  minimum_val

                    minimum_bias_val = tf.reduce_min(
                        tf.abs(tf.squeeze(layer_obj.quantized_bias)))
                    layer_obj.quantized_bias /= minimum_bias_val

            self.compile()
            sess = tf.get_default_session()

            tf.saved_model.simple_save(
                sess, export_dir=path, inputs=input_dict, outputs=output_dict)
        else:
            sess = tf.get_default_session()
            if os.path.exists(path):
                shutil.rmtree(path)
            builder = tf.saved_model.builder.SavedModelBuilder(path)
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.TRAINING],
                signature_def_map={
                    "model":
                    tf.saved_model.signature_def_utils.build_signature_def(
                        inputs=input_dict, outputs=output_dict)
                })
            builder.save()

    def save_model(self, path, int_model=False):
        # check for input placeholder
        # check for #outputs
        input_dict = {}
        for k, v in self._layers_objects.items():
            if v['inputs'] == []:
                input_dict[k] = v['layer_obj']

        output_dict = {}
        for k, v in self._output_layers.items():
            output_dict[k + '_hp'] = v['hp']
            output_dict[k + '_lp'] = v['lp']

        # print('Input dict', input_dict)
        # print('Output dict', output_dict)
        if int_model:
            for k in self._layers_objects.keys():
                layer_obj = self._layers_objects[k]['layer_obj']
                if isinstance(layer_obj, (gtc.Conv2D, gtc.Conv2D_bn, gtc.Dense)):
                    minimum_val = tf.reduce_min(
                        tf.abs(tf.squeeze(layer_obj.quantized_weights)))
                    layer_obj.quantized_weights /= minimum_val

                    minimum_bias_val = tf.reduce_min(
                        tf.abs(tf.squeeze(layer_obj.quantized_bias)))
                    layer_obj.quantized_bias /= minimum_bias_val

            self.compile()
            sess = tf.get_default_session()

            tf.saved_model.simple_save(
                sess, export_dir=path, inputs=input_dict, outputs=output_dict)
        else:
            sess = tf.get_default_session()
            if os.path.exists(path):
                shutil.rmtree(path)
            tf.saved_model.simple_save(
                sess, export_dir=path, inputs=input_dict, outputs=output_dict)

    def restore_pretrained_weight(self, pretrained_model):
        #[print(k, v['layer_obj']) for k, v in self._layers_objects.items()]
        # restorin weights
        for k, v in pretrained_model.weights.items():
            #assert self._layers_objects[k.split(
            #    '/')[1]]['layer_obj'].hp_weights, ("weights mismatch", k)
            print(k, v.shape)
            tf.assign(
                self._layers_objects[k.split('/')[1]]['layer_obj'].hp_weights,
                v,
                validate_shape=True).eval()

        for k, v in pretrained_model.biases.items():
            if self._layers_objects[k.split('/')
                                    [1]]['layer_obj'].hp_bias == False:
                raise ValueError('Bias for',
                                 k.split('/')[1],
                                 'layer is False in the model')
            tf.assign(
                self._layers_objects[k.split('/')[1]]['layer_obj'].hp_bias,
                v,
                validate_shape=True).eval()

        for k, v in pretrained_model.batchnorm.items():
            all_keys = k.split('/')

            if all_keys[1] == 'batch_normalization_22':
                continue

            if all_keys[2] == 'moving_mean':
                tf.assign(
                    self._layers_objects[k.split('/')[1]]['layer_obj'].
                    hp_moving_mean,
                    v,
                    validate_shape=True).eval()

            elif all_keys[2] == 'moving_variance':
                tf.assign(
                    self._layers_objects[k.split('/')[1]]['layer_obj'].
                    hp_moving_variance,
                    v,
                    validate_shape=True).eval()
            elif all_keys[2] == 'beta':
                tf.assign(
                    self._layers_objects[k.split('/')[1]]['layer_obj'].hp_beta,
                    v,
                    validate_shape=True).eval()
            elif all_keys[2] == 'gamma':
                tf.assign(
                    self._layers_objects[k.split('/')[1]]['layer_obj'].
                    hp_gamma,
                    v,
                    validate_shape=True).eval()
            else:
                raise ValueError('key error')
        print('Weights Restored')

    def load_pretrained_weights_from_reader(self, reader, sess=None, quant=False, skip_pretrained_bn_layers=False, missing_ok=False, verbose=True):
        '''
        :param reader:  reader should be from the class WeightsReader
        :param lp:  load low-precision weights and quantizers
        '''
        assert isinstance(reader, weights_readers.WeightsReader), \
            "Reader should be of class WeightsReader"

        all_pretrained_weights_list = reader.get_tensor_list()
        pretrained_prefix = reader.get_prefix()
        #with tf.Session() as sess:
        #    w = self._layers_objects['MobilenetV2/Conv/']['layer_obj'].hp_weights
        #    wp = reader.get_tensor('MobilenetV2/Conv/weights')
        #    tf.assign(w, wp, validate_shape=True).eval(session=sess)

            #print(w)
        print('Copying weights from %s' % reader.descrip() )

        combine_assignments = True # False for debugging. True for speed.
        all_shapes_match = True
        if sess is None:
            sess = tf.get_default_session()

        #with tf.Session() as sess:
        all_assignments = []

        conv_weight_names = ['kernel', 'depthwise_kernel', 'bias']
        bn_weight_names = ['beta', 'gamma', 'moving_mean', 'moving_variance']


        wgt_count = 0
        weights_missing = 0
        #for k, obj_dict in self._layers_objects.items():
        for layer_i, layer_name in enumerate(self._layer_names):

            obj_dict = self._layers_strings[layer_name]
            layer_obj = obj_dict['layer_obj']
            if not hasattr(layer_obj, 'get_hp_weights'):
                continue

            net_weights = layer_obj.get_hp_weights()

            if quant:
                lp_weights_assignable = [wgt for wgt in layer_obj.get_lp_weights() if hasattr(wgt, 'assign')]

                net_weights.extend(lp_weights_assignable)

                if layer_obj._quantizer is not None:
                    net_weights.extend(layer_obj._quantizer.trainable_parameters())


            for net_weight in net_weights:
                wgt_count += 1
                net_weight_name = net_weight.name.replace(':0', '')
                net_weight_name_fix = net_weight_name.replace(self._name, pretrained_prefix)

                if isinstance(layer_obj, gtc.Conv2D_bn): # assigns
                    assert '.' in net_weight_name_fix, "Fused Batchnorm name should contain a ."
                    full_layer_name, weight_name_full = net_weight_name_fix.split('/')
                    weight_name = weight_name_full.split(':')[0]
                    conv_name, bn_name = full_layer_name.split('.')
                    if weight_name in conv_weight_names:
                        net_weight_name_fix = conv_name + '/' + weight_name_full
                    elif weight_name in bn_weight_names:
                        net_weight_name_fix = bn_name + '/' + weight_name_full
                    else:
                        raise ValueError('Unrecognized weight name')

                weight_saved = None
                has_tensor = reader.has_tensor(net_weight_name_fix)

                if has_tensor:
                    weight_saved = reader.get_tensor(net_weight_name_fix)

                #elif net_weight_name_fix.endswith('biases'):
                #    weight_saved = np.zeros( net_weight.shape )
                #    has_tensor = True

                elif isinstance(layer_obj, (gtc.Conv2D_bn, gtc.BatchNormalization)) and skip_pretrained_bn_layers:
                    bn_zero_val_wgts = ['beta', 'moving_mean']
                    bn_one_val_wgts = ['gamma', 'moving_variance']

                    for wgt in bn_zero_val_wgts:
                        if wgt in net_weight_name_fix:
                            weight_saved = np.zeros(net_weight.shape)
                    for wgt in bn_one_val_wgts:
                        if wgt in net_weight_name_fix:
                            weight_saved = np.ones(net_weight.shape)

                    has_tensor = weight_saved is not None



                if has_tensor:
                    squeeze = lambda shp: tuple([d for d in shp if d != 1])

                    # Allow reshaping if copying weights from a dense layer to a 1x1 convolutional layer:
                    allow_reshaping = ('fc' in net_weight_name_fix or 'dense' in net_weight_name_fix) and isinstance(layer_obj, (gtc.Conv2D, gtc.Conv2D_bn))
                    target_shape = tuple(net_weight.shape.as_list())
                    shape_mismatch = target_shape != weight_saved.shape
                    just_missing_singletons = shape_mismatch and squeeze(target_shape) == squeeze(weight_saved.shape)

                    if shape_mismatch and (allow_reshaping or just_missing_singletons):
                        weight_saved = weight_saved.reshape(target_shape)

                    if verbose:
                        print(' (%3d) [Layer %3d] Copying %s --> %s   shape=%s' % (wgt_count, layer_i, net_weight_name_fix, net_weight.name, str(weight_saved.shape) ))

                    if shape_mismatch and not just_missing_singletons:
                        all_shapes_match = False
                        utl.cprint('   Warning: mismatch: shape of %s and %s: net=%s vs pre=%s ' % (
                                net_weight_name_fix, net_weight.name, target_shape, str(weight_saved.shape)), color=utl.Fore.RED)

                    assignment_op = tf.assign(net_weight, weight_saved, validate_shape=all_shapes_match)
                    if combine_assignments:
                        all_assignments.append( assignment_op )
                    elif all_shapes_match:
                        sess.run([assignment_op])


                    if isinstance(layer_obj, gtc.BatchNormalization) and not quant:
                        # If not copying over quantized weights as well, we also
                        # copy over weights to any low-precision weights that were not shared
                        # with the hp weights.

                        # for independent & shared:    elif isinstance(layer_obj, gtc.SharedBatchNorm):
                        indep_weights = layer_obj._not_share
                        for wgt in indep_weights:
                            wgt_keras_name = wgt.replace('running', 'moving')
                            if wgt_keras_name in net_weight_name_fix:
                                wgt_count += 1
                                lp_weight = getattr(layer_obj._batch_normalizer_lp, wgt_keras_name)
                                if verbose:
                                    print(' (%3d) [Layer %3d] Copying %s --> %s [LP, NOT SHARED WITH HP]   shape=%s' % (wgt_count, layer_i, net_weight_name_fix, lp_weight.name, str(weight_saved.shape)))

                                assignment_op = tf.assign(lp_weight, weight_saved, validate_shape=all_shapes_match)
                                if combine_assignments:
                                    all_assignments.append(assignment_op)
                                elif all_shapes_match:
                                    sess.run([assignment_op])


                else:
                    if not missing_ok:
                        print(' ** Reader has no tensor : %s  (originally:%s)' % (net_weight_name_fix, net_weight.name ))
                        weights_missing += 1
                    #raise ValueError("")


            if not hasattr(layer_obj, 'hp_weights'):
                continue

            #print('%s :  hp shape = %s.  pretrained shape = %s' % (k, str(layer_hp_weights.shape), str(pretrained_weights.shape) ))

            #tf.assign(layer_hp_weights,
            #    pretrained_weights,
            #    validate_shape=True).eval()
        print('Running assignments for %d weights/states (%d weights) [%d were missing] ...' % (
            len(all_assignments), wgt_count, weights_missing))
        if combine_assignments:
            if all_shapes_match:
                sess.run( all_assignments )
            else:
                raise ValueError('Not all tensors matched shape with pretrained weights.')

        print('Weights Restored')

    def copy_weights_from_model(self, otherModel, sess=None, verbose=True):
        if not self._compiled:
            print('Compiling first before we can copy')
            self.compile(verbose=True)
        # copy weights from another GTCModel
        assert isinstance(otherModel, gtc.GTCModel)
        other_model_weights, other_model_weights = otherModel._get_weights()
        other_model_name = otherModel._name
        other_model_weight_names = [wgt.name for wgt in other_model_weights]
        cur_model_weights = self._get_weights()

        assign_ops = []

        for wgt_i, cur_weight in enumerate(cur_model_weights):

            other_model_weight_name = cur_weight.name.replace(self._name, other_model_name)
            if other_model_weight_name in other_model_weight_names:
                idx = other_model_weight_names.index(other_model_weight_name)
                correponding_weight = other_model_weights[idx]

                assignment_op = tf.assign(cur_weight, correponding_weight,
                                          validate_shape=True)
                assign_ops.append(assignment_op)
                if verbose:
                    print('(%d/%d) Copying %s' % (
                        wgt_i+1, len(cur_model_weights), other_model_weight_name))
            else:
                utl.cprint('(%d/%d) Warning: No value for %s in %s model' % (
                    wgt_i+1, len(cur_model_weights),
                    other_model_weight_name, other_model_name), color=utl.Fore.RED )



        if sess is None:
            sess = tf.get_default_session()
            print('')
            sess.run(assign_ops)

        self._quantizers_initialized = True



    def initialize(self, sess=None, initialize_weights=False, only_uninitialized=True, verbose=True):
        self.initialize_all_zero_biases(sess, only_uninitialized, verbose)
        self.initialize_all_quantizers(sess, only_uninitialized, verbose)
        self.initialize_batchnorm_updates(sess, only_uninitialized, verbose)
        if initialize_weights:
            # Uses the weight initializers for all the trainable weights.
            # Warning: overwrites any pretrained weights loaded into the network
            # unless only_uninitialized == True
            self.initialize_weights(sess, only_uninitialized=only_uninitialized, verbose=verbose)

        # miscellaneous:
        self.initialize_misc_variables()

    def initialize_set_of_variables(self, var_list, sess=None, only_uninitialized=True, description='',verbose=True):

        if sess is None:
            sess = tf.get_default_session()

        if bool(description):
            description += ' '
        if len(var_list) == 0:
            print('No %svariables to initialize' % description)
            return

        #tf_initialized = sess.run([(tf.is_variable_initialized(var) if hasattr(var, 'is_initialized') else True) for var in var_list ])
        tf_initialized = sess.run([tf.is_variable_initialized(var)  for var in var_list])
        var_list_uninitialized = [v for (v, f) in zip(var_list, tf_initialized) if not f]
        var_list_initialized = [v for (v, f) in zip(var_list, tf_initialized) if f]

        if only_uninitialized:
            var_list_to_initialize = var_list_uninitialized
        else:
            var_list_to_initialize = var_list


        if only_uninitialized and len(var_list_uninitialized) < len(var_list):
            print('Skipping %d %sparams that were are already initialized' %
                  (len(var_list_initialized), description) )
            if verbose:
                z = [print(i + 1, v.name, ' (already initialized)') for i, v in enumerate(var_list_initialized)]

        if len(var_list_to_initialize) > 0:
            if not only_uninitialized and len(var_list_initialized) > 0:
                print('Warning: we may be overwriting %d params that were pre-loaded' % len(var_list_initialized))
            else:
                print('Initializing %d %sparams that were NOT already initialized' % (
                    len(var_list_to_initialize), description))

            if verbose:
                z = [print(i + 1, v.name, ' (UNinitialized)') for i, v in enumerate(var_list_uninitialized)]

            sess.run( tf.variables_initializer(var_list_to_initialize) )
        else:
            print('No unintialized %svariables to initialize' % description)


    def initialize_all_quantizers(self, sess=None, only_uninitialized=True, verbose=False):

        quant_var_list = self.get_quantization_variables()

        self.initialize_set_of_variables(
            quant_var_list, sess, only_uninitialized, 'quantizer', verbose)

        self._quantizers_initialized = True

    def initialize_all_zero_biases(self, sess=None, only_uninitialized=True, verbose=True):

        var_list = []
        for k, obj_dict in self._layers_objects.items():
            layer_obj = obj_dict['layer_obj']
            if isinstance(layer_obj, gtc.Conv2D) \
                and not layer_obj._use_bias and layer_obj.hp_bias is not None:
                # if is conv2d and use_bias=False but do have a bias term:
                var_list.append(layer_obj.hp_bias)

        self.initialize_set_of_variables(
            var_list, sess, only_uninitialized, 'zero-bias', verbose)

    def initialize_misc_variables(self, sess=None, only_uninitialized=True, verbose=True):

        all_vars = tf.global_variables()
        gamma_vars = [v for v in all_vars if
                      v.name.startswith(self._name) and
                      v.name.endswith('gamma_weights:0')]

        if len(gamma_vars) > 0:
            self.initialize_set_of_variables(
                gamma_vars, sess,only_uninitialized, 'l2-norm', verbose=verbose)

    def initialize_batchnorm_updates(self, sess=None, only_uninitialized=True, verbose=False):

        all_vars = tf.global_variables()
        bn_update_vars = [v for v in all_vars if
                          v.name.startswith(self._name) and
                          ('moving_mean' in v.name or 'moving_variance' in v.name) and
                          ('biased' in v.name or 'local_step' in v.name) ]

        self.initialize_set_of_variables(
            bn_update_vars, sess, only_uninitialized, 'batchnorm update', verbose=verbose)


    def initialize_weights(self, sess=None, only_uninitialized=True, verbose=False):
        if sess is None:
            sess = tf.get_default_session()

        all_weights = self._get_weights(hp=True, lp=True,
                                        trainable=True, non_trainable=True,
                                        assignable=True, non_assignable=False)

        self.initialize_set_of_variables(
            all_weights, sess, only_uninitialized, 'trainable weights', verbose)



    def _quantization_angle_diff_weights(self):
        quantization_angle_weights_dict = {}
        for k, v in self._layers_objects.items():
            if isinstance(v['layer_obj'], GTCLayer):
                if v['layer_obj'].quantization_angle_difference_for_weights(
                ) is not None:
                    quantization_angle_weights_dict[k] = v[
                        'layer_obj'].quantization_angle_difference_for_weights(
                        )
        return quantization_angle_weights_dict

    def _quantization_angle_diff_outputs(self):
        quantization_angle_diff_output_dict = {}
        for k, v in self._layers_objects.items():
            if isinstance(v['layer_obj'], GTCLayer):
                if v['layer_obj'].quantization_angle_difference_for_output(
                        self._forward_prop_inputs[k]) is not None:
                    quantization_angle_diff_output_dict[k] = v[
                        'layer_obj'].quantization_angle_difference_for_output(
                            self._forward_prop_inputs[k])

        return quantization_angle_diff_output_dict

    def _quantization_loss_weights(self):
        quantization_error_for_weights_dict = {}
        for k, v in self._layers_objects.items():
            if isinstance(v['layer_obj'], GTCLayer):
                if v['layer_obj'].quantization_error_for_weights() is not None:
                    quantization_error_for_weights_dict[k] = v[
                        'layer_obj'].quantization_error_for_weights()

        return quantization_error_for_weights_dict

    def _quantization_loss_outputs(self):
        quantization_error_for_outputs_dict = {}
        for k, v in self._layers_objects.items():
            if isinstance(v['layer_obj'], GTCLayer):
                if v['layer_obj'].quantization_error_for_output(
                        self._forward_prop_inputs[k]) is not None:
                    quantization_error_for_outputs_dict[k] = v[
                        'layer_obj'].quantization_error_for_output(
                            self._forward_prop_inputs[k])

        return quantization_error_for_outputs_dict

    def save_lp_model(self, path):
        quantized_weights_data = {}
        for k, v in self._layers_objects.items():
            if isinstance(v['layer_obj'], (gtc.Conv2D, gtc.Dense)):
                quantized_weights_data[k] = {}
                quantized_weights_data[k]['weights'] = v[
                    'layer_obj'].quantized_weights.eval()
                quantized_weights_data[k]['bias'] = v[
                    'layer_obj'].quantized_bias.eval()
        new_path = path + "quantized_weights.npy"
        # print('quantized weights', quantized_weights_data)
        np.save(new_path, quantized_weights_data)

    def eval_batch(self, data_dict):
        k = list(self._summary['lp_accuracy'].keys())[0]
        print('lp accuracy', self._summary['lp_accuracy'][k].eval(data_dict))
        print('hp_accuracy', self._summary['hp_accuracy'][k].eval(data_dict))

    def run_updates(self, sess):
        all_updates, all_per_input_updates = self._get_layer_updates()
        sess.run(all_updates)
        pass

    def forward(self, data, hp=True, sess=None):
        data_dict = self.validate_input(data)
        if sess is None:
            sess = tf.get_default_session()

        num_outputs = len(self._output_layers)

        all_outputs = {}
        for k, v in self._output_layers.items():
            if 'layer_obj' in v:
                v = v['layer_obj']
            if 'hp' in v:
                if hp:
                    v = v['hp']
                else:
                    v = v['lp']

            all_outputs[k] = sess.run( v, feed_dict=data_dict)

            if num_outputs == 1:
                k = list(self._output_layers.keys())[0]
                all_outputs = all_outputs[k]

        return all_outputs


    def eval_one_example(self, data_dict, hp=True, lp=False, top=1):
        num_outputs = len(self._output_layers)

        if self.nettask == 'classification':
            output = self.forward(data_dict)
            predicted_class = {}

            for k, v in self._output_layers.items():
                if 'layer_obj' in v:
                    v = v['layer_obj']
                if 'hp' in v:
                    if hp:
                        v = v['hp']
                    else:
                        v = v['lp']

                op_softmax = tf.nn.softmax(v)
                if top==1:
                    op_class = tf.argmax(op_softmax, axis=-1)
                else:
                    op_class = tf.nn.top_k(op_softmax, k=top).indices

                predicted_class[k] = op_class.eval(data_dict)

            if num_outputs == 1:
                k = list(self._output_layers.keys())[0]
                predicted_class = predicted_class[k]

            return predicted_class

        elif self.nettask == 'detection':
            predicted_class = {}

            k = None
            for k, v in self._output_layers.items():
                if 'layer_obj' in v:
                    v = v['layer_obj']
                if 'hp' in v:
                    if hp:
                        v = v['hp']
                    else:
                        v = v['lp']

                op_softmax = tf.nn.softmax(v)
                if top == 1:
                    op_class = tf.argmax(op_softmax, axis=-1)
                else:
                    op_class = tf.nn.top_k(op_softmax, k=top).indices

                predicted_class[k] = op_class.eval(data_dict)

            if num_outputs == 1:
                predicted_class = predicted_class[k]

            return predicted_class

    def predict_proba(self, data_dict):
        proba = {}
        for k, v in self._output_layers.items():
            op_softmax = tf.nn.softmax(v['layer_obj'])
            proba[k] = op_softmax.eval(data_dict)
        return proba



    def predict_class(self, input, hp=True, top_k=None):
        logits = self.forward(input, hp)

        if top_k is None:
            output_tf = tf.nn.softmax(logits)
        else:
            if top_k == 1:
                output_tf = tf.argmax(logits, axis=-1)
            else:
                output_tf = tf.nn.top_k(logits, k=top_k).indices

        output = output_tf.eval()
        return output

    def predict_detections(self, input, hp=True):
        detections = self.forward(input, hp)
        return detections

    def predict(self, input, hp=True, top_k=None):

        if self.nettask == 'detection':
            return self.predict_detections(input, hp)
        elif self.nettask == 'classification':
            return self.predict_class(input, hp, top_k=top_k)


    def evaluate(self, input, gt_output, hp=True, top_k=None):

        if self.nettask == 'detection':
            pred = self.predict_detections(input, hp)

        elif self.nettask == 'classification':
            pred = self.predict_class(input, hp, top_k=top_k)
            if top_k is None:
                top_k = 1
            if not isinstance(top_k, list):
                top_k = [top_k]

            top_k_accuracies = []
            for k in top_k:
                if k == 1:
                    acc = keras.metrics.categorical_accuracy(pred, gt_output)
                else:
                    acc = keras.metrics.top_k_categorical_accuracy(pred, gt_output, k=k)
                top_k_accuracies.append(acc)

            if len(top_k) == 1:
                top_k_accuracies = top_k_accuracies[0]

            return top_k_accuracies

        else:
            raise ValueError('invalid network task')

    # Edit: Using a 'training' flag for dynamic training/testing does not work, just 
    # like using the keras learning_phase()
    #def set_learning_phase(self, value):
    #    keras.backend.set_value(self._learning_phase, value)

    #def get_learning_phase(self, evaluate=True):
    #    if evaluate:
    #        return keras.backend.get_value(self._learning_phase)
    #    else:
    #        return self._learning_phase

    def get_placeholder(self):

        if self._compiled_keras:
            placeholder = self._x_placeholder_keras
        else:
            placeholder = self._x_placeholder

        assert placeholder is not None, "No placeholder defined for the network"
        return placeholder

    def validate_input(self, input):
        placeholder = self.get_placeholder()
        if isinstance(input, np.ndarray):
            input_dict = {placeholder: input}
            return input_dict
        #elif isinstance(input, tf.Tensor):
        #    assert 'Placeholder' not in input.name
        #    input_dict = {self._x_placeholder: input}
        #    return input_dict
        elif isinstance(input, dict) and self._x_placeholder in input:
            return input
        else:
            raise ValueError('Invalid input %s' % str(input))

def singlePrecisionKerasModel(model_file, weights_file, hp=None,
                              add_softmax=None, verbose=True):
    # Create a single-precision (ie. HP only, or LP only) keras model
    # using the saved model_def.json created by save_model_def()
    # and saved weights_hp.h5 from a gtcModel, created using
    # save_weights_for_keras_model(... hp=True)  (HP model) or
    # save_weights_for_keras_model(... hp=False) (LP model)

    # hp: set To True if you are building an HP model, or False if you are building an LP model
    #   If you are providing a weights file, you can leave it as None and
    #   it will automatically detect whether it is an HP/LP network based on
    #   whether there are quantizers in the saved weights file (the quantizers are only
    #   added to the LP file)

    # add_softmax: set to True if you want to add a softmax at the top of a
    #   classification model. Set to False if you want the raw logits, or if
    #   you are building an SSD model. Defaults to None, which adds a softmax
    #   to classification models but not SSD models
    #   (if we see any anchor box layers, we know it is an SSD model)
    if hp is None:
        assert bool(weights_file), "Specify HP=True/False if not providing a weights file"

    with open(model_file) as f:
        model_def = json.load(f)

    model_name = model_def['model_attributes']['_name']

    orig_name = model_name
    model_name = net.get_unique_tf_prefix(orig_name)

    try_load_by_order_if_different_name = True
    if orig_name != model_name and not try_load_by_order_if_different_name:
        raise ValueError(
            'There already exists a variable namespace "%s" in this tensorflow '
            'session. \n Renaming all the variable layers to adjust to a new name '
            'is not currently supported. \nPlease make sure that no other models/variables with '
            'this name scope are created before creating this model. ' % orig_name)
    #model = keras
    # = net.get_unique_tf_prefix(orig_name)

    # Load model layers
    model_layers = model_def['model_config']
    net.destringify_model_specs(model_layers)
    ssd_config = None

    # Keep a dictionary of quantizers to facilitate sharing quantizers
    # between layers
    quantizers_dict = OrderedDict()

    is_SSD_network = False
    have_weights_file = weights_file is not None and os.path.exists(weights_file)
    quant_params_dict = {}
    if bool(weights_file):
        if not os.path.exists(weights_file):
            raise ValueError('Weights file %s does not exist' % weights_file)

        with h5py.File(weights_file, 'r') as f:
            # If HP is None, detect whether is a HP/LP model based on whether there is a
            # group of quantizers in the h5 file.
            if hp is None:
                hp = not 'quantizers' in f
                if verbose:
                    if hp:
                        print('No quantizers in the weights file. This a HP model')
                    else:
                        print('Found quantizers in the weights file. This a LP model')

            else:
                # Make sure hp argument matches the type of file (HP vs LP)
                if hp and 'quantizers' in f:
                    raise ValueError('Specified HP model but found quantizers in the weights file.')
                if not hp and not 'quantizers' in f:
                    raise ValueError('Specified LP model but did not find any quantizer group in the weights file.')

            if not hp:
                if 'quantizers' in f:
                    quant_group = f['quantizers']
                    quant_types = list(quant_group.keys())
                    for quant_type in quant_types:
                        quant_params_dict[quant_type] = utl.h5py_to_dict(quant_group[quant_type])

    layer_outputs_dict = OrderedDict()
    # layer_names = list(model_config.keys())
    input,x = None, None
    for i, (layer_name, layer_specs) in enumerate(model_layers):

        if layer_specs['type'] == 'Placeholder':
            input = keras.layers.Input(batch_shape=layer_specs['shape'],
                                       dtype=layer_specs['dtype'])
            x = input

            if verbose:
                print('Added Layer %d Placeholder [%s]' % (i + 1, str(layer_specs['shape'])))
        else:
            layer_type_name = layer_specs['type']
            layer_config = layer_specs['config']
            # handle 'Conv2d' and older 'gtc.conv2d.Conv2d'
            if 'Conv2d' in layer_type_name and layer_config['filters'] == 'depthwise':
                # In GTCModels, standard and depthwise convolutions are implemented using
                # the same underlying type (Conv2D). With keras models, they are distinct
                layer_type_name = 'DepthwiseConv2D'
                layer_config.pop('filters')
            layer_type = gtc.from_str(layer_type_name, keras=True)

            # if we have renamed the network to avoid clashes with existing network in memory
            # propagate this name change to the layers in the network.
            '''
            if (orig_name != self._name):
                if 'name' in layer_config.keys():
                    if orig_name in layer_config['name']:
                        layer_config['name'] = layer_config['name'].replace(orig_name, self._name)
                    else:
                        layer_config['name'] = self._name + layer_config['name']
                if bool(layer_config.get('layer_inputs', False)):
                    layer_config['layer_inputs'] = [
                        nm.replace(orig_name, self._name) for nm in layer_config['layer_inputs']]
            '''
            kwargs_ignore = ['bias_quantizer', 'share']
            for k in kwargs_ignore:
                layer_config.pop(k, None)

            quantizer = None
            quantizer_name, quant_type_name = '', ''
            param_list = []
            quant_kwargs = {}
            if 'quantizer' in layer_config and not hp:
                quantizer_name = layer_config['quantizer']

                quant_type, quant_name = quantizer_name.split('.')
                if quant_type == 'linear':
                    quant_type_name = 'LinearQuantization'
                    param_list = ['slope']
                    var_list = ['num_bits']
                elif quant_type == 'gtc':
                    quant_type_name = 'GTCQuantization'
                    param_list = ['slope', 'intercept']
                    var_list = param_list
                elif quant_type == 'identity':
                    quant_type_name = 'IdentityQuantization'
                    param_list = []
                    var_list = []
                else:
                    raise ValueError('Unhandled quantizer')

                quant_kwargs = dict(variable_scope=model_name)
                for j, param in enumerate(param_list):
                    param_key = '/' + quant_name + '/' + param + ':0'
                    quant_kwargs[var_list[j]] = quant_params_dict[quant_type_name][param_key]

                if quantizer_name not in quantizers_dict:
                    quantizer = quant.from_str(quantizer_name, quant_kwargs)
                    quantizers_dict[quantizer_name] = quantizer
                else:
                    quantizer = quantizers_dict[quantizer_name]


            layer_config.pop('quantizer', None)

            layer_inputs = layer_config.pop('layer_inputs', None)
            keras_layer = layer_type.from_config(layer_config)

            if 'AnchorBoxes' in layer_type_name:
                is_SSD_network = True
                ssd_config = layer_config['ssd_config']


            if layer_inputs is not None:
                x_input = [layer_outputs_dict[name] for name in layer_inputs]
                if len(x_input) == 1:
                    x_input = x_input[0]
            else:
                x_input = x

            x = keras_layer(x_input)
            assert K.is_keras_tensor(x), "must be a keras tensor"

            if isinstance(x_input, list):
                input_shape_strs = [str(x._shape_tuple()) for x in x_input]
                in_shape_str = utl.concat_detect_repeats(input_shape_strs)
            else:
                in_shape_str = str(x_input._shape_tuple())
            out_shape_str = str(x._shape_tuple())

            if verbose:
                print('Added Layer %d [%s]  %s --> %s' % (i + 1, layer_type, in_shape_str, out_shape_str))

            if not hp:
                skip_identity_quant = True
                if quantizer is not None and isinstance(keras_layer, keras.layers.Activation) \
                    and not (isinstance(quantizer, quant.IdentityQuantization) and skip_identity_quant):
                    # LP network: Add activation quantizers in a Lambda wrapper:
                    # (weight quantization is handled separately by saving
                    # lp weights into the saved weights.h5 file)

                    x = keras.layers.Lambda(lambda xx: quantizer.quantize(xx), name=model_name + quantizer_name)(x)

                    if verbose:
                        param_str = '{' + ', '.join(['%s=%s' % (k,quant_kwargs[k]) for k in param_list]) + '}'
                        print('Added Layer %d %s for layer activations: ["%s"] %s' % (
                            i+1, quant_type_name, quantizer_name, param_str))

            layer_outputs_dict[layer_config['name']] = x

    if add_softmax is None:
        # Default: add if is a classication model (no anchor boxes)
        add_softmax = not is_SSD_network

    if add_softmax:
        #x = keras.layers.Activation('softmax')(x)
        x = keras.layers.Softmax()(x)

    keras_model = keras.models.Model(input, x,
                                     name=model_name)

    if have_weights_file:
        by_name = orig_name == model_name
        if verbose:
            print('Loading weights from %s ' % weights_file)
            if not by_name:
                print('There was already a model named %s, so this model was named %s. '
                      'So we will have to load weights by topological order instead of by name' % (orig_name, model_name))
        keras_model.load_weights(weights_file, by_name=by_name)

    if is_SSD_network:
        keras_model.ssd_config = ssd_config

    return keras_model
