# sys.path.append(Path to repository)
import numpy as np
import cv2
import os
import argparse
import time
import ssd.ssd_utils as ssd_utl
#from ssd.ssd_eval_utils import decode_y
#from datasets.ssd_data_loader import BatchGenerator
from tqdm import tqdm
import ssd.ssd_info as ssd
import gtc
import imagesize
import imageio
import ssd.post_processing_tf.visualization_utils as vis_util
from utility_scripts.sample_image_loader import SampleImageLoader
from utility_scripts import misc_utl as utl
import tensorflow as tf
import keras
import shutil
from collections import OrderedDict
import keras.backend as K
from datasets.pascal import GTCPascalLoader
from datasets.coco import GTCCocoLoader
from ssd.average_precision_evaluator import Evaluator
import warnings
import utility_scripts.lock_utils as lock
from utility_scripts import weights_readers as readers
from utility_scripts import misc_utl as utl
# TODO: Specify the directory that contains the `pycocotools` here.

from contextlib import redirect_stdout
import io
import re

try:
    # On Linux, install with git clone https://github.com/cocodataset/cocoapi && cd cocoapi/PythonAPI && python setup.py build_ext install
    # On Windows, install with pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    have_pycocotools = True
except ImportError:
    COCO, COCOeval = None, None
    have_pycocotools = False


def evaluate_model_on_dataset(model, dataset_specs=None, hp=True, verbose=True,
                              save_results_filename=None, batch_size=32):

    if isinstance(dataset_specs, str):
        if dataset_specs == 'pascal':
            dataset_specs = ssd.SSDDatasetType('pascal', 2007, 'test')
        elif dataset_specs == 'coco':
            dataset_specs = ssd.SSDDatasetType('coco', 2014, 'minival')

    if dataset_specs is None:  # assume pascal-voc
        dataset_specs = ssd.SSDDatasetType('pascal', 2007, 'test')

    if isinstance(dataset_specs, list):
        dataset_specs = dataset_specs[0]

    dataset_name = dataset_specs.name

    if isinstance(model, keras.models.Model) and hasattr(model, 'gtcModel'):
        model = model.gtcModel

    bit_precision = None
    if isinstance(model, gtc.GTCModel):
        bit_precision = 'hp' if hp else 'lp'

    #if isinstance(model, gtc.GTCModel) and model._compiled_keras:
    #    model = model.get_keras_model(hp)


    input_layer = model.layers[0]
    if isinstance(input_layer, keras.layers.InputLayer):
        # standard input layer of a keras model
        model_input_shape = input_layer.input_shape
    elif isinstance(input_layer, tf.Tensor) and input_layer.op.type == 'Placeholder':
        # input placeholder of a GTC model
        model_input_shape = input_layer._shape_as_list()
    else:
        raise ValueError('Unhandled input layer type')


    img_height = model_input_shape[1]
    img_width = model_input_shape[2]
    #imsize = (img_width, img_height)

    ssd_config = model.ssd_config

    if verbose:
        print('score_thresh', ssd_config.score_thresh)
        print('nms_iou_thresh', ssd_config.nms_iou_thresh)
        print('max_size_per_class', ssd_config.max_size_per_class)
        print('max_total_size', ssd_config.max_total_size)
        print('matching_iou_thresh', ssd_config.matching_iou_thresh)
        print('decoding_border_pixels', ssd_config.decoding_border_pixels)
        print('img_height', ssd_config.img_height)
        print('img_width', ssd_config.img_width)
        print('pred_coords', ssd_config.pred_coords)
        print('normalize_coords', ssd_config.normalize_coords)


    years = dataset_specs.years
    if dataset_specs.name == 'pascal':
        dataset = GTCPascalLoader(years=years, split=dataset_specs.split,
                                           verbose=verbose)
    elif dataset_specs.name == 'coco':
        dataset = GTCCocoLoader(years=years, split=dataset_specs.split,
                                         verbose=verbose,
                                         condensed_labels=ssd_config.condensed_labels)
        if utl.onLaptop():
            pass
            #dataset.clip_dataset_size(256)
        else:
            pass
            #dataset.clip_dataset_size(512)

    else:
        raise ValueError('Test on Pascal or COCO datasets only')

    #  2. Create a data generator for the evaluation dataset
    #dataset = GTCPascalLoader(years=[2007],
    #                          split='test', imsize=None)

    dataset.build(img_width=img_width, img_height=img_height)

    class_names = dataset.class_names
    n_classes = len(class_names)
    img_height = dataset.img_height
    img_width = dataset.img_width
    class_names_keys = [cls.lower().replace(' ', '') for cls in class_names]



    results = OrderedDict()
    map_results_strs = []
    if dataset_name == 'coco' and not have_pycocotools:
        warnings.warn('No "pycocotools" found. Will use the standard Pascal evaluation code even though we evaluating on the COCO dataset')

    if dataset_name == 'coco' and have_pycocotools:
        # TODO: Set the paths to the dataset here.
        datasets_info = ssd.get_datasets_info(dataset_specs.spec_list())
        images_dirs = datasets_info['image_dirs']
        annotations_filenames = datasets_info['annot_files']
        assert len(images_dirs) == 1
        assert len(annotations_filenames) == 1
        annotations_filename = annotations_filenames[0]

        # TODO: Set the desired output file name and the batch size.
        results_file = 'detections_val2017_ssd300_results.json'
        batch_size = 20  # Ideally, choose a batch size that divides the number of images in the dataset.

        #  We need the `classes_to_cats` dictionary. Read the documentation of this function to understand why.
        cats_to_classes, classes_to_cats, cats_to_names, classes_to_names = ssd_utl.get_coco_category_maps(annotations_filename)

        results_array = ssd_utl.predict_all_to_json(
            out_file=None,
            model=model,
            ssd_config=ssd_config,
            img_height=img_height,
            img_width=img_width,
            classes_to_cats=classes_to_cats,
            data_generator=dataset,
            batch_size=batch_size,
            data_generator_mode='resize',
            model_mode='training',
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            normalize_coords=True,
            numpy_output=True)


        coco_gt = COCO(annotations_filename)
        coco_dt = coco_gt.loadRes(results_array)
        image_ids = sorted(coco_gt.getImgIds())


        cocoEval = COCOeval(cocoGt=coco_gt,
                            cocoDt=coco_dt,
                            iouType='bbox')
        cocoEval.params.imgIds = image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        f = io.StringIO()
        with redirect_stdout(f):
            # Strangely, the cocoEval does not provide a direct way to return the mAP results.
            # It insists on printing the results to the screen. To capture the results, we
            # have to temporarily redirect std output to the StringIO object f while running
            # 'summarize()' to display the mAP results.
            cocoEval.summarize()

        eval_output = f.getvalue()
        eval_lines = eval_output.split('\n')
        # We want to capture the number at the end of this line, with IoU=0.50: eg.
        # 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.432'
        pattern = 'IoU=0.50 [\w\W]* = ([.\d]+)'
        #re.search(' ([\w\W]*) = ([.\d]+)', x).groups()
        mAP_IoU50 = [float(re.search(pattern, x).groups()[0]) for x in eval_lines if re.search(pattern, x)]
        results = { 'mAP': mAP_IoU50,
                    'all_results': eval_output }

        map_results_strs.extend(eval_lines)

    else:

        # 3. Run the evaluation
        evaluator = Evaluator(model=model,
                              ssd_config=ssd_config,
                              data_generator=dataset,
                              model_mode='training',
                              bit_precision=bit_precision,
                              verbose=verbose)

        evaluation_modes = ['sample', 'integrate']
        for eval_i, eval_mode in enumerate(evaluation_modes):

            if eval_i == 0:

                results_this_mode = evaluator(
                    img_height=img_height,
                    img_width=img_width,
                    batch_size=batch_size,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=ssd_config.matching_iou_thresh,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode=eval_mode,
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True)

                mean_average_precision, average_precisions, precisions, recalls = results_this_mode

            else:

                average_precisions = evaluator.compute_average_precisions(
                    mode=eval_mode,
                    num_recall_points=11,
                    verbose=verbose,
                    ret=True)

                mean_average_precision = evaluator.compute_mean_average_precision(ret=True)

            ap_results = OrderedDict()
            for i in range(1, len(average_precisions)):
                ap_results[class_names_keys[i]] = np.round(average_precisions[i], 4)

            results['AP_%s_classes' % eval_mode] = ap_results
            results['mAP_%s' % eval_mode] = mean_average_precision


            precision_str = '' if bit_precision is None else ' (' + bit_precision + ')'
            #print('Keras learning phase : %s' % str(keras.backend.learning_phase()) )
            s_header = 'AP evaluation mode : {} {}'.format(eval_mode, precision_str)
            s_classes = []
            for i in range(1, len(average_precisions)):
                s_classes.append("{:<14}{:<6}{}".format(class_names[i], 'AP', round(average_precisions[i], 3)))
            s_summary = "{:<14}{:<6}{}\n\n".format(eval_mode, 'mAP%s' % precision_str, round(mean_average_precision, 3))

            map_results_strs.extend([s_header] + s_classes + [s_summary])
            if verbose:
                print(s_header)
                [print(s) for s in s_classes]
                print(s_summary)


            #if utl.onLaptop() and False:
            #    evaluator.write_predictions_to_txt(out_file_prefix='F:/SRI/bitnet/mobilenet_ssd_300_eval_train_True_v2_predictions.txt')

        if dataset_specs.years[0] == 2007:
            evaluation_mode_final = 'sample'
        else:
            evaluation_mode_final = 'integrate'
        results['mAP'] = results['mAP_' + evaluation_mode_final]

    # endif dataset is coco/pascal

    bits_strs = []
    if isinstance(model, gtc.GTCModel):
        bits_per_layer = model.get_bits_per_layer(evaluate=True)

        s_header = '\n ** Bits Per Layer **'
        bits_strs.append(s_header)
        if verbose:
            print(s_header)

        mean_bits_per_layer = model.get_mean_bits_per_layer(evaluate=True)
        for i, (layer_name, num_bits) in enumerate(bits_per_layer.items()):
            s = 'Layer %d : num_bits = %6.3f :  %s' % (i + 1, num_bits, layer_name)
            bits_strs.append(s)
            if verbose:
                print(s)

        if bool(bits_per_layer):
            s_mean = '--- Mean bits per layer : %6.3f' % (mean_bits_per_layer)
            bits_strs.append(s_mean)
            results['mean_bits_per_layer'] = mean_bits_per_layer
            if verbose:
                print(s_mean)

    if save_results_filename is not None:
        with open(save_results_filename, 'w') as f:

            if save_results_filename is not None:
                with open(save_results_filename, 'w') as f:
                    for s in map_results_strs:
                        f.write(s + '\n')
                    for s in bits_strs:
                        f.write(s + '\n')

    return results




class calculate_mAP_callback(keras.callbacks.Callback):
    """Callback to calculate mAP at the end of each training epoch,
    and record the results in the logs"""

    def __init__(self, dataset_specs, epoch_freq=1, run_at_train_begin=False,
                 hp=True, lp=True, color=None, calc_in_test_phase=False,
                 mAP_per_bits_key=None,
                 use_testing_network=True, test_model_config=None,
                 **mAP_kwargs):
        # use_testing_network: use a separate network (built in keras TEST phase)
        # to do the mAP calculations.
        #   Since the training network will be using per-batch batch-normalization
        #   (built in TRAINING phase), and since dynamic learning phase hasn't worked
        #   for me, we can use a separate network (instantiated using keras TEST phase)
        #   whose weights are copied from that of the regular model before each mAP calculation

        self.dataset_specs = dataset_specs
        if not isinstance(epoch_freq, (int, list)):
            raise ValueError('epoch_freq should be an int or a list')

        # print('Epoch freq: %s' % epoch_freq)
        self.epoch_freq = epoch_freq
        self.run_at_train_begin = run_at_train_begin
        self.hp = hp
        self.lp = lp
        self.dataset_split = dataset_specs[0].split
        self.dataset_str = ','.join([str(spec) for spec in self.dataset_specs])
        if color is None:
            color = utl.Fore.RED
        self.color = color
        self.print = lambda s : utl.cprint(s, color=self.color)
        self.calc_in_test_phase = calc_in_test_phase
        self.mAP_per_bits_key = mAP_per_bits_key
        self.use_testing_network = use_testing_network
        self.test_model_config = test_model_config
        if use_testing_network:
            assert test_model_config is not None
            print('Building testing model using %s' % test_model_config)
            current_learning_phase = K.learning_phase()
            if self.calc_in_test_phase:
                K.set_learning_phase(False)

            self.test_model = gtc.GTCModel(model_file=test_model_config)

            K.set_learning_phase(current_learning_phase)

        else:
            self.test_model = None

        self.mAP_kwargs = mAP_kwargs

        super(calculate_mAP_callback, self).__init__()


    def on_train_begin(self, logs=None):
        if self.run_at_train_begin:
            self.evaluate_map(-1, logs=None)

    def on_epoch_end(self, epoch, logs=None):
        self.evaluate_map(epoch, logs=logs)

    def evaluate_map(self, epoch, logs=None):
        logs = logs or {}

        model_use = self.model
        if self.test_model is not None:
            model_use = self.test_model
            model_load_from = self.model
            if hasattr(self.model, 'gtcModel'):
                model_load_from = self.model.gtcModel
            weights_reader = readers.ModelWeightsReader(model_load_from)
            model_use.load_pretrained_weights_from_reader(weights_reader, verbose=False, quant=self.lp)


        calc_now = False
        if isinstance(self.epoch_freq, int):
            calc_now = (epoch % self.epoch_freq == 0) or (epoch == -1)
        elif isinstance(self.epoch_freq, list):
            calc_now = epoch in self.epoch_freq

        # always do on first epoch (epoch==0), so it creates a header in the log file
        # calc_now = epoch == 0 or
        hp_key_int = 'mAPi_%s_hp' % self.dataset_split
        hp_key_samp = 'mAPs_%s_hp' % self.dataset_split
        lp_key_int = 'mAPi_%s_lp' % self.dataset_split
        lp_key_samp = 'mAPs_%s_lp' % self.dataset_split

        map_per_bits_key = None
        mAP_logs = OrderedDict()
        save_mAP_per_bit = self.mAP_per_bits_key is not None

        if save_mAP_per_bit:
            map_per_bits_key = self.mAP_per_bits_key  + '_per_bit'

        if not calc_now:
            if self.hp:
                mAP_logs[hp_key_int] = -1
                mAP_logs[hp_key_samp] = -1
            if self.lp:
                mAP_logs[lp_key_int] = -1
                mAP_logs[lp_key_samp] = -1

            if save_mAP_per_bit:
                mAP_logs[map_per_bits_key] = -1

            logs.update(mAP_logs)
            return

        map_strs = []
        can_do_lp = hasattr(model_use, 'gtcModel') or isinstance(model_use, gtc.GTCModel)
        precisions_do = [prec for prec, tf in  zip(['hp', 'lp'], [self.hp, (self.lp and can_do_lp)]) if tf]
        self.print('Calculating mAP for %s ...' % ' + '.join(precisions_do))

        if self.calc_in_test_phase:
            # Switch to test phase to allow batch-norm parameters to use
            # running mean/variance instead of batch-wise mean/variance
            #K.set_learning_phase(False)
            #self.model.gtcModel.set_learning_phase(False)
            #print(" ** Set learning phase = False ...")
            # Note: dynamic learning phase does not actually work, so this is commented out
            # Instead, a separate network is used to do the mAP calculations
            pass

        mAP_kwargs = self.mAP_kwargs.copy()
        verbose = mAP_kwargs.pop('verbose', False)
        #if epoch == -1:
        #    verbose = True
        if self.hp:
            t1 = time.time()
            #mAP_hp_results = dict(mAP_sample=100, mAP_integrate=100)
            mAP_hp_results = evaluate_model_on_dataset(
                model_use, self.dataset_specs, hp=True, **mAP_kwargs,
                verbose=verbose)

            mAP_hp_integrate = mAP_hp_results['mAP_integrate']
            mAP_hp_sample = mAP_hp_results['mAP_sample']


            if logs is not None:
                mAP_logs[hp_key_int] = mAP_hp_integrate
                mAP_logs[hp_key_samp] = mAP_hp_sample

            t_hp = time.time() - t1
            map_strs.append('HP: %.3f (sample) / %.3f (integrate) [%.1f sec]' % (mAP_hp_sample, mAP_hp_integrate, t_hp) )


        if self.lp and can_do_lp:
            t2 = time.time()
            #mAP_lp_results = dict(mAP_sample=100, mAP_integrate=100)
            mAP_lp_results = evaluate_model_on_dataset(
                model_use, self.dataset_specs, hp=False, **self.mAP_kwargs)
            mAP_lp_integrate = mAP_lp_results['mAP_integrate']
            mAP_lp_sample = mAP_lp_results['mAP_sample']

            if logs is not None:
                mAP_logs[lp_key_int] = mAP_lp_integrate
                mAP_logs[lp_key_samp] = mAP_lp_sample


            t_lp = time.time() - t2
            map_strs.append('LP: %.3f (sample) / %.3f (integrate) [%.1f sec]' % (mAP_lp_sample, mAP_lp_integrate, t_lp) )


        logs.update(mAP_logs)
        #print(mAP_logs)


        if save_mAP_per_bit:
            bits_per_layer = None
            if 'val_bits_per_layer' in logs:
                bits_per_layer = logs['val_bits_per_layer']
            elif hasattr(self.model, 'gtcModel'):
                bits_per_layer = self.model.gtcModel.get_mean_bits_per_layer(evaluate=True)
            else:
                warnings.warn('Number of bits is unavailable. cannot calculate mAP per bits')

            if bits_per_layer is not None:
                # print('bits_per_layer', bits_per_layer)
                map_per_bits_key = self.mAP_per_bits_key + '_per_bit'

                mAP_per_bits_dict = {k:mAP_logs[k]/bits_per_layer for k in mAP_logs.keys()}
                # print('mAP_per_bits_dict', mAP_per_bits_dict)

                logs[map_per_bits_key] = mAP_per_bits_dict[self.mAP_per_bits_key]

            else:
                logs[map_per_bits_key] = -1


        map_str = ',  '.join(map_strs)
        self.print('mAP on %s : %s' % (self.dataset_str, map_str ))


        if self.calc_in_test_phase:
            #K.set_learning_phase(True)
            #self.model.gtcModel.set_learning_phase(True)
            #print(" ** Set learning phase = True ...")
            # Note: dynamic learning phase does not actually work, so this is commented out
            pass


    def __str__(self):
        precisions_test = [pr for pr,tf in zip(['HP', 'LP'], [self.hp, self.lp]) if tf]
        hp_lp_str = ' and '.join(precisions_test)
        freq_str = ''
        if isinstance(self.epoch_freq, int):
            freq_str = 'every %d epochs' % self.epoch_freq
        elif isinstance(self.epoch_freq, list):
            freq_str = 'on epochs %s' % str(self.epoch_freq)

        s = 'Calculate_mAP_callback on %s [%s] %s' % (
            self.dataset_str, hp_lp_str, freq_str)
        return s


def get_arg(arg_value, value_if_none):
    if arg_value is not None:
        if isinstance(value_if_none, bool):
            arg_value = bool(arg_value)
        return arg_value
    else:
        return value_if_none






def test_ssd_model(model_file, weights_file, hp=True, lp=True, save_results=False,
                   verbose=True, redo=False):

    all_prec = [prec for prec, do in zip(['hp', 'lp'], [hp, lp]) if do]
    #redo = False
    #tf.reset_default_graph()
    # create a new graph
    #K.clear_session()
    #with tf.Session() as sess:

    model = None
    K.set_learning_phase(False)

    for prec in all_prec:
        hp_i = prec == 'hp'
        results_file = weights_file.replace('.h5', '_mAP_%s.txt' % prec)

        if os.path.exists(results_file) and not redo:
            if utl.fileOlderThan(weights_file, results_file):
                print('    Results file %s already exists! [and is not too old]' % (results_file))
                continue
            else:
                print('    Results file %s is older than weights file. Recalculating!' % (results_file))

        save_file = results_file if save_results else None

        '''
        save_activations_now = False
        if save_activations_now:
            sample_image = net_utl.get_sample_image(imsize=model._input_shape[1:3])
            net_utl.save_activations_of_gtc_model(model, sample_image,
                                                  filename='F:/SRI/bitnet/mobilenet_ssd_300_eval_train_True_v3_TRAIN.hkl',
                                                  save_weights=True)
            model.save_model_def('F:/SRI/bitnet/mobilenet_ssd_300_eval_train_True_v2.json', sort_keys=True)
        '''

        use_locks = True and not utl.onLaptop()
        if use_locks:
            lock_name = os.path.basename(weights_file).replace('/', '_').replace('\\', '_').replace(':', '_')
            with lock.scoped_lock(lock_name) as test_lock:
                if not test_lock.islocked:
                    break

                if model is None:
                    model = gtc.GTCModel(model_file=model_file, weights_file=weights_file, verbose=False)

                results = evaluate_model_on_dataset(model, dataset, hp=hp_i,
                                                    verbose=verbose, save_results_filename=save_file, batch_size=16)

        else:
            if model is None:
                model = gtc.GTCModel(model_file=model_file, weights_file=weights_file, verbose=False)

            results = evaluate_model_on_dataset(model, dataset, hp=hp_i,
                                                verbose=verbose, save_results_filename=save_file, batch_size=16)

        utl.cprint('  --- %s mAP [%s]:  11-point sample: %.3f.  integrated: %.3f\n\n' % (
            prec, str(learning_phase), results['mAP_sample'], results['mAP_integrate']), color=utl.Fore.RED)



def test_ssd_folder(folder, hp=True, lp=True, save_results=False,
                    newest_only=False, verbose=False, redo=False):

    all_weights_files = glob.glob(folder + '/*.h5')
    print(' Found %d weight files in %s : ' % (len(all_weights_files), folder))

    n_newest_max = 3
    if len(all_weights_files) > n_newest_max and newest_only:
        # get newest first - those with highest modification time: so want reverse
        file_dates = [os.path.getmtime(f) for f in all_weights_files]
        idxs_descending = np.argsort(file_dates)[::-1]
        idxs_use = idxs_descending[:n_newest_max].tolist()
        all_weights_files = [w for i,w in enumerate(all_weights_files) if i in idxs_use]

    for i, weights_file in enumerate(all_weights_files):

        model_file_default = weights_file.replace('.h5', '.json')
        if os.path.exists(model_file_default):
            model_file_use = model_file_default
        else:
            all_model_files = glob.glob(folder + '/*.json')
            if len(all_model_files) == 0:
                raise ValueError('No .json model files in %s' % folder)
            if len(all_model_files) > 1:
                warnings.warn('Multiple .json model files in %s. Using first one.' % folder)
            model_file_use = all_model_files[0]


        utl.cprint(' Evaluating on model %d / %d : %s' % (
            i + 1, len(all_weights_files), weights_file), color=utl.Fore.GREEN)

        test_ssd_model(model_file_use, weights_file, hp=hp, lp=lp,
                           save_results=save_results, verbose=verbose, redo=redo)




def test_ssd_folder_tree(root_folder, hp=True, lp=True, save_results=False,
                         newest_only=False, verbose=False, redo=False):
    all_folders = []
    folders_count = 0
    test_folders_count = 0
    for root, dirs, files in os.walk(root_folder):
        #root + '/' + dir
        if 'tensorboard' in root or 'ignore' in root:
            continue
        folders_count += 1
        weight_files = glob.glob(root + '/*.h5')
        model_files =  glob.glob(root + '/*.json')
        if len(weight_files) > 0 and len(model_files) > 0:
            test_folders_count += 1
            all_folders.append(root)


    print(' Scanned %d folders under %s. %d have model & weight files ...' % (
        folders_count, root_folder, test_folders_count))

    for folder_idx, folder in enumerate(all_folders):
        utl.cprint(' Folder %d / %d : %s ' % (
            folder_idx+1, len(all_folders), folder), color=utl.Fore.CYAN)
        verbose_use = True if folder_idx == 0 else verbose
        test_ssd_folder(folder, hp=hp, lp=lp, save_results=save_results,
                        newest_only=newest_only, verbose=verbose_use, redo=redo)



if __name__ == "__main__":
    import glob
    from networks import net_utils as net_utl

    parser = argparse.ArgumentParser(description='Evaluation script')
    #parser.add_argument('--voc_dir_path', type=str,
    #                    help='VOCdevkit directory path')
    parser.add_argument('--dataset', type=str, default='pascal')
    parser.add_argument('--model_file', type=str,
                        help='weight file path', default=None)
    parser.add_argument('--weights_file', type=str,
                        help='weight file path', default=None)
    parser.add_argument('--training_dir', type=str,
                        help='training directory (containing one or more .h5 saved weights)', default=None)
    parser.add_argument('--recursive', type=str,
                        help='walk through all subdirectories of the training_dir to check for saved weights', default=None)

    parser.add_argument('--hp', type=str,
                        help='evaluate hp branch', default='1')
    parser.add_argument('--lp', type=str,
                        help='evaluate lp branch', default='0')

    parser.add_argument('--redo', type=str,
                        help='redo', default='0')
    parser.add_argument('--verbose', type=str,
                        help='verbose', default='0')
    parser.add_argument('--learning_phase', type=str,
                        help='verbose', default='0')
    parser.add_argument('--save_txt', type=str,
                        help='save results to text files', default='1')
    parser.add_argument('--newest_only', type=str,
                        help='verbose', default='1')
    parser.add_argument('--repeat_delay', type=float,
                        help='repeat after number of minuts', default=None)

    sess = tf.InteractiveSession()

    args = parser.parse_args()

    dataset = args.dataset
    model_file = args.model_file
    weights_file = args.weights_file
    training_dir = args.training_dir
    recursive = args.recursive == '1'
    do_hp = args.hp == '1'
    do_lp = args.lp == '1'
    redo = args.redo == '1'
    learning_phase = args.learning_phase == '1'
    save_txt = args.save_txt == '1'
    verbose = args.verbose == '1'
    newest_only = args.newest_only == '1'
    repeat_delay = args.repeat_delay


    if utl.onLaptop():
        vgg16_hp_only_dir = 'F:/models/_buckbeak/ssd_vgg16-keras/gtc-ssd_vgg16-keras__pascal-07-12-pxcf__tune-cls-2890__sgd_m9-lr0.001x80-0.0001x20-1e-05x20__1000b32/2020-03-09-17-20-09/'
        mobilenet_v2_300_hp_only_dir = 'F:/models/saved/gtc-ssd_mobilenet_v2-keras__pascal-07-12-pxcf__tune-cls__sgd_m9-lr0.001x80-0.0001x20-1e-05x20__1000b32/2020-03-10-16-07-26/'
        mobilenet_v2_224_hp_only_dir = 'F:/models/saved/gtc-ssd_mobilenet_v2-keras-224__pascal-07-12-pxcf__tune-cls__sgd_m9-lr0.001x80-0.0001x20-1e-05x20__1000b32/2020-03-10-16-07-18/'
        mobilenet_v2_224_gtc_dir = 'F:/models/saved/gtc-ssd_mobilenet_v2-keras-224__pascal-07-12-pxcf__tune-cls-7821__s1-GII__LamLp_e0__sgd_m9-lr0.002-drop0.1-ep160-40__1000b16/2020-03-27-04-49-57/'
        #test_dir = 'F:/models/ssd_mobilenet_v2-keras/a/'
        #training_dir = vgg16_hp_only_dir
        #training_dir = mobilenet_v2_300_hp_only_dir
        #training_dir = mobilenet_v2_224_hp_only_dir
        test_dir1 = 'F:/models/saved/gtc-ssd_mobilenet_v2-keras-224__pascal-07-12-pxcf__tune-cls__sgd_m9-lr0.001x80-0.0001x20-1e-05x20__1000b32/2020-03-10-16-07-18/test1_lin/'
        test_dir2 = 'F:/models/saved/deliverable/a/'

        #model_file = 'F:/SRI/bitnet/mobilenet_ssd_300_eval_train_True_v2.json'  # bn.share = ALL   -- 0.673
        #model_file = 'F:/SRI/bitnet/mobilenet_ssd_300_trainKerasModel.json'      # bn.share = NONE  -- 0.633
        #model_file = mobilenet_v2_300_hp_only_dir + 'ssd_mobilenet-v2-300_1.0.0.json'
        #weights_file = mobilenet_v2_300_hp_only_dir + 'ssd_mobilenet-v2-300_1.0.0.h5'

        #if False:
            #with open(model_file) as f:
            #    s = json.load(f)
            #with open

        do_hp = 1
        do_lp = 1
        redo = True
        verbose = True
        learning_phase = False

        #training_dir = 'F:/models/ssd_mobilenet_v2-keras/a/'
        #training_dir = mobilenet_v2_224_gtc_dir
        training_dir = test_dir2

        #training_dir = 'F:/models/ssd_vgg16-keras/'
        recursive = False
        newest_only = True


    keras.backend.set_learning_phase(learning_phase)
    print('LEARNING PHASE : %s' % str(learning_phase))


    if weights_file is not None:
        # Run test for a single weights file.
        # First, get the model file. Can use either model_file argument, or training_dir (if only one .json model file)
        print('Using model file %s' % model_file)
        if model_file is not None:
            model_file_use = model_file
        elif  training_dir is not None:
            print('Training dir : %s' % training_dir)
            if not os.path.exists(training_dir):
                raise ValueError('Training dir does not exist')
            all_model_files = glob.glob(training_dir + '/*.json')
            if len(all_model_files) == 0:
                raise ValueError('No .json model files in %s' % training_dir)
            if len(all_model_files) > 1:
                warnings.warn('Multiple .json model files in %s. Using first one.' % training_dir)

            model_file_use = all_model_files[0]
            print('Found model file %s in %s' % (model_file, training_dir))
        else:
            raise ValueError('Please define model_file or training_dir')

        test_ssd_model(model_file_use, weights_file, hp=do_hp, lp=do_lp, save_results=save_txt, redo=redo)

    elif training_dir is not None:

        while True:

            if recursive:
                test_ssd_folder_tree(training_dir, hp=do_hp, lp=do_lp, save_results=save_txt, redo=redo, newest_only=newest_only, verbose=verbose)

            else:
                test_ssd_folder(training_dir, hp=do_hp, lp=do_lp, save_results=save_txt, redo=redo, newest_only=newest_only, verbose=verbose)

            if repeat_delay is None:
                break
            else:
                print('Pausing for %d minutes, then scanning again...' % repeat_delay)
                time.sleep(repeat_delay * 60)  # sleep for repeat_delay minutes before starting again
                print('Scanning again...')

    else:
        raise ValueError('Specify either a model_file or a training_dir')


