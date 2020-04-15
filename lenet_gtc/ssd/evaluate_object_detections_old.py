# sys.path.append(Path to repository)
import numpy as np
import cv2
import os
import argparse
import time
import ssd.ssd_utils as ssd_utl
#from ssd.ssd_eval_utils import decode_y
from datasets.ssd_data_loader import BatchGenerator
from tqdm import tqdm
import ssd.ssd_info as ssd
import imagesize
import imageio
import ssd.post_processing_tf.visualization_utils as vis_util
import datasets
from utility_scripts.sample_image_loader import SampleImageLoader
from utility_scripts import misc_utl as utl
import tensorflow.compat.v1 as tf
import keras
import shutil

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


#img_height = 300  # Height of the input images
#img_width = 300  # Width of the input images


def evaluate_network_on_dataset(model, dataset_specs, hp=True, verbose=True, **kwargs):

    tf.logging.set_verbosity(tf.logging.ERROR)

    if not isinstance(dataset_specs, list):
        dataset_specs = [dataset_specs]

    use_box_encoder = False
    if use_box_encoder:
        ssd_box_encoder = ssd_utl.SSDBoxEncoder(model.ssd_config,
                                                model.anchor_box_sizes)
    else:
        ssd_box_encoder = None

    years = [d.year for d in dataset_specs]
    splits = [d.split for d in dataset_specs]
    if dataset_specs[0].name == 'pascal':
        ssd_loader = datasets.GTCPascalLoader(years=years, split=splits[0],
                                              ssd_box_encoder=ssd_box_encoder, verbose=verbose)
    elif dataset_specs[0].name == 'coco':
        ssd_loader = datasets.GTCCocoLoader(years=years, split=splits[0],
                                            ssd_box_encoder=ssd_box_encoder, verbose=verbose)
    else:
        raise ValueError('Test on Pascal or COCO datasets only')

    #ssd_loader.build()
    #dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])

    iou_threshold = 0.5
    iou_threshold = kwargs.get('iou_threshold', iou_threshold)

    # The XML parser needs to now what object class names to look for and in which order to map them to integers.
    ssd_config = model.ssd_config

    filenames = ssd_loader.filenames
    all_labels = ssd_loader.labels
    #image_ids = ssd_loader.image_ids
    #ssd_loader.label_mapping_to_annot

    num_classes = ssd_config.n_classes + 1   # include background class
    dataset_size = ssd_loader.dataset_size
    #size = len(filenames)
    detected_labels = []

    ########################### Batch Size, number of samples
    batch_size = 32
    batch_size = kwargs.get('batch_size', batch_size)
    if verbose:
        print('Batch size: %d' % batch_size)

    max_num_batches = 16
    max_num_batches = None
    max_num_batches = kwargs.get('max_num_batches', max_num_batches)

    if max_num_batches is not None:
        if verbose:
            print('max_num_batches: %d' % max_num_batches)
        max_num_samples = batch_size * max_num_batches
        dataset_size = max_num_samples


    all_detections = [[None for _ in range(num_classes)] for j in range(dataset_size)]
    all_annotations = [[None for _ in range(num_classes)] for j in range(dataset_size)]
    all_ssd_durations = []
    all_ssd_post_durations = []

    class_ids = list(range(1, num_classes))



    all_decoded_predictions = []
    use_orig_im_coords = True
    img_width, img_height = ssd_config.img_width, ssd_config.img_height

    orig_2_net_size = lambda box, w,h: [int(box[0] * img_width / w),
                                        int(box[1] * img_height / h),
                                        int(box[2] * img_width / w),
                                        int(box[3] * img_height / h)]

    net_2_orig_size = lambda box, w,h: [int(box[0] * w / img_width),
                                        int(box[1] * h / img_height),
                                        int(box[2] * w / img_width),
                                        int(box[3] * h / img_height)]

    save_gt_path, save_dr_path, save_im_path = '', '', ''
    save_to_text_files_to_compare = False
    save_images_to_compare = False
    if save_to_text_files_to_compare:
        # To compare mAP calculation in this script with the calculation in this repo: https://github.com/Cartucho/mAP
        # we can save the detection/ground-truth to text files for easy use with that repo.
        # TL;DR:  mAP calculations from this script and that repo are very similar (within 1% of each other)

        save_base_dir = 'F:/SRI/bitnet/mAP/%s/' % str(model.type)
        save_gt_path = save_base_dir + '/input/ground-truth/'
        save_dr_path = save_base_dir + '/input/detection-results/'
        save_im_path = save_base_dir + '/input/images-optional/'
        if not os.path.exists(save_gt_path):
            os.makedirs(save_gt_path)
        if not os.path.exists(save_dr_path):
            os.makedirs(save_dr_path)
        if not os.path.exists(save_im_path) and save_images_to_compare:
            os.makedirs(save_im_path)


    if not use_orig_im_coords:
        all_labels_300 = []
        for i,gt_boxes in enumerate(all_labels):
            width_i, height_i = imagesize.get(filenames[i])
            gt_boxes_300 = [ box[:1] + orig_2_net_size(box[1:], width_i, height_i) for box in gt_boxes]
            all_labels_300.append(gt_boxes_300)

        all_labels = all_labels_300

    ssd_loader.build(img_width=img_width,img_height=img_height)
    it = ssd_loader.flow(batch_size=batch_size, file_order=True,
                         max_num_batches=max_num_batches,
                         returns=('processed_images', 'processed_labels', 'image_ids', 'filenames', 'original_images'),
                         keep_images_without_gt=True)


    if model.type.style == 'tensorflow':
        post_processor = ssd_utl.get_post_processor(ssd_config=ssd_config)
                                                    #label_mapping=label_mapping_to_annot)
    elif model.type.style_abbrev == 'keras':
        def post_processor(y_pred):
            y_pred_decoded = ssd_utl.decode_y(y_pred,
                                              ssd_config=ssd_config,
                                              to_dict=True)

            #y_pred_decoded = ssd_utl.decode_y_orig_repo(y_pred)
            return y_pred_decoded
    else:
        raise ValueError('Unrecognized network type')

    #num_batches = 50;  size = batch_size * num_batches
    batch_start = 0
    #batch_start = 1

    num_batches = int(np.ceil(dataset_size / batch_size))
    precision = 'hp' if hp else 'lp'
    if verbose:
        print('Running network on all images... (Testing on precision : %s)' % precision)

    class_labels_tf = ssd.get_tf_labels(dataset_specs[0].name)  # ssd.coco_labels

    #for i_start in tqdm( range(0, size, batch_size) ):
    for batch_idx, batch_data \
            in tqdm(enumerate(it, start=batch_start), total=num_batches, disable=not verbose):

        #images, labels, image_ids, y_true, batch_filenames, images_orig = batch_data
        images, labels, image_ids, batch_filenames, images_orig = batch_data

        #(images, labels, image_ids, y_true) = it[batch_idx]
        batch_size_i = len(labels)
        #input_images = np.array(image1)
        i_start = batch_idx*batch_size
        idxs = np.arange(i_start, min(i_start + batch_size_i, dataset_size) )

        orig_images = images
        input_images = np.array([cv2.resize(im, (img_height, img_width)) for im in orig_images])

        #print('Running batch %d size= %s ' % (batch_idx, str(input_images.shape)))


        hp_kwargs = dict(hp=hp)
        if hp or isinstance(model, keras.Model):
            hp_kwargs = {}

        start_time = time.time()
        y_pred = model.predict(input_images, **hp_kwargs)
        t_elapsed = time.time() - start_time
        all_ssd_durations.append(t_elapsed)

        if isinstance(model, keras.Model) and isinstance(y_pred, list):
            # GTCKerasModel that predicts hp and lp in a list.
            if hp:
                y_pred = y_pred[0]
            else:
                y_pred = y_pred[1]



        #print('post processor for batch %d ' % batch_idx)
        start_time = time.time()
        y_pred_decoded_array = ssd_utl.decode_y(y_pred,
                                          ssd_config=ssd_config,
                                          to_dict=False)
        y_pred_decoded = post_processor(y_pred)
        t_elapsed2 = time.time() - start_time
        #print('post processor for batch %d completed' % batch_idx)
        all_ssd_post_durations.append(t_elapsed2)
        all_decoded_predictions.extend(y_pred_decoded)


        do_chk = False
        if do_chk:
            loader = SampleImageLoader(imsize=ssd_config.imsize[:2], image_path='F:/SRI/bitnet/gtc-tensorflow/test_code/dogs.jpg')
            img_dogs, _ = loader.load()

            y_pred_dogs = model.predict(img_dogs)

            ncls = ssd_config.n_classes + 1
            y_true_cls = y_true[0, :, :ncls]
            y_pred_cls = y_pred[0, :, :ncls]
            pred_cls = np.argmax(y_pred_cls,axis=1)
            idx_pred_cls = pred_cls.nonzero()[0]
            idxs_pos = (np.argmax(y_true_cls, axis=1) > 0).nonzero()[0]
            cls_true = np.argmax(y_true_cls[idxs_pos], axis=1)
            cls_pred = np.argmax(y_pred_cls[idxs_pos], axis=1)


        for b_im_idx, glob_idx in enumerate(idxs):

            pred_boxes = []
            pred_labels = []

            visualize_detections=False
            if visualize_detections:
                b_im_idx_show = 3
                glob_idx = idxs[b_im_idx_show]
                output_dict = y_pred_decoded[b_im_idx_show] # mobilenet_ssd.post_processor(y_pred, sess=sess, config=SSDConfig, feed_dict=data_dict)

                im_orig_size0 = orig_images[b_im_idx_show].copy() # make copy for hp and for lp
                y_pred_all_raw = ssd_utl.decode_y(y_pred, ssd_config=ssd_config, do_nms=False, to_dict=True)
                y_pred_all_nms = ssd_utl.decode_y(y_pred, ssd_config=ssd_config, do_nms=True, to_dict=True)
                # PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
                category_index = ssd.get_tf_labels(dataset_specs[0].name) # ssd.coco_labels
                show_pred_labels = False
                show_gt_labels = True
                if show_pred_labels:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        im_orig_size0,
                        output_dict['boxes_norm'], #[:,[1,0,3,2]],
                        output_dict['classes'],
                        output_dict['scores'],
                        category_index,
                        instance_masks=None,
                        use_normalized_coordinates=True,
                        line_thickness=2)

                if show_gt_labels:
                    gt_dict = dict(boxes=np.array([L[1:5] for L in labels[b_im_idx]]),
                                   classes=[L[0] for L in labels[b_im_idx]],
                                   scores=np.ones(len(labels[b_im_idx]) ))
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        im_orig_size0,
                        gt_dict['boxes'], #[:,[1,0,3,2]],
                        gt_dict['classes'],
                        gt_dict['scores'],
                        category_index,
                        instance_masks=None,
                        use_normalized_coordinates=False,
                        line_thickness=2)

                display_image = True  # sometimes crashes when run from inside docker.
                if display_image:
                    cv2.imshow("image %d [Close window to see next image]" % (b_im_idx_show+1), im_orig_size0[..., ::-1])
                    cv2.waitKey(0)

                save_image = False
                if save_image:
                    precision = 'hp'

                    filename = 'output_%s_ssd_tf_im%d__%s.png' % (str(model.type), glob_idx+1, filenames[glob_idx])
                    cv2.imwrite(filename, im_orig_size0[..., ::-1])
                    precision_str = 'high' if precision=='hp' else ' low'
                    print('Saved %s-precision SSD output to %s' % (precision_str, filename))
                    #time.sleep(2)

            # 1. Add the predicted labels to the 'all_predictions' list
            num_detections = y_pred_decoded[b_im_idx]['num_detections']
            for box_i in range(num_detections):

                box_class_id = y_pred_decoded[b_im_idx]['classes'][box_i]
                box_score = y_pred_decoded[b_im_idx]['scores'][box_i]
                box_coords = y_pred_decoded[b_im_idx]['boxes'][box_i]

                if use_orig_im_coords:
                    h,w = orig_images[b_im_idx].shape[:2]
                    box_coords = net_2_orig_size(box_coords, w, h)
                else:
                    box_coords = box_coords.astype(np.int).tolist()

                pred_boxes.append( box_coords + [box_score])
                pred_labels.append(box_class_id)

            pred_boxes = np.array(pred_boxes)
            pred_labels = np.array(pred_labels)

            if len(pred_labels) > 0:
                for class_id in class_ids:
                    all_detections[glob_idx][class_id] = pred_boxes[pred_labels == class_id, :]


            # 2. Add the ground truth labels to the 'all_annotations' list
            true_label = np.array(labels[b_im_idx])

            for class_id in class_ids:
                if len(true_label) > 0:
                    all_annotations[glob_idx][class_id] = true_label[true_label[:, 0] == class_id, 1:5].copy()
                else:
                    all_annotations[glob_idx][class_id] = np.zeros((0,4))

            if save_to_text_files_to_compare:
                image_id = image_ids[b_im_idx]
                save_annotation(all_annotations[glob_idx], save_gt_path + image_id + '.txt', class_labels_tf)
                save_annotation(all_detections[glob_idx], save_dr_path + image_id + '.txt', class_labels_tf)
                if save_images_to_compare:
                    dst_file = save_im_path + os.path.basename(batch_filenames[b_im_idx])
                    if not os.path.exists(dst_file):
                        shutil.copy(batch_filenames[b_im_idx], save_im_path)
                    #imageio.imwrite(save_im_path + image_id + '.jpg', )




    def calculate_mAP_for_annotations_and_detections(annotations_in, detections_in, verbose=True):
        average_precisions = {}
        all_lbl_times = []

        num_annotations_each_class = np.zeros(num_classes)

        if verbose:
            print('Evaluating Accuracy across all %d categories ...' % len(class_ids))
        for class_id in tqdm(class_ids, disable=not verbose):
            start_time = time.time()

            false_positives = [] # np.zeros((0,))
            true_positives = [] # np.zeros((0,))
            scores = [] # np.zeros((0,))
            num_annotations = 0

            #annotations_this_class = [ann[class_id] for ann in all_annotations]
            #num_annotations0 = np.sum([ann.shape[0] for ann in annotations_this_class])
            dataset_size_i = len(annotations_in)
            for i in range(dataset_size_i):
                annotations = annotations_in[i][class_id]
                annotations = annotations.astype(np.float32)
                if annotations.size == 0:
                    assert annotations.shape[0] == 0
                num_annotations += annotations.shape[0]
                detected_annotations = []
                detections = detections_in[i][class_id]
                if detections is not None:
                    detections = detections.astype(np.float32)

                    for d in detections:
                        scores.append(d[4])

                        try:
                            annotations[0][0]
                        except IndexError:
                            false_positives.append(1)
                            true_positives.append(0)
                            continue

                        overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                        assigned_annotation = np.argmax(overlaps, axis=1)
                        max_overlap = overlaps[0, assigned_annotation]

                        if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:

                            false_positives.append(0)
                            true_positives.append(1)
                            detected_annotations.append(assigned_annotation)
                        else:
                            false_positives.append(1)
                            true_positives.append(0)

            num_annotations_each_class[class_id] = num_annotations

            if num_annotations == 0:
                average_precisions[class_id] = 0
            else:
                indices = np.argsort(-np.array(scores))
                false_positives = np.array(false_positives)[indices]
                true_positives = np.array(true_positives)[indices]

                false_positives = np.cumsum(false_positives)
                true_positives = np.cumsum(true_positives)

                recall = true_positives / num_annotations

                precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

                average_precision = compute_ap(recall, precision)
                average_precisions[class_id] = average_precision

            t_elapsed = time.time() - start_time
            all_lbl_times.append(t_elapsed)

        if verbose:
            print('Total time: ', sum(all_lbl_times))

        count = 0
        for k in average_precisions.keys():
            count = count + float(average_precisions[k])

        # For the COCO dataset, only 80 out of the 90 classes have labels
        num_classes_with_data = int(np.sum(num_annotations_each_class > 0))
        if verbose:
            print('%d/%d classes have data.' % (num_classes_with_data, len(average_precisions.keys())))

        map = count / num_classes_with_data
        if verbose:
            print(average_precisions)
            print('MAP is :', map)

        return map, average_precisions, num_annotations_each_class

    t_start = time.time()
    map, average_precisions, num_each_class = calculate_mAP_for_annotations_and_detections(
        all_annotations, all_detections, verbose=verbose)
    t_elapsed_mAP = time.time() - t_start


    test_on_mini_batches = False
    if test_on_mini_batches:

        all_cls = list(range(1, num_classes))
        cum_ap = {i:0.0 for i in all_cls}
        cum_wgt_ap = {i:0.0 for i in all_cls}
        cum_n_each_class = {i: 0 for i in all_cls}

        batch_maps, batch_ap, batch_n_per_class = [], [], []
        #cum_maps, cum_ap_l, cum_n_per_class = [], [], []
        n_batches = dataset_size // batch_size
        for i in range(n_batches):
            slc_i = slice(i*batch_size, (i+1)*batch_size)
            map_i, average_precisions_i, num_each_class_i = \
                calculate_mAP_for_annotations_and_detections(
                    all_annotations[slc_i], all_detections[slc_i], verbose=False)
            for j in all_cls:
                cum_ap[j] += average_precisions_i[j]
                cum_wgt_ap[j] += average_precisions_i[j] * num_each_class_i[j]
                cum_n_each_class[j] += num_each_class_i[j]

            batch_maps.append(map_i)
            batch_ap.append(average_precisions_i)
            batch_n_per_class.append(num_each_class_i)

            mean_mAP_i = float( np.mean(batch_maps) )

            mean_cum_ap = float(np.mean( list(cum_ap.values()) ) )
            mean_cum_wgt_ap = float( np.sum(list(cum_wgt_ap.values())) / np.sum(list(cum_n_each_class.values())) )
            # naive cumulative average:
            if verbose:
                print('%i. mean mAP: %.4f. mean_cum_ap : %.4f.   mean_cum_wgt_ap : %.4f' % (
                    i, mean_mAP_i, mean_cum_ap, mean_cum_wgt_ap ) )

            a = 1

    if verbose:
        print('Completed evaluation of network %s on these dataset(s) ' % (
            str(model.type)))
        print(dataset_specs)

        durations_sec = np.array(all_ssd_durations)
        mean_duration_ms = durations_sec.mean()*1000
        print('Average speed (ms per detection) : %.3f' % mean_duration_ms)

        post_durations_sec = np.array(all_ssd_post_durations)
        mean_post_duration_ms = post_durations_sec.mean()*1000
        print('Average speed (ms per post-processing) : %.3f' % mean_post_duration_ms)

        print('Average forward time per batch: : %.3f sec' % (durations_sec.sum() / num_batches))
        print('Average post_processing time per batch:  %.3f sec' % (post_durations_sec.sum() / num_batches))
        print('Average time for mAP per batch : %.3f sec' % (t_elapsed_mAP / num_batches))

        print('Preprocessing step : ', [s for s in model._layer_names if 'Preprocess' in s])
        print('ssd_config.steps : ', ssd_config.steps)
        print('confidence threshold ', ssd_config.score_thresh)
        print('max_size_per_class ', ssd_config.max_size_per_class)
        print('max_total_size ', ssd_config.max_total_size)

    #else:
    #    print('Calculated mAP = %.3f' % map)

    return map


def save_annotation(annotations, save_file, all_labels):
    with open(save_file, 'w') as f:
        for cls_i, annot_i in enumerate(annotations):
            if annot_i is None:
                #assert cls_i == 0
                continue
            for box_i in annot_i:
                label_i = all_labels[cls_i]
                if isinstance(label_i, dict):
                    label_i = label_i['name']
                label_i = label_i.lower().replace(' ', '')
                box_str = '%d %d %d %d' % tuple(box_i[:4])
                if len(box_i) == 4:
                    f.write('%s %s\n' % (label_i, box_str))
                elif len(box_i) == 5:
                    f.write('%s %.6f %s\n' % (label_i, box_i[-1], box_str))
                aa = 1


class calculate_mAP_callback(keras.callbacks.Callback):
    """Callback to calculate mAP at the end of each training epoch,
    and record the results in the logs"""

    def __init__(self, dataset_specs, epoch_freq=1,
                 hp=True, lp=True, color=None, **mAP_kwargs):
        self.dataset_specs = dataset_specs
        self.epoch_freq = epoch_freq
        self.hp = hp
        self.lp = lp
        self.dataset_split = dataset_specs[0].split
        self.dataset_str = ','.join([str(spec) for spec in self.dataset_specs])
        if color is None:
            color = utl.Fore.RED
        self.color = color
        self.print = lambda s : utl.cprint(s, color=self.color)

        self.mAP_kwargs = mAP_kwargs

        super(calculate_mAP_callback, self).__init__()


    def on_epoch_end(self, epoch, logs=None):

        # always do on first epoch (epoch==0), so it creates a header in the log file
        calc_now = epoch == 0 or (epoch % self.epoch_freq == 0)
        if not calc_now:
            return

        map_strs = []
        self.print('Calculating mAP ...')
        if self.hp:
            t1 = time.time()
            mAP_hp = evaluate_network_on_dataset(
                self.model, self.dataset_specs, hp=True, **self.mAP_kwargs)
            hp_key = 'mAP_%s_hp' % self.dataset_split
            logs[hp_key] = mAP_hp
            t_hp = time.time() - t1
            map_strs.append('HP: %.3f  [%.1f sec]' % (mAP_hp, t_hp) )

        if self.lp:
            t2 = time.time()
            mAP_lp = evaluate_network_on_dataset(
                self.model, self.dataset_specs, hp=False, **self.mAP_kwargs)
            lp_key = 'mAP_%s_lp' % self.dataset_split
            logs[lp_key] = mAP_lp
            t_lp = time.time() - t2
            map_strs.append('LP: %.3f  [%.1f sec]' % (mAP_lp, t_lp))

        map_str = ',  '.join(map_strs)
        self.print('mAP on %s : %s' % (self.dataset_str, map_str ))


    def __str__(self):
        precisions_test = [pr for pr,tf in zip(['HP', 'LP'], [self.hp, self.lp]) if tf]
        hp_lp_str = ' and '.join(precisions_test)
        s = 'Calculate_mAP_callback on %s [%s] Every %d epochs' % (
            self.dataset_str, hp_lp_str, self.epoch_freq)
        return s


def get_arg(arg_value, value_if_none):
    if arg_value is not None:
        if isinstance(value_if_none, bool):
            arg_value = bool(arg_value)
        return arg_value
    else:
        return value_if_none



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--voc_dir_path', type=str,
                        help='VOCdevkit directory path')
    parser.add_argument('--weight_file', type=str,
                        help='weight file path')

    #args = parser.parse_args()
    #evaluate_network_on_dataset(None, args)
