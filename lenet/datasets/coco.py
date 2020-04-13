import os
import glob
import imageio
import cv2
import numpy as np
import hickle as hkl
import h5py
import json
import pickle as pkl
import zipfile

import ssd.ssd_info as ssd

from utility_scripts import misc_utl as utl
from datasets.dataset_loaders import DatasetLoader, DownloadDataset
from collections import OrderedDict
import ssd.ssd_utils as ssd_utl

coco_root = utl.datasets_root() + '/COCO/'


from datasets.ssd_data_loader import BatchGenerator

class GTCCocoLoader(BatchGenerator):

    def __init__(self, years=(2014,), split='train', imsize=None,
                 condensed_labels=True,
                 ssd_box_encoder=None, verbose=True):


        DownloadCocoDataset(verbose=False)
        if not isinstance(years, list):
            years=[years]

        dataset_types = ssd.SSDDatasetType(
            name='coco', years=years, split=split)

        datasets_info = ssd.get_datasets_info(dataset_types.spec_list())
        images_dirs = datasets_info['image_dirs']
        annotations_filenames = datasets_info['annot_files']


        #class_names = ssd.get_coco_class_names(include_bckg_class=True)



        if condensed_labels:
            # The dataset class indices are already in condensed format [1..80]
            class_idx_mapping = None

            class_names = ssd.get_coco_class_names(include_bckg_class=True, fill_missing=False)
        else:
            # We can convert the class indices [1..80] labels in dataset to 1..90 for tensorflow models
            classes_raw_tf = ssd.get_tf_coco_labels()
            class_idx_mapping = np.array( [0] + list(classes_raw_tf.keys()) )

            class_names = ssd.get_coco_class_names(include_bckg_class=True, fill_missing=True)

        super(GTCCocoLoader, self).__init__(
            imsize=imsize, split=split, ssd_box_encoder=ssd_box_encoder,
            class_names=class_names, class_idx_mapping=class_idx_mapping)


        self.parse_json(
            images_dirs=images_dirs,
            annotations_filenames=annotations_filenames,
            ground_truth_available=True,
            include_classes='all', ret=True, verbose=verbose)





class CocoLoader_old(DatasetLoader):
    def __init__(self, imsize=None, split='val', year=2014, batch_size=1,
                 max_num_images=None, max_num_images_per_class=None,
                 shuffle=False, root_dir=None, **kwargs):

        if root_dir is None:
            root_dir = coco_root
        if not root_dir.endswith('/'):
            root_dir += '/'
        self._coco_root_dir = root_dir

        valid_splits = ['train', 'val', 'test']
        if not split in valid_splits:
            raise ValueError('Invalid split : %s' % split)

        year_root_dir = root_dir + str(year) + '/'

        # F:/datasets/COCO/2014/val2014
        self._imdir = year_root_dir + split + str(year) + '/'
        if split in ['train', 'val']:
            annot_path = year_root_dir + 'annotations_trainval%d/annotations/' % year
            annot_file = 'instances_%s%d.json' % (split, year)
        else:
            raise NotImplementedError('Test loading not implemented yet')

        annot_file_json = annot_path + annot_file
        annot_file_pkl = annot_file_json.replace('.json', '.pkl')
        annot_file_hickle = annot_file_json.replace('.json', '.hkl')
        annot_file_h5 = annot_file_json.replace('.json', '.h5')

        if os.path.exists(annot_file_pkl):
            print("Loading %s ... " % annot_file_pkl)
            annot_data = pkl.load(open(annot_file_pkl, 'rb'))
            print("done")

        else:
            print("loading %s (size=%.1f MB)" % (annot_file_json, os.path.getsize(annot_file_json)/(1024**2) ))
            annot_data = json.load(open(annot_file_json))
            with open(annot_file_pkl, 'wb') as f:
                pkl.dump(annot_data, f)

        self.annot_data = annot_data

        self._imsize = imsize
        self._max_num_images = max_num_images
        self._max_num_images_per_class = max_num_images_per_class



        # self._train_synsets = [os.path.basename(s) for s in glob.glob(self._train_dir + '/n*')]
        # self._val_synsets = [os.path.basename(s) for s in glob.glob(self._val_dir + '/n*')]

        # self._train_syn_labelids = [self._synset_to_labelid[s] for s in self._train_synsets]
        # self._val_syn_labelids = [self._synset_to_labelid[s] for s in self._val_synsets]

        #base_dir = self._split_dir

        all_im_filenames = []
        all_filenames = glob.glob(self._imdir + '*.jpg')

        all_filebases = [ os.path.basename(f) for f in all_filenames]


        self._filebases = all_filebases

        self._cur_idx = 0
        self._batch_size = batch_size
        self._random_order = shuffle

        self._num_images_tot = len(self._filebases)
        self._num_images_use = self._num_images_tot

        if max_num_images is not None:
            self._num_images_use = min(max_num_images, self._num_images_tot)

        kwargs['num_classes'] = 90

        super(CocoLoader_old, self).__init__(
            num_images=self._num_images_use, batch_size=batch_size, shuffle=shuffle, imsize=imsize, **kwargs)


    def __len__(self):
        return self._num_images_use // self._batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self._cur_idx < self._num_images_use:
            return self.load()

        raise StopIteration

    def reset(self, shuffle=True):
        self._cur_idx = 0

        idx_order = list(range(self._num_images_tot))
        if self._random_order:
            idx_order = np.random.permutation(idx_order)
            idx_order = idx_order[:self._num_images_use]
        self._idx_order = idx_order

    #def get_label_names(self, label_ids):
    #    label_names = [self._label_to_name[id] for id in label_ids]
    #    return label_names

    def load_annotation(self, id):
        pass

    def load(self, batch_size=None, get_names=False):
        if batch_size is None:
            batch_size = self._batch_size

        idx_hi = min(self._cur_idx + batch_size, self._num_images_use)
        batch_idxs = list(range(self._cur_idx, idx_hi))
        im_idxs = [self._idx_order[i] for i in batch_idxs]

        all_im = []
        all_annot = []
        for i in im_idxs:
            im_file = self._imdir + self._filebases[i]
            im =  imageio.imread(im_file)
            if self._imsize is not None:
                im = cv2.resize(im, self._imsize)

            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

            all_im.append(im)
            annot = [] #self.load_annotation(i)
            #if not isinstance(annot, list):
            #    annot = [annot]

            all_annot.append( annot )

        if self._imsize is not None: # Concatenate all images into single array now
            all_im = np.asarray(all_im)


        self._cur_idx += batch_size

        #if get_names:
        #    all_names = [self._label_to_name[id] for id in all_labels]
        #    return all_im_merge, all_labels, all_names
        #else:
        #    return all_im_merge, all_labels
        return all_im, all_annot

    def load_all(self, get_names=False):

        self._cur_idx = 0
        return self.load(batch_size=self._num_images_use)



def CreateCocoMinivalSet(minival_name, ids_filename, redo=False, verbose=True):
    # Create COCO minival json file.
    coco_annot_subdir = coco_root + 'annotations/'
    val_file = coco_annot_subdir + 'instances_val2014.json'
    minival_file = coco_annot_subdir + 'instances_%s2014.json' % minival_name
    if os.path.exists(minival_file) and not redo:
        if verbose:
            print(' **  %s file already created' % minival_file)
        return

    print(' - Creating coco minival.json file : %s' % minival_file)

    minival_ids_file = coco_annot_subdir + ids_filename
    with open(minival_ids_file) as f:
        minival_ids = [int(s) for s in f.readlines()]

    with open(val_file) as f_val:
        S_val = json.load(f_val)
        if verbose:
            print(' - Loaded val.json file : %s' % val_file)

    S_minival = OrderedDict()
    # 1a,b: info, licenses
    S_minival['info'] = S_val['info'].copy()
    S_minival['licenses'] = S_val['licenses'].copy()

    # 2. Copy images:
    mini_images = []
    im_copy_count = 0
    for j,im_info in enumerate(S_val['images']):
        if im_info['id'] in minival_ids:
            mini_images.append(im_info.copy())
            im_copy_count += 1
    if verbose:
        print(' - Copied %d/%d images for minival set' % (
            im_copy_count, len(S_val['images'])))
    S_minival['images'] = mini_images

    S_minival['type'] = 'instances'

    # 3. Copy annotations:
    mini_annot = []
    annot_copy_count = 0
    for j,annot_info in enumerate(S_val['annotations']):
        if annot_info['image_id'] in minival_ids:
            mini_annot.append(annot_info.copy())
            annot_copy_count += 1
    if verbose:
        print(' - Copied %d/%d annotations for minival set' % (
            annot_copy_count, len(S_val['annotations'])))
    S_minival['annotations'] = mini_annot

    S_minival['categories'] = S_val['categories'].copy()

    #for id in minival_ids:
    #    S_minival = {}

    with open(minival_file, 'w') as f_mini:
        json.dump(S_minival, f_mini)

    if verbose:
        print(' - Finished creating minival file : %s' % minival_file)





def DownloadCocoDataset(verbose=True):

    delete_downloaded_tar_files = False

    #coco_subdir = '2014/'
    all_coco_archive_files = [
        # training images
        dict(url='http://images.cocodataset.org/zips/train2014.zip',
            sample_file='train2014/COCO_train2014_000000000009.jpg'),

        # validation images
        dict(url='http://images.cocodataset.org/zips/val2014.zip',
             sample_file='val2014/COCO_val2014_000000000042.jpg'),

        # training/validation annotations
        dict(url='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
             sample_file='annotations/instances_val2014.json'),


        # minival 5k dataset info  [This seems to be a different minival split, used for Mask-RCNN evals]
        # See https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md
        dict(url='https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip',
             sample_file='annotations/instances_minival2014.json',
             dest_subdir='annotations/'),

        # val 35k dataset info  [Used when training models with COCO 'trainval35k']
        # See https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md
        dict(url='https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip',
             sample_file='annotations/instances_valminusminival2014.json',
             dest_subdir='annotations/'),

        # Google tensorflow models use this list of minival ids (with 8k images) for evaluation
        # (We will use this list of ids, as well as the full instances_val2014.json file to create a minival8k json file
        dict(url='https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_minival_ids.txt',
             sample_file='annotations/mscoco_minival_ids.txt',
             dest_subdir='annotations/'),
    ]

    DownloadDataset(all_coco_archive_files, coco_root,
                    delete_downloaded_tar_files, verbose=verbose)


    CreateCocoMinivalSet('minival8k', 'mscoco_minival_ids.txt', redo=False, verbose=verbose)
    if utl.onLaptop():
        CreateCocoMinivalSet('microval64_', 'mscoco_microval64_ids.txt', redo=False, verbose=verbose)
        CreateCocoMinivalSet('microval256_', 'mscoco_microval256_ids.txt', redo=False, verbose=verbose)




if __name__ == "__main__":

    import ssd.post_processing_tf.visualization_utils as vis_util
    import networks.net_types as net
    DownloadCocoDataset(verbose=True)
    #D = CocoLoader_old('val', year=2014)
    #val_im = D.load(batch_size=12)
    # print('Loaded %s images/labels ' % (len(val_im)))

    net_type = net.MobilenetSSDType(mobilenet_version=2, style='tf')
    ssd_config = ssd_utl.SSDConfig(net_type=net_type, class_activation_in_network=True)

    create_labeled_examples = True
    if create_labeled_examples:
        show_encoded_box_labels = True
        show_matched_anchors = True
        if show_encoded_box_labels:
            skip_labels = True
            predictor_sizes = ssd_utl.predictor_sizes_mobilenet_300
            ssd_box_encoder = ssd_utl.SSDBoxEncoder(ssd_config, predictor_sizes)
            imsize = (300,300)
        else:
            ssd_box_encoder = None
            imsize = None
            skip_labels = False

        save_scale = 2.0
        scale_str = '_zoom%d' % save_scale if save_scale != 1 else ''
        split = 'train'
        n_batches_label = 10
        loader = GTCCocoLoader(years=2014, split=split, imsize=imsize,
                               ssd_box_encoder=ssd_box_encoder)
        if show_encoded_box_labels:
            loader.build(resize=imsize)

        batch_size = 1
        it = loader.flow(batch_size=batch_size, file_order=True,
                         returns=('filenames', 'original_images', 'processed_images',
                                  'original_labels',  'processed_labels', 'encoded_labels', 'matched_anchors'),
                         keep_images_without_gt=True)

        anchors_name = ''
        for batch_idx in range(1,n_batches_label):
            fn, x_raw, x, \
            labels_raw, labels_processed, y_true, y_anchors = it[batch_idx]

            # PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'

            if show_encoded_box_labels:
                if show_matched_anchors:
                    y_use = y_anchors
                    anchors_name = '_anchors'
                else:
                    y_use = y_true
                    anchors_name = '_gtboxes'
                #y_decoded = ssd_utl.decode_y(y_use,
                #                             iou_threshold=2,
                #                             top_k=1000, ssd_config=ssd_config, to_dict=True)
            else:
                y_use = labels

            for im_idx in range(batch_size):

                if len(labels_raw[im_idx]) == 0:
                    continue

                if show_encoded_box_labels:
                    fn_suffix = '_boxes'
                    #boxes = y_decoded[im_idx]
                    boxes = y_use[im_idx]

                else: # show ground truth labels
                    fn_suffix = ''
                    boxes = labels_raw[im_idx]

                dir_base = os.path.dirname(fn[im_idx])
                dir_save = os.path.dirname(fn[im_idx]) + '_labels/'
                fn_base = os.path.basename(fn[im_idx]).replace('.jpg', '') #fn_suffix + anchors_name + scale_str + '.png')


                # Visualize ground truth boxes:
                do_gt_boxes = False
                if do_gt_boxes:
                    fn_labels = dir_save + fn_base + '_labels.png'
                    ssd_utl.visualize_boxes_on_image(labels_raw[im_idx], x_raw[im_idx],  ssd_config=ssd_config, display_image=True,
                                                     save_filename=fn_labels, scale=save_scale,
                                                     line_thickness=1, max_boxes_to_draw=100, fontsize=15, )

                do_gt_anchor_boxes = True
                if do_gt_anchor_boxes:
                    fn_anchors = dir_save + fn_base + '_ytrue.png'
                    ssd_utl.visualize_boxes_on_image(y_true, x_raw[im_idx], ssd_config=ssd_config, display_image=True,
                                                     save_filename=fn_anchors, scale=save_scale, do_nms=False,
                                                     line_thickness=1, max_boxes_to_draw=100, fontsize=10, )

                do_anchor_boxes = True
                if do_anchor_boxes:
                    fn_anchors = dir_save + fn_base + '_anchors.png'
                    ssd_utl.visualize_boxes_on_image(y_anchors, x_raw[im_idx], ssd_config=ssd_config, display_image=True,
                                                     save_filename=fn_anchors, scale=save_scale, do_nms=False,
                                                     line_thickness=1, max_boxes_to_draw=100, fontsize=15, )


                a = 1

    im_ids = loader.image_ids
    filename = 'F:/datasets/COCO/minival_ids.txt'
    with open(filename, 'w') as f:
        f.writelines(['%d\n' % id for id in im_ids])

    fn = 'F:/datasets/COCO/mscoco_minival_ids.txt'
    with open(fn) as f2:
        xd = [int(xi) for xi in x]

    fn2 = 'F:/datasets/COCO/mscoco_minival_ids_sorted.txt'
    with open(fn2, 'w') as f3:
        f3.writelines(['%d\n' % y for y in sorted(xd)])
        x = f2.readlines().rstrip()


    it = loader.flow(batch_size=32, shuffle=False,
                         returns=('processed_images', 'processed_labels'),
                         keep_images_without_gt=True)
    #it = loader.flow(batch_size=32)

    x,y = it[0]



    a = 1

