import tensorflow.compat.v1 as tf
from collections import namedtuple
from utility_scripts import misc_utl as utl
#from networks.net_types import SSDNetType


#SSDNetType = namedtuple('SSDNetType', ['mobilenet_version', 'ssd_lite', 'tf'])


#SSDDatasetType1 = namedtuple('SSDDatasetType', ['name', 'year', 'split'])

# For the COCO dataset, because the validation set is so large (~40k images), many authors
# add most of it to the training dataset, and leave the remaining images as the validation set.
# E.g in RCNN, SSD papers, they split the validation set into 35k + 5k. They add the
# 35k to the training set [train + val35k = trainval35k], and use the remaining 5k for
# actual validation experiments ['minival5k', or just minival, for short]
# For the tensorflow ssd_mobilenet checkpoint evaluations, they use a different subset of the
# validation set with 8k images for validation/evaluation: this is the 'minival8k' set]

valid_splits = ['trainval35k', 'trainval',  'train',
                'minival8k', 'minival5k', 'minival', 'val35k',  'val',
                'microval256', 'microval64',
                'test']

class SSDDatasetType():
    def __init__(self, name, years, split):
        self.name = name
        if isinstance(years, int):
            years = [years]
        self.years = years
        assert split in valid_splits, "%s is not a valid split" % split
        self.split = split

    def __str__(self):
        years_str = '+'.join([str(yr)[-2:] for yr in self.years])
        name_str = '_'.join([self.name, years_str, self.split])
        return name_str

    def spec_list(self):
        # Return a list of SSDDataTypes, one for each year
        return [SSDDatasetType(self.name, year, self.split) for year in self.years]

    @classmethod
    def from_str(cls, s):
        if 'pascal' in s.lower():
            name = 'pascal'
            available_years = [2007, 2012]
        elif 'coco' in s.lower():
            name = 'coco'
            available_years = [2014, 2015, 2017]
        else:
            raise ValueError('Unrecongnized SSD dataset')


        years = []
        for yr in available_years:
            if str(yr)[-2:] in s:
                years.append(yr)
        if len(years) == 0:
            raise ValueError('No valid years specified')

        #valid_splits = valid_splits
        split = None
        for split_i in valid_splits:
            if split_i in s:
                split = split_i
                break
        if split is None:
            raise ValueError('No split defined')

        return SSDDatasetType(name, years, split)




def get_datasets_info(dataset_specs):
    assert isinstance(dataset_specs, list)
    dset_infos = [get_dataset_info(dset) for dset in dataset_specs]

    dsets_info = dict(
        image_dirs=[d['image_dir'] for d in dset_infos],
        annot_dirs=[d['annot_dir'] for d in dset_infos],
        annot_files=[d['annot_file'] for d in dset_infos],
        id_files=[d['id_file'] for d in dset_infos]
    )
    return dsets_info


def get_dataset_info(dataset_spec):

    dset_info = {}
    dset_name = dataset_spec.name.lower()
    dset = dataset_spec

    if dset_name == 'pascal':
        root_dir = utl.datasets_root() + '/Pascal_VOC/VOCdevkit/'
        year_subdir = 'VOC%d/' % dset.years[0]
        dset_info['image_dir'] = root_dir + year_subdir + 'JPEGImages/'
        dset_info['annot_dir'] = root_dir + year_subdir + 'Annotations/'
        dset_info['annot_file'] = None
        dset_info['id_file'] = root_dir + year_subdir + \
                               'ImageSets/Main/%s.txt' % dset.split

    elif dset_name == 'coco':
        root_dir = utl.datasets_root() + '/COCO/'
        annot_fileprefix = 'image_info' if dset.split == 'test' else 'instances'
        if 'trainval' in dset.split:
            # for trainval, instantiate as two separate specs ('train', and 'val')
            raise ValueError('Invalid split: for trainval splits, use two different specs (e.g. "train", and "val"')

        dset_split_for_images = 'val' if 'val' in dset.split else 'train'
        split_dict = {'train':     'train',            # train set (82k)
                      'val':       'val',              # all 40k in validation set
                      'val35k':    'valminusminival',  # 35k / 40k
                      'minival':   'minival',          # remaining 5k / 40k
                      'minival5k': 'minival',          # remaining 5k / 40k (alias)
                      'minival8k': 'minival8k',        # separate subset of 8k/40k used for TF evaluation
                      'microval64': 'microval64_',     # my own tiny split with 64 images for testing
                      'microval256': 'microval256_'    # my own tiny split with 256 images for testing
                      }
        annotation_file_descrip = split_dict[dset.split]

        dset_year = dset.years[0]
        #im_subdir = 'train%d' if dset.split == 'train'
        dset_info['image_dir'] = root_dir + '%s%d/' % (
            dset_split_for_images, dset_year)
        dset_info['annot_dir'] = None
        dset_info['annot_file'] =root_dir + 'annotations/' + \
                                 annot_fileprefix + '_%s%d.json' % (
                                     annotation_file_descrip, dset_year)
        #'F:\datasets\COCO\annotations\instances_val2014.json'

        dset_info['id_file'] = None

    else:
        raise ValueError('unrecognized dataset name : %s' % dset.name)

    return dset_info

ssd_tf_ckpt = namedtuple('ssd_tf_ckpt', ['checkpoint', 'placeholder_name', 'imsize'])

all_mobilenet_ssd_tf_models = {
    'ssd_mobilenet_v1-tf': ssd_tf_ckpt(
        checkpoint='ssd_mobilenet_v1_coco_2017_11_17',
        placeholder_name='image_tensor:0', imsize=(300, 300)),

    'ssd_mobilenet_v2-tf': ssd_tf_ckpt(
        checkpoint='ssd_mobilenet_v2_coco_2018_03_29',
        placeholder_name='image_tensor:0', imsize=(300, 300)),

    'ssdlite_mobilenet_v2-tf': ssd_tf_ckpt(
        checkpoint='ssdlite_mobilenet_v2_coco_2018_05_09',
        placeholder_name='image_tensor:0', imsize=(300, 300)),

    'ssd_mobilenet_v3_large-tf': ssd_tf_ckpt(
        checkpoint='ssd_mobilenet_v3_large_coco_2019_08_14',
        placeholder_name='import/normalized_input_image_tensor:0', imsize=(320, 320)),

    'ssd_mobilenet_v3_small-tf': ssd_tf_ckpt(
        checkpoint='ssd_mobilenet_v3_small_coco_2019_08_14',
        placeholder_name='import/normalized_input_image_tensor:0' , imsize=(320, 320))
}




def get_ckpt_info(mobilenet_version):
    #if isinstance(mobilenet_version, SSDNetType):
    mobilenet_version = str(mobilenet_version)
    mobilenet_version = mobilenet_version.replace('-224', '')

    info = all_mobilenet_ssd_tf_models[mobilenet_version]
    return info


def get_tensorflow_ssd_model_ckpt_dir(net_type):
    #if isinstance(mobilenet_version, SSDNetType):

    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_ckpt_date_name = get_ckpt_info(net_type).checkpoint

    model_file = model_ckpt_date_name + '.tar.gz'
    model_dir_base = tf.keras.utils.get_file(
        fname=model_ckpt_date_name,
        origin=base_url + model_file,
        cache_subdir='models',
        untar=True)

    ckpt_dir = str(model_dir_base) + '/'
    return ckpt_dir


def get_tensorflow_model_ckpt(net_type):

    ckpt_dir = get_tensorflow_ssd_model_ckpt_dir(net_type)
    return ckpt_dir + 'model.ckpt'




#pascal_labels = ['background',
#               'aeroplane', 'bicycle', 'bird', 'boat',
#               'bottle', 'bus', 'car', 'cat',
#               'chair', 'cow', 'diningtable', 'dog',
#               'horse', 'motorbike', 'person', 'pottedplant',
#               'sheep', 'sofa', 'train', 'tvmonitor']

pascal_class_names = [
    'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
    'Chair', 'Cow', 'Dining table', 'Dog', 'Horse', 'Motorbike', 'Person',
    'Potted plant', 'Sheep', 'Sofa', 'Train', 'TV monitor']


def get_pascal_class_names(include_bckg_class=True):
    class_names = pascal_class_names.copy()
    if include_bckg_class:
        class_names.insert(0, '[Background]')

    return class_names


def get_coco_class_names(include_bckg_class=True, fill_missing=False):
    coco_label_idxs = list(range(1,91))
    if fill_missing:
        class_names = [coco_labels[i]['name'] if i in coco_labels
                   else '[%d-unavailable]'%i for i in coco_label_idxs]

    else:
        class_names = [coco_labels[i]['name'].title()
                       for i in coco_label_idxs if i in coco_labels]
    if include_bckg_class:
        class_names.insert(0, '[Background]')
    return class_names

def get_class_names(dataset_name, include_bckg_class):
    if dataset_name.lower() == 'pascal':
        return get_pascal_class_names(include_bckg_class)
    elif dataset_name.lower() == 'coco':
        return get_coco_class_names(include_bckg_class)
    else:
        raise ValueError('Invalid dataset name : %s' % dataset_name)

def get_tf_labels(dataset_name, **kwargs):
    if dataset_name == 'pascal':
        return get_tf_pascal_labels()
    elif dataset_name == 'coco':
        return get_tf_coco_labels(**kwargs)
    else:
        raise ValueError('Unrecognized dataset : %s' % dataset_name)

def get_tf_coco_labels(remove_missing_ids=False, add_missing_labels=False):
    existing_idxs = sorted(list(coco_labels.keys()))
    if remove_missing_ids:
        # re-number the indexes so they go from 1..80 instead of 1..90
        labels_condensed = {i+1:coco_labels[idx] for i,idx in enumerate(existing_idxs)}
        return labels_condensed
    elif add_missing_labels:
        # add placeholder labels for the 10 missing classes [12,26,29,30,45,66,68,69,71,83]
        existing_idxs = sorted(list(coco_labels.keys()))
        missing_label = lambda id: {'name': '<Background>' if id == 0 else '<Missing>' }
        labels_expanded = {idx:coco_labels[idx] if idx in coco_labels else missing_label(idx)
                           for idx in range(max(existing_idxs)+1) }

        return labels_expanded

    else:
        # Return the original list 80 indexes numbered 1 to 90, with a few missing labels
        return coco_labels.copy()


def get_tf_pascal_labels():
    class_names = pascal_class_names.copy()
    class_names.insert(0, '<Background>')

    class_labels = {i: {'name': nm} for i,nm in enumerate(class_names)}
    return class_labels



coco_labels = {
    # each entry is a dict with 'name':<name> format to be compatible with the tensorflow format
    # https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt
    1: {'name': 'person'},
    2: {'name': 'bicycle'},
    3: {'name': 'car'},
    4: {'name': 'motorcycle'},
    5: {'name': 'airplane'},
    6: {'name': 'bus'},
    7: {'name': 'train'},
    8: {'name': 'truck'},
    9: {'name': 'boat'},
    10: {'name': 'traffic light'},
    11: {'name': 'fire hydrant'},
    13: {'name': 'stop sign'},
    14: {'name': 'parking meter'},
    15: {'name': 'bench'},
    16: {'name': 'bird'},
    17: {'name': 'cat'},
    18: {'name': 'dog'},
    19: {'name': 'horse'},
    20: {'name': 'sheep'},
    21: {'name': 'cow'},
    22: {'name': 'elephant'},
    23: {'name': 'bear'},
    24: {'name': 'zebra'},
    25: {'name': 'giraffe'},
    27: {'name': 'backpack'},
    28: {'name': 'umbrella'},
    31: {'name': 'handbag'},
    32: {'name': 'tie'},
    33: {'name': 'suitcase'},
    34: {'name': 'frisbee'},
    35: {'name': 'skis'},
    36: {'name': 'snowboard'},
    37: {'name': 'sports ball'},
    38: {'name': 'kite'},
    39: {'name': 'baseball bat'},
    40: {'name': 'baseball glove'},
    41: {'name': 'skateboard'},
    42: {'name': 'surfboard'},
    43: {'name': 'tennis racket'},
    44: {'name': 'bottle'},
    46: {'name': 'wine glass'},
    47: {'name': 'cup'},
    48: {'name': 'fork'},
    49: {'name': 'knife'},
    50: {'name': 'spoon'},
    51: {'name': 'bowl'},
    52: {'name': 'banana'},
    53: {'name': 'apple'},
    54: {'name': 'sandwich'},
    55: {'name': 'orange'},
    56: {'name': 'broccoli'},
    57: {'name': 'carrot'},
    58: {'name': 'hot dog'},
    59: {'name': 'pizza'},
    60: {'name': 'donut'},
    61: {'name': 'cake'},
    62: {'name': 'chair'},
    63: {'name': 'couch'},
    64: {'name': 'potted plant'},
    65: {'name': 'bed'},
    67: {'name': 'dining table'},
    70: {'name': 'toilet'},
    72: {'name': 'tv'},
    73: {'name': 'laptop'},
    74: {'name': 'mouse'},
    75: {'name': 'remote'},
    76: {'name': 'keyboard'},
    77: {'name': 'cell phone'},
    78: {'name': 'microwave'},
    79: {'name': 'oven'},
    80: {'name': 'toaster'},
    81: {'name': 'sink'},
    82: {'name': 'refrigerator'},
    84: {'name': 'book'},
    85: {'name': 'clock'},
    86: {'name': 'vase'},
    87: {'name': 'scissors'},
    88: {'name': 'teddy bear'},
    89: {'name': 'hair drier'},
    90: {'name': 'toothbrush'}
}
