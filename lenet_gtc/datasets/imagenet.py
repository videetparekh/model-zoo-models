import matplotlib.pyplot as plt

import os
import glob
import imageio
import cv2
import numpy as np
from utility_scripts import misc_utl as utl
import networks.net_utils_core as net
from datasets.dataset_loaders import DatasetLoader
from datasets.dataset_loaders import GTCBaseImageDataGenerator
import tensorflow.keras as keras
import platform
import warnings
from collections import OrderedDict

_imagenet_root = '/home/gtc-tensorflow/Data/' + 'imagenet/'      # utl.datasets_root() + 'imagenet/'

_imagenet_data_subdir = '/home/gtc-tensorflow/Data/CLS-LOC/'


class GTCImagnetLoader(GTCBaseImageDataGenerator):


    def load_dataset(self, split):
        featurewise_mean = np.array([123.675, 116.28 , 103.53])
        featurewise_std = np.array([58.395, 57.12 , 57.375])
        self.mean = featurewise_mean
        self.std  = featurewise_std
        # Filter out some warnings we get when loading the imagenet .jpeg files
        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
        warnings.filterwarnings("ignore", "Metadata Warning, tag \d+ had too many entries", UserWarning)

        self._directory = _imagenet_root + _imagenet_data_subdir + split + '/'

        # First time: load data
        synset_file = _imagenet_root + 'imagenet_lsvrc_2015_synsets.txt'
        with open(synset_file) as f:
            all_synsets = f.read().splitlines()

        synset_label_file = _imagenet_root + 'imagenet_metadata.txt'
        with open(synset_label_file) as f:
            all_synset_labels = f.readlines()

        self._synset_to_labelid = {}
        for i, synset in enumerate(all_synsets):
            self._synset_to_labelid[synset] = i

        self._synset_to_name = {}
        self._label_to_name = {}
        for s in all_synset_labels:
            # synset_to_human = {}
            parts = s.strip().split('\t')
            assert len(parts) == 2
            synset = parts[0]
            human = parts[1]
            self._synset_to_name[synset] = human

            if synset in self._synset_to_labelid:
                label_id = self._synset_to_labelid[synset]
                self._label_to_name[label_id] = human

        # overwrite flow --> point to flow_from_directory, so we can always
        # just call flow(), no matter which datasetloader we are using.
        self.flow = self.flow_from_directory

        self._dataset_loaded = True

        self._num_classes = 1000

        # For convenience, we hard-code the number of training samples in the
        # train and val splits of the dataset.
        if self._split == 'train':
            self._dataset_size = 1281167
        elif self._split == 'val':
            self._dataset_size = 50000




class ImagenetLoader(DatasetLoader):
    def __init__(self, imsize=(224,224), split='val', batch_size=16,
                 max_num_images=None, max_num_images_per_class=None,
                 shuffle=True, seed=None, dataset_root=None, display_filenames=False, **kwargs):

        if dataset_root is None:
            dataset_root = _imagenet_root

        dataset_data_subdir = dataset_root + _imagenet_data_subdir

        valid_splits = ['train', 'val', 'test']
        if not split in valid_splits:
            raise ValueError('Invalid split : %s' % split)

        self._split_dir = dataset_data_subdir + '/' + split + '/'
        self._imsize = imsize
        self._max_num_images = max_num_images
        self._max_num_images_per_class = max_num_images_per_class

        synset_file = dataset_root + 'imagenet_lsvrc_2015_synsets.txt'
        with open(synset_file) as f:
            all_synsets = f.read().splitlines()

        synset_label_file = dataset_root + 'imagenet_metadata.txt'
        with open(synset_label_file) as f:
            all_synset_labels = f.readlines()

        self._synset_to_labelid = {}
        for i, synset in enumerate(all_synsets):
            self._synset_to_labelid[synset] = i

        self._synset_to_name = {}
        self._label_to_name = {}
        for s in all_synset_labels:
            #synset_to_human = {}
            parts = s.strip().split('\t')
            assert len(parts) == 2
            synset = parts[0]
            human = parts[1]
            self._synset_to_name[synset] = human

            if synset in self._synset_to_labelid:
                label_id = self._synset_to_labelid[synset]
                self._label_to_name[label_id] = human


        # self._train_synsets = [os.path.basename(s) for s in glob.glob(self._train_dir + '/n*')]
        # self._val_synsets = [os.path.basename(s) for s in glob.glob(self._val_dir + '/n*')]

        # self._train_syn_labelids = [self._synset_to_labelid[s] for s in self._train_synsets]
        # self._val_syn_labelids = [self._synset_to_labelid[s] for s in self._val_synsets]

        base_dir = self._split_dir
        all_synsets = sorted([os.path.basename(s) for s in glob.glob(base_dir + '/n*')])
        self._synsets = all_synsets

        # Filter out warnings we often see when loading imagenet data files.
        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
        warnings.filterwarnings("ignore", "Metadata Warning, tag \d+ had too many entries", UserWarning)

        all_filenames = []
        all_labelids = []
        for i, syn in enumerate(all_synsets):
            dname = base_dir + '/' + syn + '/'
            labelid = self._synset_to_labelid[syn]

            filenames = sorted(glob.glob(dname + '*.JPEG'))

            if self._max_num_images_per_class is not None:
                filenames = filenames[:self._max_num_images_per_class]

            filebases = [ syn + '/' + os.path.basename(f) for f in filenames]
            all_filenames.extend(filebases)
            all_labelids.extend([labelid] * len(filebases))

            #all_filenames = all_filenames[:max_num_images]
            #all_labelids = all_labelids[:max_num_images]

        self._filenames = all_filenames
        self._labelids = all_labelids
        self._display_filenames = display_filenames

        idx_order = list(range(len(all_filenames)))
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            idx_order = np.random.permutation(idx_order)

        if max_num_images is not None and len(idx_order) > max_num_images:
            idx_order = idx_order[:max_num_images]
        num_images = len(idx_order)
        self._idx_order = idx_order

        kwargs['num_classes'] = 1000
        kwargs['classification'] = True

        super(ImagenetLoader, self).__init__(
            num_images=num_images, batch_size=batch_size,
            split=split, shuffle=shuffle,
            imsize=imsize, **kwargs)


    def label_to_name(self, id):
        return self._label_to_name[id]


    def load(self, index=None):

        if index is None:
            index = self._cur_idx

        idx = self._idx_order[index]

        im_filename = self._split_dir + '/' + self._filenames[ idx ]

        im = self.load_image(im_filename)
        y = self._labelids[ idx ]

        if self._display_filenames:
            #imsize = im.shape
            print('(%d / %d [%.2f %%]) Loading %s' % (
                self._cur_idx, self._num_images,
                self._cur_idx / self._num_images * 100.0, im_filename))

        self._cur_idx += 1
        return im, y



    def __load_old__(self, batch_size=None, get_names=False):

        if batch_size is None:
            batch_size = self._batch_size

        base_dir = self._split_dir
        batch_size = int(batch_size)
        idx_hi = min(self._cur_idx + batch_size, self._num_images)
        batch_idxs = list(range(self._cur_idx, idx_hi))
        im_idxs = [self._idx_order[i] for i in batch_idxs]

        all_im = []
        for i in im_idxs:
            im = self.load_idxs(base_dir + '/' + self._filenames[i])
            all_im.append(im)
        all_im_merge = np.asarray(all_im)


        all_labels = np.array([self._labelids[i] for i in im_idxs], dtype=np.int32)
        all_labels = all_labels[:, np.newaxis]

        self._cur_idx += batch_size

        if get_names:
            all_names = [self._label_to_name[id] for id in all_labels]
            return all_im_merge, all_labels, all_names
        else:
            return all_im_merge, all_labels





if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import xxhash
    split = 'val' if utl.onLaptop() else 'train'
    hasher = lambda x: xxhash.xxh64(x).hexdigest()
    test_new_loader = True
    if test_new_loader:

        #lambda_wgts = net.LambdaWeights(hp=1, lp=0, distillation_loss=0.01)
        lambda_wgts = net.LambdaWeights(hp=1)
        np.random.seed(1)
        new_loader = GTCImagnetLoader(split=split,
                                      featurewise_center=False,
                                      featurewise_std_normalization=False,
                                      horizontal_flip=False,
                                      vertical_flip=False,
                                      lambda_weights=lambda_wgts)

        batch_size = 1
        it = new_loader.flow(batch_size=batch_size, shuffle=False)
        n_tot = len(new_loader)//batch_size
        #x,y = next(it)
        hash_counts = OrderedDict()
        hash_images = OrderedDict()
        hash_indexes = OrderedDict()
        hash_labels = OrderedDict()
        all_hashes = []

        dup_count = 0
        im_idx = 0
        for batch_i, (x,y) in tqdm(enumerate(it), total=n_tot):
            if batch_i == len(new_loader):
                break
            y_labels_item = y[0]
            y_labels = np.argmax(y_labels_item, axis=1)
            for j in range(x.shape[0]):
                y_label = int(y_labels[j])
                if batch_i in [1768-1, 7189-1]:
                    aa = 1
                x_hash = hasher( x[j].tostring())

                all_hashes.append(x_hash)
                if x_hash in hash_counts:
                    dup_count+=1
                    hash_counts[x_hash] += 1
                    hash_indexes[x_hash].append(batch_i)
                    hash_labels[x_hash].append(y_label)
                    #hash_images[x_hash].append(x[j])
                    #joined_im = np.concatenate(hash_images[x_hash], axis=1).astype(np.uint8)
                    mult_labels = hash_labels[x_hash]
                    mult_label_names = [new_loader._label_to_name[k] for k in mult_labels]

                    print('batch %d : Discovered duplicate %d : images %s.  labels: %s.  label_names: (%s)' % (batch_i, dup_count, hash_indexes[x_hash],  hash_labels[x_hash], mult_label_names))
                    aa = 1
                else:
                    hash_counts[x_hash] = 1
                    hash_indexes[x_hash] = [batch_i]
                    hash_labels[x_hash] = [y_label]




        a =1
        duplicates= OrderedDict( [(k, v) for k,v in hash_indexes.items() if len(v) > 1])
        print('%d duplicates')
        print(duplicates)


        raise ValueError('done!')

    test_load_all_files = True
    if test_load_all_files:
        split = 'train'
        if platform.node() == 'ziskinda-7730':
            split = 'val'

        loader = ImagenetLoader(split=split, imsize=(224, 224), batch_size=16, shuffle=False, display_filenames=True, idx_start=0)
        for x,y in loader.__iter__():
            pass


    load_sample_images = False
    if load_sample_images:
        np.random.seed(1234)
        loader = ImagenetLoader(split='val', imsize=(224,224), batch_size=16, shuffle=True, display_filenames=True)
        #def __init__(self, imsize=(224,224), split='val', batch_size=16, max_num_images=None, max_num_images_per_class=None, shuffle=False, dataset_root=None):

        (val_im, val_label) = loader.load_batch(batch_index=0)
        label_names = loader.get_label_names(val_label)

        print('Loaded %s images/labels ' % (len(val_im)))
        for i in range(len(val_im)):
            plt.imshow(val_im[i])
            plt.title(label_names[i])
            plt.show()
            a = 1



    a = 1

