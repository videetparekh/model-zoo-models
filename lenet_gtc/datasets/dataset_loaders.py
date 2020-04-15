import os
import wget
import tarfile
from zipfile import ZipFile
import shutil
import tensorflow.keras as keras
import numpy as np
import imageio
import cv2
from datetime import datetime
import glob
from tensorflow.keras import backend




from networks import net_types as net
from keras.preprocessing.image import ImageDataGenerator

from utility_scripts import misc_utl as utl

import xxhash


#from datasets.dataset_loaders import DatasetLoader


def GetGTCDatasetLoader(dataset_name):
    from datasets.imagenet import GTCImagnetLoader
    from datasets.pascal import GTCPascalLoader
    from datasets.coco import GTCCocoLoader
    from datasets.keras_datasets import GTCMNistDatasetLoader, \
        GTCCifar10DatasetLoader, GTCCifar100DatasetLoader

    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == 'imagenet':
        return GTCImagnetLoader
    elif dataset_name_lower == 'mnist':
        return GTCMNistDatasetLoader
    elif dataset_name_lower == 'cifar10':
        return GTCCifar10DatasetLoader
    elif dataset_name_lower == 'cifar100':
        return GTCCifar100DatasetLoader

    elif dataset_name_lower == 'pascal':
        return GTCPascalLoader
    elif dataset_name_lower == 'coco':
        return GTCCocoLoader


    #elif dataset_name_lower == 'pascal':
    #    return PascalLoader
    #elif dataset_name_lower == 'coco':
    #    return CocoLoader

    else:
        raise ValueError('Unrecognized dataset name : %s' % dataset_name)


def get_dataset_loader(dataset_name):
    dataset_name_lower = dataset_name.lower()
    from datasets.imagenet import ImagenetLoader
    from datasets.keras_datasets import MNISTLoader, CIFAR10Loader, CIFAR100Loader
    from datasets.pascal import PascalLoader_old as PascalLoader
    from datasets.coco import  CocoLoader_old as CocoLoader

    if dataset_name_lower == 'imagenet':
        return ImagenetLoader
    elif dataset_name_lower == 'mnist':
        return MNISTLoader
    elif dataset_name_lower == 'cifar10':
        return CIFAR10Loader
    elif dataset_name_lower == 'cifar100':
        return CIFAR100Loader

    elif dataset_name_lower == 'pascal':
        return PascalLoader
    elif dataset_name_lower == 'coco':
        return CocoLoader

    else:
        raise ValueError('Unrecognized dataset name : %s' % dataset_name)


import platform
class DatasetInfo():
    def __init__(self, dataset_name):
        dataset_name = dataset_name.lower()

        if dataset_name == 'imagenet':
            imsize = (224, 224, 3)
            num_classes = 1000
            if platform.node() == 'ziskinda-7730':
                train_val_splits = ['val', 'val'] # on my laptop, i don't have the train set
            else:
                train_val_splits = ['train', 'val']

        elif dataset_name in ['cifar10', 'cifar100']:
            imsize = (32, 32, 3)
            num_classes = 10 if dataset_name == 'cifar10' else 100
            train_val_splits = ['train', 'test']

        elif dataset_name == 'mnist':
            imsize = (28, 28, 1)
            num_classes = 10
            train_val_splits = ['train', 'test']

        elif dataset_name == 'pascal':
            imsize = None
            num_classes = 20
            train_val_splits = ['trainval', 'test']
            #train_val_splits = ['train', 'val']

        elif dataset_name == 'coco':
            imsize = None
            num_classes = 90
            train_val_splits = ['train', 'minival']

        else:
            raise ValueError('Unrecognized dataset name: %s' % dataset_name)

        self.name = dataset_name
        self.imsize = imsize
        self.num_classes = num_classes
        self.train_split = train_val_splits[0]
        self.test_split = train_val_splits[1]












class GTCBaseImageDataGenerator():

    # Parent class for image data generators for classification datasets.
    # Based on the keras.preprocessing.image.ImageDataGenerator API, and actually
    # an instance of that class is created as one of the attributes of this class.
    # Differences between this class and that one:
    #   (1) Dataset augmentations parameters are not passed during initialization,
    #       but in a later method (build())
    #   (1) accepts lambda weights which augments the data labels with a copy for the
    #      lp branch, if necessary, and zeros for distillation/bit-loss
    #   (2) can convert to categorical for flow(...)
    #   (3) added the method load_dataset(dataset_name, split) for subclasses
    #       which loads the data/directory, so it doesn't need to be passed
    #       to the flow() method. In most cases, load_dataset is called automatically
    #   (4) for imagenet, overrides flow_from_directory() with flow()
    #       so you should always use the flow() method. Also hard-codes the
    #       channelwise mean/stds so you don't need to call .fit() for
    #       featurewise_center=True or featurewise_std_normalization=True
    #
    #   To use this class, instantiate one of the subclasses.
    #        loader = GTCCifar10DataGenerator(split='train')
    #
    #     model.fit( loader.flow(batch_size=32, ...), ...


    def __init__(self,
                 split,
                 ):

        # Pretty plain initialization class. The type of the parent class
        # defines which dataset. All that is left is to define which split
        # (train, val, test).
        self._split = split.lower()

        # These should be defined by child classes when load_dataset() is called.
        self._dataset_loaded = False
        self._use_directory = None
        self._directory = None
        #self._directory_iterator
        self._x = None
        self._y = None
        self._class_labels = None
        self._num_classes = None
        self._label_to_name = None
        self._x_train, self._y_train = None, None
        self._x_test, self._y_test = None, None
        self._dataset_size = None
        self.imsize = None

        self._augment_args = None
        self.load_dataset(split)

        # if not defined by child class, assume is False
        if not hasattr(self, 'need_to_fit_statistics'):
            self.need_to_fit_statistics = False

        assert self._dataset_size is not None, \
            "load_dataset should have defined dataset_size"

    @property
    def dataset_size(self):
        return self._dataset_size

    def build(self,
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
              dtype='float32',
              lambda_weights=None,
              ):

        if data_format is None:
            data_format = backend.image_data_format()
        if dtype is None:
            dtype = backend.floatx()

        augment_args = dict(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            validation_split=validation_split,
            interpolation_order=interpolation_order,
            dtype=dtype)

        if keras.__version__ == '2.2.5':
            augment_args.pop('interpolation_order', None)

        self._augment_args = augment_args
        self.imageDataGenerator = ImageDataGenerator(**augment_args)

        if self.need_to_fit_statistics:
            # This can only be run after we have called build with the
            # augmentation arguments, and we know whether to center/scale/etc.
            # the dataset.
            self.fit_statistics()

        self._lambda_weights = lambda_weights


    def fit_statistics(self):

        gen = self.imageDataGenerator
        if gen.featurewise_center \
                or gen.featurewise_std_normalization \
                or gen.zca_whitening:
            print('Running fit() to calculate dataset statistics')
            gen.fit(self._x)
            if gen.featurewise_center:
                print(' split=%s. mean = %s' % (self._split, str(gen.mean)) )
            if gen.featurewise_std_normalization:
                print(' split=%s. std  = %s' % (self._split, str(gen.std)) )
            if gen.zca_whitening:
                print(' split=%s. comp = %s' % (self._split, str(gen.principal_components)) )



    def get_name(self):
        if self._augment_args is None:
            raise ValueError('Please build before calling name')

        dataset_augment_type = net.KerasDatasetAugmentationsType(**self._augment_args)

        '''
        names = []
        if self._augment_args['featurewise_center']:
            names.append('ms')
        if self._augment_args['featurewise_std_normalization']:
            names.append('std')
        if self._augment_args['zca_whitening']:
            names.append('zca-%g' % self._augment_args['zca_whitening'])

        if self._augment_args['rotation_range'] != 0:
            names.append('rot%g' % self._augment_args['rotation_range'])

        if self._augment_args['width_shift_range'] != 0:
            names.append('ws%g' % self._augment_args['width_shift_range'])
        if self._augment_args['height_shift_range'] != 0:
            names.append('hs%g' % self._augment_args['height_shift_range'])
        if self._augment_args['brightness_range'] is not None:
            names.append('br%g' % self._augment_args['brightness_range'])
        if self._augment_args['channel_shift_range'] != 0:
            names.append('chsh%g' % self._augment_args['channel_shift_range'])



        if self._augment_args['shear_range'] != 0:
            names.append('sh%g' % self._augment_args['shear_range'])
        if self._augment_args['zoom_range'] != 0:
            names.append('zm%g' % self._augment_args['zoom_range'])

        if self._augment_args['horizontal_flip']:
            names.append('hFlp')
        if self._augment_args['vertical_flip']:
            names.append('vFlp')

        name = '-'.join(names)
        return name
        '''
        return str(dataset_augment_type)

    def __str__(self):
        return self.get_name()

        # unused so far:
        #samplewise_center = False,
        #samplewise_std_normalization = False,
        #channel_shift_range = 0.,
        #fill_mode = 'nearest',
        #cval = 0.,
        #rescale = None,
        #preprocessing_function = None,
        #data_format = 'channels_last',
        #validation_split = 0.0,
        #interpolation_order = 1,
        #dtype = 'float32',


    def load_dataset(self, split):
        # subclasses should define their own loaders
        raise NotImplementedError

    def __len__(self):
        if self._dataset_size is None:
            raise ValueError('Dataset size has not yet been defined')
        return self._dataset_size

    def flow(self,
             label_offset=0,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             save_stubs_only=False,
             subset=None,
             convert_to_categorical=True,
             max_num_samples=None,
             max_num_batches=None,
             max_num_classes=None, # ignored (included to match flow_directory)
             static=False):

        assert self._dataset_loaded, "Load dataset using load_dataset() method"

        assert self._x is not None and self._y is not None, \
            "x and y should be defined from a call to load_dataset()"

        assert max_num_samples is None or max_num_batches is None, \
            "Can't specify both max_num_samples and max_num_batches: Pick one "

        if max_num_samples is not None:
            max_num_batches = max_num_samples // batch_size

        if not shuffle:
            utl.cprint('**************** warning: shuffle = False *************************', color=utl.Fore.RED)

        print('Loaded %d images of size %s ... ' %
              (self._x.shape[0], str(self._x.shape[1:])) )

        if save_stubs_only:
            save_kwargs = {}
        else:
            save_kwargs = dict(save_to_dir=save_to_dir,
                               save_prefix=save_prefix,
                               save_format=save_format)

        data_iterator = self.imageDataGenerator.flow(
            self._x, self._y,
            batch_size=batch_size,shuffle=shuffle,
            sample_weight=sample_weight, seed=seed,
            subset=subset, **save_kwargs)

        stubs_save_to_dir = save_to_dir if save_to_dir is not None and save_stubs_only else None

        gtc_data_iterator = GTCAugmenter(
            data_iterator, self._lambda_weights, batch_size,
            stubs_save_to_dir=stubs_save_to_dir,
            max_num_batches=max_num_batches,
            convert_to_categorical=convert_to_categorical,
            num_classes=self._num_classes,
            label_offset=label_offset, static=static)

        #x0, y0 = data_iterator[0]
        #x1, y1 = gtc_data_iterator[0]

        return gtc_data_iterator


    def flow_from_directory(self,
                            label_offset=0,
                            max_num_samples=None,
                            max_num_batches=None,
                            max_num_classes=None,
                            target_size=(224, 224),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            save_stubs_only=False,
                            follow_links=False,
                            subset=None,
                            interpolation='nearest',
                            static=False,
                            max_samples_per_class=None,
                            loop=True,
                            convert_to_categorical=None,
                            ):

        assert self._directory is not None, \
            "directory should be defined from a call to load_dataset()"

        assert max_num_samples is None or max_num_batches is None, \
            "Can't specify both max_num_samples and max_num_batches: Pick one "

        if max_num_samples is not None:
            max_num_batches = max_num_samples // batch_size

        if convert_to_categorical is not None:
            if convert_to_categorical:
                class_mode = 'categorical'
            else:
                class_mode = 'sparse'

        # If class_mode == 'categorical', y is already a 1-hot vector:
        # we do not have to convert from sparse to categorical
        convert_to_categorical = class_mode != 'categorical'
        if max_num_classes is not None:
            # For debugging, it can save time to just read a couple of classes
            # instead of loading the iterator for, e.g. the entire imagenet dataset.
            # if we do this, though, the loader will think there are fewer classes.
            # To fix this, we have to manually convert to categorical with the true number of classes.
            class_mode = 'sparse'
            convert_to_categorical = True
            all_classes = os.listdir(self._directory)
            classes = all_classes[:max_num_classes]

        if save_stubs_only:
            save_kwargs = {}
        else:
            save_kwargs = dict(save_to_dir=save_to_dir,
                               save_prefix=save_prefix,
                               save_format=save_format)

        print('Scanning directory %s for images ... ' % self._directory)
        if max_samples_per_class is None:
            directory_iterator = self.imageDataGenerator.flow_from_directory(
                directory=self._directory,
                target_size=target_size,
                color_mode=color_mode,
                classes=classes,
                class_mode=class_mode,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                **save_kwargs,
                follow_links=follow_links,
                subset=subset,
                interpolation=interpolation)
        else:
            # use my own directory iterator, which can limit the number of
            # samples per class (but this can't do any data augmentation).s
            # useful for doing shortened evaluations on the validation split.
            directory_iterator = my_directory_iterator(
                directory=self._directory,
                target_size=target_size,
                color_mode=color_mode,
                classes=classes,
                max_num_classes=max_num_classes,
                class_mode=class_mode,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                max_samples_per_class=max_samples_per_class
            )


        # Wrap the output of the directory iterator with the gtc-augmenter
        # (which duplicates the labels for the lp branch of the networks),
        # and provides the zero targets for distillation loss & bit loss
        stubs_save_to_dir = save_to_dir if save_to_dir is not None and save_stubs_only else None

        gtc_data_iterator = GTCAugmenter(
            directory_iterator, self._lambda_weights, batch_size,
            stubs_save_to_dir=stubs_save_to_dir,
            max_num_batches=max_num_batches,
            convert_to_categorical=convert_to_categorical,
            num_classes=self._num_classes,
            label_offset=label_offset, static=static)

        return gtc_data_iterator




class GTCAugmenter(keras.utils.Sequence):
    # A wrapper class for the DatasetGenerator which augments the output
    # of the DatasetGenerator for the multiple outputs of the GTCKerasModel.
    # It inherits from the keras.utils.Sequence and thus is safe for multiple
    # workers with multiprocessing=True

    def __init__(self, dataset_iterator, lambda_weights, batch_size,
                 stubs_save_to_dir=None,
                 max_num_batches=None, loop=True,
                 convert_to_categorical=True,
                 num_classes = None, label_offset=0, static=False):

        self._dataset_iterator = dataset_iterator
        self._lambda_weights = lambda_weights
        self._batch_size = batch_size
        self._stubs_save_to_dir = stubs_save_to_dir
        self._max_num_batches = max_num_batches
        self._loop = loop
        self._convert_to_categorical = convert_to_categorical
        self._num_classes = num_classes
        self._label_offset = label_offset
        self._static = static
        self._batch_counter = 0


    def __len__(self):
        return len(self._dataset_iterator)

    def __getitem__(self, item):

        # create a new iterator, that iterates over the original dataset
        # but also duplicates the y labels for the lp-branch.


        # use_sequence_api = True

        #max_batch_id = np.floor( self._dataset_size / batch_size )
        #if max_num_batches is not None: #
        #    max_batch_id = min(max_batch_id, max_num_batches)
        n_batches_tot = len(self)
        if self._static: # debug mode: repeat same batch over and over
            item = 0

        x, y = self._dataset_iterator[item]
        self._batch_counter += 1

        #if count > min(self._dataset_size // batch_size, max_batch_id):

        if self._max_num_batches is not None and self._batch_counter >= min(self._max_num_batches, n_batches_tot):
            raise StopIteration
        if not self._loop and self._batch_counter >= n_batches_tot:
            raise StopIteration

        if self._stubs_save_to_dir is not None:
            # Saving stubs with filename equal to the hash of the image, followed
            # by the current time can be useful for debugging, to check/verify
            # that the same image is not being loaded multiple times per epoch.
            hasher = lambda x: xxhash.xxh64(x).hexdigest()
            for i in range(x.shape[0]):
                hash_str = hasher(x[i])
                y_label = y[i]
                if len(y_label) > 1:
                    y_label = np.argmax(y_label)
                else:
                    y_label = y_label[0]
                timestamp_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
                file_stub_name = self._stubs_save_to_dir + 'batch_%04d-%02d__%s__%02d--%s' % \
                                 (self._batch_counter, i, hash_str, y_label, timestamp_str)
                open(file_stub_name, 'w').close()

        if self._label_offset != 0:
            y += self._label_offset

        if self._convert_to_categorical:
            assert self._num_classes is not None
            num_classes_tot = self._num_classes + int(self._label_offset)
            y = keras.utils.to_categorical(y, num_classes_tot)

        if self._lambda_weights is not None:
            y = self._lambda_weights.AugmentOutput(y)

        return x, y





# Parent class for my older dataset loaders
class DatasetLoader(keras.utils.Sequence):

    """
    Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement
    `on_epoch_end`. The method `__getitem__` should return a complete batch.
    """

    def __init__(self, num_images, batch_size, shuffle, imsize,
                 split=None, **kwargs):

        self._num_images = num_images
        self._batch_size = batch_size
        self._num_batches = num_images // batch_size
        self._shuffle = shuffle
        self._imsize = imsize
        self._cur_idx = kwargs.pop('idx_start', 0)
        self._split = split


        self._classification = kwargs.pop('classification', False)

        self._convert_to_categorical = kwargs.pop('convert_to_categorical', False)
        self._num_classes = kwargs.pop('num_classes', None)
        self._label_offset = kwargs.pop('label_offset', 0)

        self._lambda_weights = kwargs.pop('lambda_weights', None)
        self._class_labels = kwargs.pop('class_labels', [])

        self._idx_order = np.arange(self._num_images)
        self._name = kwargs.pop('name', None)

        data_loader_log_prefix = kwargs.pop('data_loader_log_prefix', None)
        self._keep_log = data_loader_log_prefix is not None
        if self._keep_log:
            split_str = '' if self._split is None else '_' + self._split
            self._data_loader_log_file = data_loader_log_prefix + split_str + '.txt'
            with open(self._data_loader_log_file, 'w') as fid:
                pass

        self._batch_counter = 0
        if shuffle:
            self._idx_order = np.random.permutation(self._num_images)


        for k in kwargs.keys():
            print('Warning: ignoring option %s' % k)

        pass

    def __len__(self):
        #return self._num_images // self._batch_size

        num_batches = int( np.ceil(self._num_images / self._batch_size))
        return num_batches


    def __next__(self):
        if self._cur_idx < self._num_images:
            return self.load(self._cur_idx)

        raise StopIteration

    def reset(self):
        self._cur_idx = 0

        if self._shuffle:
            self._idx_order = np.random.permutation(self._num_images)


    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """

        return self.load_batch(batch_index=index)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    # Inherited properties
    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        #self.reset()
        pass


    def get_label_names(self, label_ids):

        if isinstance(label_ids, np.ndarray):
            label_ids = label_ids.flatten().tolist()

        is_list = isinstance(label_ids, (tuple, list))
        if not is_list:
            label_ids = [label_ids]

        label_names = [self.label_to_name(id) for id in label_ids]

        if not is_list:
            label_names = label_names[0]

        return label_names


    def label_to_name(self, id):
        raise NotImplementedError


    def load_batch(self, batch_index, batch_size=None):
        # if feeding in the non-default batch-size, make sure to feed the same
        # batch size throughout the entire epoch.
        if batch_size is None:
            batch_size = self._batch_size
        idx_start = batch_index * batch_size
        idx_end = min(idx_start + batch_size, self._num_images)
        idxs = list(range(idx_start, idx_end))

        self._batch_counter += 1
        if self._batch_counter % self._num_batches == 0:
            self.reset()


        if self._keep_log:
            date_str = datetime.now().strftime("%Y %m %d  %H %M %S")
            im_idxs = [str(self._idx_order[i]) for i in idxs]
            im_idxs_str = ' '.join(im_idxs)

            epoch_id = self._batch_counter // self._num_batches + 1
            log_str = date_str + '     %3d  %7d        %d %d           %s' % (
                epoch_id, self._batch_counter, idx_start, idx_end, im_idxs_str)
            with open(self._data_loader_log_file, 'a+') as fid:
                #if self._batch_counter % self._num_batches == 0:
                #    fid.write('======Epoch %d====\n' % (self._batch_counter // self._num_batches + 1))
                fid.write(log_str + '\n')

        return self.load_items(idxs)

    def load_image(self, im_file):

        im = imageio.imread(im_file)

        if self._imsize is not None:
            im = cv2.resize(im, self._imsize)

        if im.ndim == 2:     # replicate channels to RGB if only 1 channel
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

        if im.shape[2] == 4: # remove alpha channel if present
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)

        return im


    def load_items(self, idxs):
        # Working template for loading multiple items (image + labels) as a batch
        batch_x = []
        batch_y = []
        for idx in idxs:
            x,y = self.load(idx)
            batch_x.append(x)
            batch_y.append(y)

        #print('sizes: %s' % ','.join([str(x.shape) for x in batch_x]))
        batch_x = np.asarray(batch_x)

        if self._classification:
            batch_y = np.asarray(batch_y) + self._label_offset
            if self._convert_to_categorical:
                batch_y = keras.utils.to_categorical(batch_y, self._num_classes + self._label_offset)

            if self._lambda_weights is not None:
                batch_y = self._lambda_weights.AugmentOutput(batch_y)

        return batch_x, batch_y


    def load(self, index=None):
        raise NotImplementedError



    def load_all(self):

        self.reset()
        return self.load_batch(batch_index=0, batch_size=self._num_images)




# simple directory iterator, to support max-num-samples per class
class my_directory_iterator(keras.utils.Sequence):
    def __init__(
            self, directory,
            target_size=(224, 224),
            color_mode='rgb',
            classes=None,
            max_num_classes=None,
            class_mode='categorical',
            batch_size=32,
            shuffle=True,
            seed=None,
            file_ext='.JPEG',
            max_samples_per_class=None,
            label_offset=0):

        self._base_dir = directory

        sub_dirs = [name for name in os.listdir(self._base_dir)
                    if os.path.isdir(os.path.join(self._base_dir, name))]
        self._num_classes = len(sub_dirs)
        if classes is not None:
            sub_dirs_use = classes
        else:
            sub_dirs_use = sub_dirs

        if max_num_classes is not None:
            sub_dirs_use = sub_dirs_use[:max_num_classes]

        self._class_mode = class_mode
        self._color_mode = color_mode
        self._target_size = target_size
        self._label_offset = label_offset

        all_filenames = []
        all_labelids = []
        for labelid, syn in enumerate(sub_dirs_use):
            dname = self._base_dir + '/' + syn + '/'

            filenames = glob.glob(dname + '*' + file_ext)

            if max_samples_per_class is not None:
                filenames = filenames[:max_samples_per_class]

            filebases = [syn + '/' + os.path.basename(f) for f in filenames]
            all_filenames.extend(filebases)
            all_labelids.extend([labelid] * len(filebases))

            # all_filenames = all_filenames[:max_num_images]
            # all_labelids = all_labelids[:max_num_images]

        self._filenames = all_filenames
        self._labelids = all_labelids
        self._num_images = len(all_filenames)

        self._idx_order = np.arange(self._num_images)
        self._num_batches = int( np.ceil(self._num_images / batch_size)  )
        self._batch_size = batch_size
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            self._idx_order = np.random.permutation(self._num_images)

        self._shuffle = shuffle
        self._cur_batch_idx = 0

        super(my_directory_iterator, self).__init__()

    def __len__(self):
        # return self._num_images // self._batch_size
        return self._num_batches

    def __next__(self):
        if self._cur_batch_idx < self._num_batches:
            cur_batch = self.load_batch(self._cur_batch_idx)
            self._cur_batch_idx += 1
            return cur_batch

        raise StopIteration

    def reset(self):
        self._cur_batch_idx = 0

        if self._shuffle:
            self._idx_order = np.random.permutation(self._num_images)

    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """

        return self.load_batch(batch_index=index)

    def load_batch(self, batch_index):
        # if feeding in the non-default batch-size, make sure to feed the same
        # batch size throughout the entire epoch.

        idx_start = batch_index * self._batch_size
        idx_end = min(idx_start + self._batch_size, self._num_images)
        idxs = list(range(idx_start, idx_end))

        return self.load_items(idxs)

    def load_items(self, idxs):
        # Working template for loading multiple items (image + labels) as a batch
        batch_x = []
        batch_y = []
        for idx in idxs:
            x, y = self.load_sample(idx)
            batch_x.append(x)
            batch_y.append(y)

        # print('sizes: %s' % ','.join([str(x.shape) for x in batch_x]))
        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)

        if self._label_offset != 0:
            batch_y += self._label_offset

        if self._class_mode == 'categorical':
            batch_y = keras.utils.to_categorical(batch_y, self._num_classes + self._label_offset)

        return batch_x, batch_y


    def load_sample(self, index):

        idx = self._idx_order[index]

        im_filename = self._base_dir + '/' + self._filenames[idx]

        im = self.load_image(im_filename)
        y = self._labelids[idx]

        return im, y

    def load_image(self, im_file):

        im = imageio.imread(im_file)

        if self._target_size is not None:
            im = cv2.resize(im, self._target_size)

        if self._color_mode == 'rgb':

            if im.ndim == 2:     # replicate channels to RGB if only 1 channel
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

            if im.shape[2] == 4: # remove alpha channel if present
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)

        elif self._color_mode == 'grayscale':
            if im.ndim > 2 or im.shape[2] > 1:     # replicate channels to RGB if only 1 channel
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        elif self._color_mode == 'rgba':
            im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)

        else:
            raise ValueError('unknown color mode : %s' % self._color_mode)

        return im


    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item





def DownloadDataset(archive_files, dataset_root_dir, delete_after_download=False, verbose=True):

    if verbose:
        print('dataset_root_dir: %s' % dataset_root_dir)
        print(archive_files)

    if not os.path.exists(dataset_root_dir):
        os.makedirs(dataset_root_dir)

    for archive_idx, archive in enumerate(archive_files):
        file_url = archive['url']
        file_base = os.path.basename(file_url)
        dest_sub_dir = '/' + archive.get('dest_subdir', '')
        sample_file = dataset_root_dir + archive['sample_file']


        if os.path.exists(sample_file):
            if verbose:
                print('%d/%d : %s already downloaded and unzipped.' % (
                    archive_idx+1, len(archive_files), file_base))
            continue
        else:
            print('%d/%d : %s contents not present.' % (archive_idx+1, len(archive_files), file_base) )

        downloaded_file_now = False
        dst_file = dataset_root_dir + file_base

        if not os.path.exists(dst_file):
            print('    Archive file %s does not exist. Downloading now.' % dst_file)
            wget.download(file_url, dataset_root_dir)
            downloaded_file_now = True
        else:
            print('    Archive file %s already exists. No need to download ' % dst_file)

        if '.tar' in dst_file:
            print('    Unzipping tar file %s ... ' % dst_file)
            tfile = tarfile.open(dst_file)
            tfile.extractall(dataset_root_dir + dest_sub_dir)
        elif '.zip' in dst_file:
            print('    Unzipping zip file %s ... ' % dst_file)
            with ZipFile(dst_file, 'r') as zip:
                zip.extractall(dataset_root_dir + dest_sub_dir)
        elif '.txt' in dst_file: # Downloaded a text file : copy to dst_dir
            print('    Unzipping zip file %s ... ' % dst_file)
            shutil.copy(dst_file, dataset_root_dir + dest_sub_dir)
        else:
            raise ValueError('Unrecognized archive type')

        if delete_after_download and downloaded_file_now:
            print('Removing downloaded archive file %s' % dst_file)
            os.remove(dst_file)


if __name__ == "__main__":

    val_dir = 'F:/datasets/imagenet/ILSVRC/Data/CLS-LOC/val/'
    it = my_directory_iterator(val_dir, max_num_classes=10,
                               max_samples_per_class=5, batch_size=10,
                               shuffle=False,
                               class_mode='sparse')
    all_y = []
    for x,y in it:
        all_y.append(y)
        
    a = 1

