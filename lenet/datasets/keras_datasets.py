import matplotlib.pyplot as plt

import os
import glob
import imageio
import cv2
import numpy as np
from utility_scripts import misc_utl as utl
from datasets.dataset_loaders import DatasetLoader, GTCBaseImageDataGenerator

import keras.datasets

# keras datasets include:
# mnist, imdb, reuters, cifar10, cifar100, boston_housing, fashion_mnist

# Currently implemented below: MNIST, CIFAR10, CIFAR100

default_split='train'
default_batch_size=16
default_shuffle=True
default_imsize=None


cifar10_class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck']


cifar100_coarse_class_labels = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers',
    'fruit_and_vegetables', 'household_electrical_devices',
    'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores', 'medium_mammals',
    'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
    'trees', 'vehicles_1', 'vehicles_2']

cifar100_fine_class_labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


class GTCKerasDatasetLoader(GTCBaseImageDataGenerator):

    def __init__(self, split):

        self.need_to_fit_statistics = True # After dataset augmentations
                                           # have been defined, run fit()

        super(GTCKerasDatasetLoader, self).__init__(split=split)


    def load_dataset_finalize(self, split):
        self._split = split.lower()

        if self._split == 'train':
            self._x = self._x_train
            self._y = self._y_train
        elif self._split in ['val', 'test']:
            self._x = self._x_test
            self._y = self._y_test
        else:
            raise ValueError('invalid split : %s' % self._split)

        if self._x.ndim < 4:
            self._x = np.expand_dims(self._x, axis=-1)

        self.imsize = self._x.shape[1:]

        self._num_classes = len(self._class_labels)

        self._dataset_loaded = True

        self._dataset_size = self._x.shape[0]



class GTCCifar10DatasetLoader(GTCKerasDatasetLoader):

    #def __init__(self, split):
    #    super(GTCCifar10DatasetLoader, self).__init__(split=split)

    def load_dataset(self, split):
        (self._x_train, self._y_train), (self._x_test, self._y_test) = \
            keras.datasets.cifar10.load_data()
        self._class_labels = cifar10_class_labels

        super(GTCCifar10DatasetLoader, self).load_dataset_finalize(split)


class GTCCifar100DatasetLoader(GTCKerasDatasetLoader):

    def load_dataset(self, split):
        (self._x_train, self._y_train), (self._x_test, self._y_test) = \
            keras.datasets.cifar100.load_data()
        self._class_labels = cifar100_fine_class_labels

        super(GTCCifar100DatasetLoader, self).load_dataset_finalize(split)



class GTCMNistDatasetLoader(GTCKerasDatasetLoader):

    def load_dataset(self, split):
        (self._x_train, self._y_train), (self._x_test, self._y_test) = \
            keras.datasets.mnist.load_data()
        self._class_labels = [str(x) for x in range(10)]

        super(GTCMNistDatasetLoader, self).load_dataset_finalize(split)





################ Older (deprecated class with image dataset augmentation support) ##############################3

class KerasDatasetLoader(DatasetLoader):
    def __init__(self, keras_dataset,
                 imsize=default_imsize, split=default_split,
                 batch_size=default_batch_size,
                 shuffle=default_shuffle, **kwargs):

        if isinstance(keras_dataset, tuple):
            (x_train, y_train), (x_test, y_test) = keras_dataset
        else:
            (x_train, y_train), (x_test, y_test) = keras_dataset.load_data()

        valid_splits = ['train', 'test']
        if not split in valid_splits:
            raise ValueError('Invalid split : %s' % split)

        if split == 'train':
            self._x = x_train
            self._y = y_train
        elif split == 'test':
            self._x = x_test
            self._y = y_test
        else:
            raise ValueError('Invalid split : %s' % split)

        self._imsize = imsize
        num_images = len(self._x)

        class_labels = kwargs['class_labels']
        kwargs['num_classes'] = len(class_labels)
        kwargs['classification'] = True

        self.need_to_fit_statistics = True # After dataset augmentations
                                           # have been defined, run fit()

        super(KerasDatasetLoader, self).__init__(
            num_images=num_images, batch_size=batch_size, shuffle=shuffle,
            imsize=imsize, split=split, **kwargs)


    def load(self, index=None):

        if index is None:
            index = self._cur_idx

        idx = self._idx_order[index]

        x = self._x[idx]
        if np.ndim(x) == 2:
            x = x[..., np.newaxis] # (32,32) --> (32,32,1)

        y = self._y[idx]

        self._cur_idx += 1
        return x, y

    def label_to_name(self, id):
        return self._class_labels[id]


class MNISTLoader(KerasDatasetLoader):
    def __init__(self, imsize=default_imsize, split=default_split,
                 batch_size=default_batch_size,
                 shuffle=default_shuffle, **kwargs):
        keras_dataset = keras.datasets.mnist
        class_labels = [str(i) for i in range(10)]
        super(MNISTLoader, self).__init__(keras_dataset,
            imsize, split, batch_size, shuffle, class_labels=class_labels, **kwargs)



class CIFAR10Loader(KerasDatasetLoader):
    def __init__(self, imsize=default_imsize, split=default_split,
                 batch_size=default_batch_size,
                 shuffle=default_shuffle, **kwargs):
        keras_dataset = keras.datasets.cifar10
        class_labels = cifar10_class_labels
        super(CIFAR10Loader, self).__init__(keras_dataset,
            imsize, split, batch_size, shuffle, class_labels=class_labels, **kwargs)



class CIFAR100Loader(KerasDatasetLoader):
    def __init__(self, label_mode='fine', imsize=default_imsize, split=default_split,
                 batch_size=default_batch_size,
                 shuffle=default_shuffle, **kwargs):

        keras_dataset = keras.datasets.cifar100.load_data(label_mode=label_mode)
        if label_mode == 'fine':
            class_labels = cifar100_fine_class_labels
        elif label_mode == 'coarse':
            class_labels = cifar100_coarse_class_labels
        else:
            raise ValueError('label mode must be "coarse" or "fine"')

        super(CIFAR100Loader, self).__init__(keras_dataset,
            imsize, split, batch_size, shuffle, class_labels=class_labels, **kwargs)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import xxhash
    from collections import OrderedDict
    import re

    hasher = lambda x: xxhash.xxh64(x).hexdigest()
    use_new_loader = True
    if use_new_loader:

        lambda_wgts = None #utl.lambda_weights(hp=1, lp=0, distillation_loss=0.01)

        split_name = 'val'
        new_loader = GTCCifar10DatasetLoader(split=split_name)
        new_loader.build(featurewise_center=False,
                         featurewise_std_normalization=False,
                         horizontal_flip=False,
                         vertical_flip=False,
                         lambda_weights=lambda_wgts)

        #count = 0
        #count_n = 0
        #for x, y in new_loader.flow(batch_size=16):
        #    count_n += x.shape[0]
        #    count += 1


        hash_counts = OrderedDict()
        hash_images = OrderedDict()
        hash_indexes = OrderedDict()
        all_hashes = []

        batch_size = 1
        save_dir = 'F:/SRI/bitnet/debug/cifar10/' + split_name + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        it = new_loader.flow(batch_size=batch_size, shuffle=False,
                             save_to_dir=save_dir,
                             save_stubs_only=True,
                             )
        n_tot = len(new_loader) // batch_size
        dup_count = 0
        for batch_i, (x, y) in tqdm(enumerate(it), total=n_tot):
            if batch_i == len(new_loader):
                break
            for j in range(x.shape[0]):
                if batch_i in [1768 - 1, 7189 - 1]:
                    aa = 1
                x_hash = hasher(x[j].tostring())

                all_hashes.append(x_hash)
                if x_hash in hash_counts:
                    dup_count += 1
                    hash_counts[x_hash] += 1
                    hash_indexes[x_hash].append(batch_i)
                    # hash_images[x_hash].append(x[j])
                    # joined_im = np.concatenate(hash_images[x_hash], axis=1).astype(np.uint8)
                    print('batch %d : Discovered duplicate %d ' % (batch_i, dup_count))
                    aa = 1
                else:
                    hash_counts[x_hash] = 1
                    hash_indexes[x_hash] = [batch_i]

        duplicates = OrderedDict([(k, v) for k, v in hash_indexes.items() if len(v) > 1])
        print('%d duplicates')
        print(duplicates)
        files = os.listdir(save_dir)

        all_hash_labels =  [ '-'.join(re.search('batch_(\d+)-(\d+)__(\w+)__(\d+)', s).groups()[2:4]) for s in files]
        all_labels =  [int(re.search('batch_(\d+)-(\d+)__(\w+)__(\d+)', s).groups()[3]) for s in files]


    load_mnist = False
    if load_mnist:
        np.random.seed(1234)
        loader = MNISTLoader(split='train', batch_size=8, shuffle=True, label_offset=0, convert_to_categorical=False)
        #def __init__(self, imsize=(224,224), split='val', batch_size=16, max_num_images=None, max_num_images_per_class=None, shuffle=False, dataset_root=None):

        (all_im, all_label) = loader.load_batch(batch_index=0)

        print('Loaded %s images/labels ' % (len(all_im)))
        for i in range(len(all_im)):
            label_i = all_label[i]
            if isinstance(label_i, np.ndarray) and len(label_i) > 1:
                label_i = str( np.argmax(label_i) )
            else:
                label_i = str(label_i)

            plt.imshow(all_im[i][:,:,0])
            plt.title(label_i)
            plt.show()
            a = 1


    load_cifar10 = False
    if load_cifar10:
        np.random.seed(1234)
        loader = CIFAR10Loader(split='train', batch_size=8, shuffle=True, label_offset=0)
        #def __init__(self, imsize=(224,224), split='val', batch_size=16, max_num_images=None, max_num_images_per_class=None, shuffle=False, dataset_root=None):

        (all_im, all_label) = loader.load_batch(batch_index=0)

        print('Loaded %s images/labels ' % (len(all_im)))
        for i in range(len(all_im)):
            label_i = loader.get_label_names( all_label[i].flatten()[0] )
            #if isinstance(label_i, np.ndarray) and len(label_i) > 1:
            #    label_i = str( np.argmax(label_i) )
            #else:
            #    label_i = str(label_i)

            plt.imshow(all_im[i])
            plt.title(label_i)
            plt.show()
            a = 1

    load_cifar100 = True
    if load_cifar100:
        np.random.seed(1234)
        loader = CIFAR100Loader(label_mode='fine', split='train', batch_size=16, shuffle=True, label_offset=0)
        # def __init__(self, imsize=(224,224), split='val', batch_size=16, max_num_images=None, max_num_images_per_class=None, shuffle=False, dataset_root=None):

        (all_im, all_label) = loader.load_batch(batch_index=0)

        print('Loaded %s images/labels ' % (len(all_im)))
        for i in range(len(all_im)):
            label_i = loader.get_label_names(all_label[i].flatten()[0])
            # if isinstance(label_i, np.ndarray) and len(label_i) > 1:
            #    label_i = str( np.argmax(label_i) )
            # else:
            #    label_i = str(label_i)

            plt.imshow(all_im[i])
            plt.title(label_i)
            plt.show()
            a = 1


