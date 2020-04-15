'''
Includes:
* A batch generator for SSD model training and inference which can perform online data agumentation
* An offline image processor that saves processed images and adjusted labels to disk

Copyright (C) 2017 Pierluigi Ferrari

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import division
from collections import defaultdict
import warnings
import numpy as np
import cv2
import random
import sklearn.utils
from copy import deepcopy
from PIL import Image
import csv
import os
from tqdm import tqdm
import imageio
import json
from bs4 import BeautifulSoup
import pickle
from utility_scripts import misc_utl as utl
from ssd import ssd_utils as ssd_utl
import inspect
import keras

# Image processing functions used by the generator to perform the following image manipulations:
# - Translation
# - Horizontal flip
# - Scaling
# - Brightness change
# - Histogram contrast equalization
import datasets.ssd_data_augmentations as ssd_aug
from collections import OrderedDict

default_labels_seq = ssd_aug.default_labels_seq
default_labels_format = ssd_aug.default_labels_format

date_repickle = (2020, 3, 8,  22, 0, 0)

class BatchGenerator(keras.utils.Sequence):
    '''
    A generator to generate batches of samples and corresponding labels indefinitely.

    Can shuffle the dataset consistently after each complete pass.

    Currently provides two methods to parse annotation data: A general-purpose CSV parser
    and an XML parser for the Pascal VOC datasets. If the annotations of your dataset are
    in a format that is not supported by these parsers, you could just add another parser
    method and still use this generator.

    Can perform image transformations for data conversion and data augmentation,
    for details please refer to the documentation of the `generate()` method.
    '''

    def __init__(self,
                 labels_format=None,
                 ssd_box_encoder=None,
                 imsize=None,
                 split=None,
                 #mean_color=None,
                 class_names=None,
                 class_idx_mapping=None
                 ):
        '''
        This class provides parser methods that you call separately after calling the constructor to assemble
        the list of image filenames and the list of labels for the dataset from CSV or XML files.

        In case you would like not to load any labels at all, simply pass a list of image filenames here.

        Arguments:
            box_output_format (list, optional): A list of five strings representing the desired order of the five
                items class ID, xmin, ymin, xmax, ymax in the generated data. The expected strings are
                'xmin', 'ymin', 'xmax', 'ymax', 'class_id'. If you want to train the model, this
                must be the order that the box encoding class requires as input. Defaults to
                `['class_id', 'xmin', 'ymin', 'xmax', 'ymax']`. Note that even though the parser methods are
                able to produce different output formats, the SSDBoxEncoder currently requires the format
                `['class_id', 'xmin', 'ymin', 'xmax', 'ymax']`. This list only specifies the five box parameters
                that are relevant as training targets, a list of filenames is generated separately.
            ssd_box_encoder (class SSDBoxEncoder)
            imsize: default image size to reshape to if no other augmentation parameters passed in 'build()'
            split (str, optional): name of dataset split
            class_names: names of the classes
            class_idx_mapping: class index mapping to convert saved labels
                    (used to convert COCO labels from 1..80 to 1..90)

        '''
        if labels_format is None:
            labels_format = default_labels_seq
        labels_format_dict = {name: labels_format.index(name) for name in default_labels_seq}

        self.labels_format = labels_format
        self.labels_format_dict = labels_format_dict

        if ssd_box_encoder is not None:
            assert isinstance(ssd_box_encoder, ssd_utl.SSDBoxEncoder)
        self.ssd_box_encoder = ssd_box_encoder


        self.batch_size = None
        self.imsize = imsize
        self.img_width = None
        self.img_height = None
        if imsize is not None:
            self.img_height, self.img_width = imsize[:2]
        #if mean_color is None:
        #    mean_color = (123, 117, 104)
        #self.mean_color = mean_color

        self.built = False
        self.class_idx_mapping = None
        self.split = split
        self.class_names = class_names
        self.class_idx_mapping = class_idx_mapping

        self.filenames = None
        self.labels = None
        self.image_ids = None
        self.eval_neutral = None
        #self._idxs_order = None

        warnings.filterwarnings("ignore", "The input (.*) could not be retrieved. It could be because a worker has died.",
                                UserWarning)


    @property
    def dataset_size(self):
        return self._dataset_size

    def get_dataset_size(self):
        return self._dataset_size

    def clip_dataset_size(self, max_size):
        if self.filenames:
            self.filenames = self.filenames[:max_size]
        if self.labels:
            self.labels = self.labels[:max_size]
        if self.image_ids:
            self.image_ids = self.image_ids[:max_size]
        if self.eval_neutral:
            self.eval_neutral = self.eval_neutral[:max_size]
        self._dataset_size = min(self._dataset_size, max_size)

    def parse_csv(self,
                  images_dir,
                  labels_filename,
                  input_format,
                  include_classes='all',
                  random_sample=False,
                  ret=False):
        '''
        Arguments:
            images_dir (str): The path to the directory that contains the images.
            labels_filename (str): The filepath to a CSV file that contains one ground truth bounding box per line
                and each line contains the following six items: image file name, class ID, xmin, xmax, ymin, ymax.
                The six items do not have to be in a specific order, but they must be the first six columns of
                each line. The order of these items in the CSV file must be specified in `input_format`.
                The class ID is an integer greater than zero. Class ID 0 is reserved for the background class.
                `xmin` and `xmax` are the left-most and right-most absolute horizontal coordinates of the box,
                `ymin` and `ymax` are the top-most and bottom-most absolute vertical coordinates of the box.
                The image name is expected to be just the name of the image file without the directory path
                at which the image is located. Defaults to `None`.
            input_format (list): A list of six strings representing the order of the six items
                image file name, class ID, xmin, xmax, ymin, ymax in the input CSV file. The expected strings
                are 'image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'. Defaults to `None`.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. Defaults to 'all', in which case all boxes will be included
                in the dataset.
            random_sample (float, optional): Either `False` or a float in `[0,1]`. If this is `False`, the
                full dataset will be used by the generator. If this is a float in `[0,1]`, a randomly sampled
                fraction of the dataset will be used, where `random_sample` is the fraction of the dataset
                to be used. For example, if `random_sample = 0.2`, 20 precent of the dataset will be randomly selected,
                the rest will be ommitted. The fraction refers to the number of images, not to the number
                of boxes, i.e. each image that will be added to the dataset will always be added with all
                of its boxes. Defaults to `False`.
            ret (bool, optional): Whether or not the image filenames and labels are to be returned.
                Defaults to `False`.

        Returns:
            None by default, optionally the image filenames and labels.
        '''

        # Set class members.
        self.images_dir = images_dir
        self.labels_filename = labels_filename
        self.input_format = input_format
        self.include_classes = include_classes

        # Before we begin, make sure that we have a labels_filename and an input_format
        if self.labels_filename is None or self.input_format is None:
            raise ValueError(
                "`labels_filename` and/or `input_format` have not been set yet. You need to pass them as arguments.")

        # Erase data that might have been parsed before
        self.filenames = []
        self.labels = []

        # First, just read in the CSV file lines and sort them.

        data = []

        with open(self.labels_filename, newline='') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            next(csvread)  # Skip the header row.
            for row in csvread:  # For every line (i.e for every bounding box) in the CSV file...
                if self.include_classes == 'all' or int(row[self.input_format.index(
                        'class_id')].strip()) in self.include_classes:  # If the class_id is among the classes that are to be included in the dataset...
                    box = []  # Store the box class and coordinates here
                    box.append(row[self.input_format.index(
                        'image_name')].strip())  # Select the image name column in the input format and append its content to `box`
                    for element in self.labels_format:  # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                        box.append(int(row[self.input_format.index(
                            element)].strip()))  # ...select the respective column in the input format and append it to `box`.
                    data.append(box)

        data = sorted(data)  # The data needs to be sorted, otherwise the next step won't give the correct result

        # Now that we've made sure that the data is sorted by file names,
        # we can compile the actual samples and labels lists

        current_file = data[0][0]  # The current image for which we're collecting the ground truth boxes
        current_labels = []  # The list where we collect all ground truth boxes for a given image
        add_to_dataset = False
        for i, box in enumerate(data):

            if box[0] == current_file:  # If this box (i.e. this line of the CSV file) belongs to the current image file
                current_labels.append(box[1:])
                if i == len(data) - 1:  # If this is the last line of the CSV file
                    if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0, 1)
                        if p >= (1 - random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
            else:  # If this box belongs to a new image file
                if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                    p = np.random.uniform(0, 1)
                    if p >= (1 - random_sample):
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                else:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(self.images_dir, current_file))
                current_labels = []  # Reset the labels list because this is a new file.
                current_file = box[0]
                current_labels.append(box[1:])
                if i == len(data) - 1:  # If this is the last line of the CSV file
                    if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0, 1)
                        if p >= (1 - random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))

        self.post_parsing()

        if ret:  # In case we want to return these
            return self.filenames, self.labels

    def parse_xml(self,
                  images_dirs,
                  image_set_filenames,
                  annotations_dirs=(),
                  classes=('background', 'Input', 'IP', 'OP', 'Output'),
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  pickle_parsed_file=True,
                  ret=False, verbose=True):
        '''
        This is an XML parser for the Pascal VOC datasets. It might be applicable to other datasets with minor changes to
        the code, but in its current form it expects the data format and XML tags of the Pascal VOC datasets.

        Arguments:
            images_dirs (list): A list of strings, where each string is the path of a directory that
                contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                into one (e.g. one directory that contains the images for Pascal VOC 2007, another that contains
                the images for Pascal VOC 2012, etc.).
            image_set_filenames (list): A list of strings, where each string is the path of the text file with the image
                set to be loaded. Must be one file per image directory given. These text files define what images in the
                respective image directories are to be part of the dataset and simply contains one image ID per line
                and nothing else.
            annotations_dirs (list, optional): A list of strings, where each string is the path of a directory that
                contains the annotations (XML files) that belong to the images in the respective image directories given.
                The directories must contain one XML file per image and the name of an XML file must be the image ID
                of the image it belongs to. The content of the XML files must be in the Pascal VOC format.
            classes (list, optional): A list containing the names of the object classes as found in the
                `name` XML tags. Must include the class `background` as the first list item. The order of this list
                defines the class IDs. Defaults to the list of Pascal VOC classes in alphabetical order.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. Defaults to 'all', in which case all boxes will be included
                in the dataset.
            exclude_truncated (bool, optional): If `True`, excludes boxes that are labeled as 'truncated'.
            exclude_difficult (bool, optional): If `True`, excludes boxes that are labeled as 'difficult'.
            ret (bool, optional): Whether or not the image filenames and labels are to be returned.

        Returns:
            None by default, optionally the image filenames and labels.
        '''
        # Set class members.
        self.images_dirs = images_dirs
        self.annotations_dirs = annotations_dirs
        self.image_set_filenames = image_set_filenames
        self.classes = classes
        # print (self.classes)
        self.include_classes = include_classes

        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []
        self.eval_neutral = []

        if not annotations_dirs:
            self.labels = None
            annotations_dirs = [None] * len(images_dirs)

        for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
            # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.

            classes_in_file = [cls.lower().replace(' ', '') for cls in self.classes]

            redo = False
            image_set_pickle_file = image_set_filename.replace('.txt', '.pkl')
            if os.path.exists(image_set_pickle_file) and not redo and not \
                    utl.fileOlderThan(image_set_pickle_file, date_repickle):
                if verbose:
                    print("Loading %s ... " % image_set_pickle_file)
                saved_data = pickle.load( open(image_set_pickle_file, 'rb') )
                filenames = saved_data['filenames']
                labels = saved_data['labels']
                image_ids = saved_data['image_ids']
                eval_neutrals = saved_data['eval_neutrals']

            else:
                filenames, labels, image_ids, eval_neutrals = [], [], [], []
                print('Scanning %s to collect image files & labels... ' % image_set_filename)

                with open(image_set_filename) as f:
                    image_ids = [line.strip() for line in f]  # Note: These are strings, not integers.

                # Loop over all images in this dataset.
                # for image_id in image_ids:
                for image_id in tqdm(image_ids, desc=os.path.basename(image_set_filename)):

                    filename = '{}'.format(image_id) + '.jpg'
                    filenames.append(os.path.join(images_dir, filename))

                    if annotations_dir is not None:
                        # Parse the XML file for this image.
                        with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                            soup = BeautifulSoup(f, 'xml')

                        folder = soup.folder.text  # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset an image belongs to.
                        # filename = soup.filename.text

                        boxes = []  # We'll store all boxes for this image here
                        eval_neutr = []
                        objects = soup.find_all('object')  # Get a list of all objects in this image

                        # Parse the data for each object
                        for obj in objects:
                            class_name = obj.find('name').text
                            # print class_name
                            class_id = classes_in_file.index(class_name)
                            # Check if this class is supposed to be included in the dataset
                            if (not self.include_classes == 'all') and (not class_id in self.include_classes): continue
                            pose = obj.pose.text
                            truncated = int(obj.truncated.text)
                            if exclude_truncated and (truncated == 1): continue
                            difficult = int(obj.difficult.text)
                            if exclude_difficult and (difficult == 1): continue
                            xmin = int(obj.bndbox.xmin.text)
                            ymin = int(obj.bndbox.ymin.text)
                            xmax = int(obj.bndbox.xmax.text)
                            ymax = int(obj.bndbox.ymax.text)

                            item_dict = {'folder': folder,
                                         'image_name': filename,
                                         'image_id': image_id,
                                         'class_name': class_name,
                                         'class_id': class_id,
                                         'pose': pose,
                                         'truncated': truncated,
                                         'difficult': difficult,
                                         'xmin': xmin,
                                         'ymin': ymin,
                                         'xmax': xmax,
                                         'ymax': ymax}
                            box = []
                            for item in self.labels_format:
                                box.append(item_dict[item])
                            boxes.append(box)
                            if difficult: eval_neutr.append(True)
                            else: eval_neutr.append(False)

                        # end loop over objects

                        # print 'size of boxex',len(boxes)

                        labels.append(boxes)
                        eval_neutrals.append(eval_neutr)
                    # endif have annotation_dir

                # end loop over image_ids

                if pickle_parsed_file:
                    print('Creating pickled version of %s for faster loading next time' % image_set_filename)
                    saved_data = dict(filenames=filenames, labels=labels,
                                      image_ids=image_ids, eval_neutrals=eval_neutrals)
                    print('Saving %s' % image_set_pickle_file)
                    pickle.dump(saved_data, open(image_set_pickle_file, 'wb') )

            # endif hickle file does not exist

            self.filenames += filenames
            self.labels += labels
            self.image_ids += image_ids
            self.eval_neutral += eval_neutrals
        # end loop over imagedirs

        self._dataset_size = len(self.filenames)

        self.post_parsing()

        if ret:
            return self.filenames, self.labels, self.image_ids, self.eval_neutral
        #if ret:
        #    return self.images, self.filenames, self.labels, self.image_ids, self.eval_neutral


    def parse_json(self,
                   images_dirs,
                   annotations_filenames,
                   ground_truth_available=False,
                   include_classes='all',
                   pickle_parsed_file=True,
                   ret=False, verbose=True):
        '''
        This is an JSON parser for the MS COCO datasets. It might be applicable to other datasets with minor changes to
        the code, but in its current form it expects the JSON format of the MS COCO datasets.

        Arguments:
            images_dirs (list, optional): A list of strings, where each string is the path of a directory that
                contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                into one (e.g. one directory that contains the images for MS COCO Train 2014, another one for MS COCO
                Val 2014, another one for MS COCO Train 2017 etc.).
            annotations_filenames (list): A list of strings, where each string is the path of the JSON file
                that contains the annotations for the images in the respective image directories given, i.e. one
                JSON file per image directory that contains the annotations for all images in that directory.
                The content of the JSON files must be in MS COCO object detection format. Note that these annotations
                files do not necessarily need to contain ground truth information. MS COCO also provides annotations
                files without ground truth information for the test datasets, called `image_info_[...].json`.
            ground_truth_available (bool, optional): Set `True` if the annotations files contain ground truth information.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. Defaults to 'all', in which case all boxes will be included
                in the dataset.
            ret (bool, optional): Whether or not the image filenames and labels are to be returned.

        Returns:
            None by default, optionally the image filenames and labels.
        '''
        self.images_dirs = images_dirs
        self.annotations_filenames = annotations_filenames
        self.include_classes = include_classes
        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []
        if not ground_truth_available:
            self.labels = None

        def load_annotation_json_file(json_file, use_pickle=True):
            if use_pickle:
                pickle_file = json_file.replace('.json', '.pkl')
                if os.path.exists(pickle_file):
                    if verbose:
                        print("Loading %s ... " % pickle_file)
                    annot_data = pickle.load(open(pickle_file, 'rb'))

                else:
                    print("Loading %s ... " % json_file)
                    annot_data = json.load(open(json_file))

                    with open(pickle_file, 'wb') as f:
                        print("Saving %s ... " % pickle_file)
                        pickle.dump(annot_data, f)
            else:
                annot_data = json.load(open(json_file))

            return annot_data



        # Build the dictionaries that map between class names and class IDs.
        #with open(annotations_filenames[0], 'r') as f:
        #    annotations = json.load(f)
        annotations = load_annotation_json_file(annotations_filenames[0])


        # Unfortunately the 80 MS COCO class IDs are not all consecutive. They go
        # from 1 to 90 and some numbers are skipped. Since the IDs that we feed
        # into a neural network must be consecutive, we'll save both the original
        # (non-consecutive) IDs as well as transformed maps.
        # We'll save both the map between the original
        self.cats_to_names = {}  # The map between class names (values) and their original IDs (keys)
        self.classes_to_names = []  # A list of the class names with their indices representing the transformed IDs
        self.classes_to_names.append(
            'background')  # Need to add the background class first so that the indexing is right.
        self.cats_to_classes = {}  # A dictionary that maps between the original (keys) and the transformed IDs (values)
        self.classes_to_cats = {}  # A dictionary that maps between the transformed (keys) and the original IDs (values)
        for i, cat in enumerate(annotations['categories']):
            self.cats_to_names[cat['id']] = cat['name']
            self.classes_to_names.append(cat['name'])
            self.cats_to_classes[cat['id']] = i + 1
            self.classes_to_cats[i + 1] = cat['id']

        # Iterate over all datasets.
        for images_dir, annotations_filename in zip(self.images_dirs, annotations_filenames):
            # Load the JSON file.

            parsed_annotations_filename = annotations_filename.replace('.json', '_parsed.pkl')

            redo = False
            #image_set_pickle_file = image_set_filename.replace('.txt', '.pkl')
            if os.path.exists(parsed_annotations_filename) and not redo:
                if verbose:
                    print("Loading %s" % parsed_annotations_filename)
                saved_data = pickle.load( open(parsed_annotations_filename, 'rb') )

                filenames = saved_data['filenames']
                labels = saved_data['labels']
                image_ids = saved_data['image_ids']

            else:
                filenames, labels, image_ids = [], [], []
                print('Scanning %s to collect image files & labels... ' % annotations_filename)

                annotations = load_annotation_json_file(annotations_filename)
                #with open(annotations_filename, 'r') as f:
                #    annotations = json.load(f)

                image_ids_to_annotations = None
                if ground_truth_available:
                    # Create the annotations map, a dictionary whose keys are the image IDs
                    # and whose values are the annotations for the respective image ID.
                    image_ids_to_annotations = defaultdict(list)
                    for annotation in annotations['annotations']:
                        image_ids_to_annotations[annotation['image_id']].append(annotation)

                # Iterate over all images in the dataset.
                for img in tqdm(annotations['images']):

                    filenames.append(os.path.join(images_dir, img['file_name']))
                    image_ids.append(img['id'])

                    if ground_truth_available:
                        # Get all annotations for this image.
                        annotations = image_ids_to_annotations[img['id']]
                        boxes = []
                        for annotation in annotations:
                            cat_id = annotation['category_id']
                            # Check if this class is supposed to be included in the dataset.
                            if (not self.include_classes == 'all') and (not cat_id in self.include_classes): continue
                            # Transform the original class ID to fit in the sequence of consecutive IDs.
                            class_id = self.cats_to_classes[cat_id]
                            xmin = annotation['bbox'][0]
                            ymin = annotation['bbox'][1]
                            width = annotation['bbox'][2]
                            height = annotation['bbox'][3]
                            # Compute `xmax` and `ymax`.
                            xmax = xmin + width
                            ymax = ymin + height
                            item_dict = {'image_name': img['file_name'],
                                         'image_id': img['id'],
                                         'class_id': class_id,
                                         'xmin': xmin,
                                         'ymin': ymin,
                                         'xmax': xmax,
                                         'ymax': ymax}
                            box = []
                            for item in self.labels_format:
                                box.append(item_dict[item])
                            boxes.append(box)
                        labels.append(boxes)

                # end loop over image_ids
                if pickle_parsed_file:
                    print('Creating pickled version of %s for faster loading next time' % annotations_filename)
                    saved_data = dict(filenames=filenames, labels=labels, image_ids=image_ids)
                    print('Saving %s' % parsed_annotations_filename)
                    pickle.dump(saved_data, open(parsed_annotations_filename, 'wb') )
                    print('  done.')
            # endif hickle file does not exist

            self.filenames += filenames
            self.labels    += labels
            self.image_ids += image_ids

        self._dataset_size = len(self.filenames)

        self.post_parsing()

        if ret:
            return self.filenames, self.labels, self.image_ids

    def post_parsing(self):
        if self.class_idx_mapping is not None:
            for im_idx in range(len(self.labels)):
                label_list_image_i = self.labels[im_idx]
                for lbl_idx in range(len(label_list_image_i)):
                    orig_id = self.labels[im_idx][lbl_idx][0]
                    mapped_class_id = self.class_idx_mapping[orig_id]
                    self.labels[im_idx][lbl_idx][0] = mapped_class_id
        a = 1

    def load(self, filenames=None,
             filenames_type='text',
             images_dir=None,
             labels=None,
             image_ids=None):

        '''
            filenames (string or list, optional): `None` or either a Python list/tuple or a string representing
                a filepath. If a list/tuple is passed, it must contain the file names (full paths) of the
                images to be used. Note that the list/tuple must contain the paths to the images,
                not the images themselves. If a filepath string is passed, it must point either to
                (1) a pickled file containing a list/tuple as described above. In this case the `filenames_type`
                argument must be set to `pickle`.
                Or
                (2) a text file. Each line of the text file contains the file name (basename of the file only,
                not the full directory path) to one image and nothing else. In this case the `filenames_type`
                argument must be set to `text` and you must pass the path to the directory that contains the
                images in `images_dir`.
            filenames_type (string, optional): In case a string is passed for `filenames`, this indicates what
                type of file `filenames` is. It can be either 'pickle' for a pickled file or 'text' for a
                plain text file. Defaults to 'text'.
            images_dir (string, optional): In case a text file is passed for `filenames`, the full paths to
                the images will be composed from `images_dir` and the names in the text file, i.e. this
                should be the directory that contains the images to which the text file refers.
                If `filenames_type` is not 'text', then this argument is irrelevant. Defaults to `None`.
            labels (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain Numpy arrays
                that represent the labels of the dataset.
            image_ids (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain the image
                IDs of the images in the dataset.

        '''

        # The variables `self.filenames`, `self.labels`, and `self.image_ids` below store the output from the parsers.
        # This is the input for the `generate()`` method. `self.filenames` is a list containing all file names of the image samples (full paths).
        # Note that it does not contain the actual image files themselves.
        # `self.labels` is a list containing one 2D Numpy array per image. For an image with `k` ground truth bounding boxes,
        # the respective 2D array has `k` rows, each row containing `(xmin, xmax, ymin, ymax, class_id)` for the respective bounding box.
        # Setting `self.labels` is optional, the generator also works if `self.labels` remains `None`.

        if not filenames is None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                with open(filenames, 'rb') as f:
                    if filenames_type == 'pickle':
                        self.filenames = pickle.load(f)
                    elif filenames_type == 'text':
                        self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
                    else:
                        raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
            else:
                raise ValueError(
                    "`filenames` must be either a Python list/tuple or a string representing a filepath (to a pickled or text file). The value you passed is neither of the two.")
        else:
            self.filenames = []

        if not labels is None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.labels = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.labels = labels
            else:
                raise ValueError(
                    "`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.labels = None

        if not image_ids is None:
            if isinstance(image_ids, str):
                with open(image_ids, 'rb') as f:
                    self.image_ids = pickle.load(f)
            elif isinstance(image_ids, (list, tuple)):
                self.image_ids = image_ids
            else:
                raise ValueError(
                    "`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.image_ids = None



    def build(self,
              # dataset augmentation details (moved to initialization call to match other dataset loaders)
              #train=True,

              mean_color=(123, 117, 104),

              subtract_mean=None,
              divide_by_stddev=None,
              swap_channels=False,

              photometric=False,
              expand=False,
              crop=False,
              flip=False,
              pad_in_eval_mode=False,

              img_width=None,
              img_height=None,
              limit_boxes=True,
              include_thresh=0.3,
              keep_images_without_gt=False,
              degenerate_box_handling='remove',

              lambda_weights=None
              ):
        """
              '''
              equalize=False,
              brightness=False,
              flip=False,
              translate=False,
              scale=False,
              max_crop_and_resize=False,
              random_pad_and_resize=False,
              random_crop=False,
              crop=False,
              resize=False,
              gray=False,
              '''
        Can perform image transformations for data conversion and data augmentation.
        Each data augmentation process can set its own independent application probability.
        The transformations are performed in the order of their arguments, i.e. translation
        is performed before scaling. All conversions and transforms default to `False`.

        `prob` works the same way in all arguments in which it appears. It must be a float in [0,1]
        and determines the probability that the respective transform is applied to a given image.

        Arguments:
            equalize (bool, optional): If `True`, performs histogram equalization on the images.
                This can improve contrast and lead the improved model performance.
            brightness (tuple, optional): `False` or a tuple containing three floats, `(min, max, prob)`.
                Scales the brightness of the image by a factor randomly picked from a uniform
                distribution in the boundaries of `[min, max]`. Both min and max must be >=0.
            flip (float, optional): `False` or a float in [0,1], see `prob` above. Flip the image horizontally.
                The respective box coordinates are adjusted accordingly.
            translate (tuple, optional): `False` or a tuple, with the first two elements tuples containing
                two integers each, and the third element a float: `((min, max), (min, max), prob)`.
                The first tuple provides the range in pixels for the horizontal shift of the image,
                the second tuple for the vertical shift. The number of pixels to shift the image
                by is uniformly distributed within the boundaries of `[min, max]`, i.e. `min` is the number
                of pixels by which the image is translated at least. Both `min` and `max` must be >=0.
                The respective box coordinates are adjusted accordingly.
            scale (tuple, optional): `False` or a tuple containing three floats, `(min, max, prob)`.
                Scales the image by a factor randomly picked from a uniform distribution in the boundaries
                of `[min, max]`. Both min and max must be >=0.
            max_crop_and_resize (tuple, optional): `False` or a tuple of four integers, `(height, width, min_1_object, max_#_trials)`.
                This will crop out the maximal possible image patch with an aspect ratio defined by `height` and `width` from the
                input image and then resize the resulting patch to `(height, width)`. This preserves the aspect ratio of the original
                image, but does not contain the entire original image (unless the aspect ratio of the original image is the same as
                the target aspect ratio) The latter two components of the tuple work identically as in `random_crop`.
                Note the difference to `random_crop`: This operation crops patches of variable size and fixed aspect ratio from the
                input image and then resizes the patch, while `random_crop` crops patches of fixed size and fixed aspect ratio from
                the input image. If this operation is active, it overrides both `random_crop` and `resize`.
            random_pad_and_resize (tuple, optional): `False` or a tuple of four integers and one float,
                `(height, width, min_1_object, max_#_trials, mix_ratio)`. The input image will first be padded with zeros such that
                it has the aspect ratio defined by `height` and `width` and afterwards resized to `(height, width)`. This preserves
                the aspect ratio of the original image an scales it to the maximum possible size that still fits inside a canvas of
                size `(height, width)`. The third and fourth components of the tuple work identically as in `random_crop`.
                `mix_ratio` is only relevant if `max_crop_and_resize` is active, in which case it must be a float in `[0, 1]` that
                decides what ratio of images will be processed using `max_crop_and_resize` and what ratio of images will be processed
                using `random_pad_and_resize`. If `mix_ratio` is 1, all images will be processed using `random_pad_and_resize`.
                Note the difference to `max_crop_and_resize`: While `max_crop_and_resize` will crop out the largest possible patch
                that still lies fully within the input image, the patch generated here will always contain the full input image.
                If this operation is active, it overrides both `random_crop` and `resize`.
            random_crop (tuple, optional): `False` or a tuple of four integers, `(height, width, min_1_object, max_#_trials)`,
                where `height` and `width` are the height and width of the patch that is to be cropped out at a random
                position in the input image. Note that `height` and `width` can be arbitrary - they are allowed to be larger
                than the image height and width, in which case the original image will be randomly placed on a black background
                canvas of size `(height, width)`. `min_1_object` is either 0 or 1. If 1, there must be at least one detectable
                object remaining in the image for the crop to be valid, and if 0, crops with no detectable objects left in the
                image patch are allowed. `max_#_trials` is only relevant if `min_1_object == 1` and sets the maximum number
                of attempts to get a valid crop. If no valid crop was obtained within this maximum number of attempts,
                the respective image will be removed from the batch without replacement (i.e. for each removed image, the batch
                will be one sample smaller).
            crop (tuple, optional): `False` or a tuple of four integers, `(crop_top, crop_bottom, crop_left, crop_right)`,
                with the number of pixels to crop off of each side of the images.
                The targets are adjusted accordingly. Note: Cropping happens before resizing.
            resize (tuple, optional): `False` or a tuple of 2 integers for the desired output
                size of the images in pixels. The expected format is `(height, width)`.
                The box coordinates are adjusted accordingly. Note: Resizing happens after cropping.
            gray (bool, optional): If `True`, converts the images to grayscale. Note that the resulting grayscale
                images have shape `(height, width, 1)`.
            subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
                of any shape that is broadcast-compatible with the image shape. The elements of this array will be
                subtracted from the image pixel intensity values. For example, pass a list of three integers
                to perform per-channel mean normalization for color images.
            divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
                floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
                intensity values will be divided by the elements of this array. For example, pass a list
                of three integers to perform per-channel standard deviation normalization for color images.
            swap_channels (bool, optional): If `True` the color channel order of the input images will be reversed,
                i.e. if the input color channel order is RGB, the color channels will be swapped to BGR.
            convert_to_3_channels (bool, optional): If `True`, single-channel images will be converted
                to 3-channel images.

        Yields:
            The next batch as a tuple of items as defined by the `returns` argument. By default, this will be
            a 2-tuple containing the processed batch images as its first element and the encoded ground truth boxes
            tensor as its second element if in training mode, or a 1-tuple containing only the processed batch images if
            not in training mode. Any additional outputs must be specified in the `returns` argument.
        """

        '''
        self.equalize=equalize
        self.brightness=brightness
        self.flip=flip
        self.translate=translate
        self.scale=scale
        self.max_crop_and_resize=max_crop_and_resize
        self.random_pad_and_resize=random_pad_and_resize
        self.random_crop=random_crop
        self.crop=crop
        self.resize=resize
        self.gray=gray
        '''

        self.subtract_mean=subtract_mean
        self.divide_by_stddev=divide_by_stddev
        self.swap_channels=swap_channels

        if img_width is not None:
            self.img_width = img_width
        else:
            img_width = self.imsize[0]
            self.img_width = img_width

        if img_height is not None:
            self.img_height = img_height
        else:
            img_height = self.imsize[1]
            self.img_height = img_height

        self.convert_to_3_channels=True
        self._lambda_weights = lambda_weights

        self.limit_boxes = limit_boxes
        self.include_thresh = include_thresh
        self.keep_images_without_gt = keep_images_without_gt
        self.degenerate_box_handling = degenerate_box_handling
        if self.degenerate_box_handling == 'remove':
            box_filter = ssd_aug.BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True,
                                   labels_format=self.labels_format_dict)
        else:
            box_filter = None


        if mean_color is None:
            mean_color = (123, 117, 104)

        self.mean_color = mean_color

        do_augmentations = photometric or expand or crop or flip

        if do_augmentations:
            aug_dict = dict(photometric=photometric, expand=expand, crop=crop, flip=flip)
            # for training dataset: do SSD augmentations
            transformations = [ssd_aug.SSDDataAugmentation(
                img_height=self.img_height, img_width=self.img_width,
                augmentations_dict=aug_dict,
                background=self.mean_color,
                labels_format=self.labels_format_dict)]
        else:
            # for validation dataset: just resize to network image size
            convert_to_3channels = ssd_aug.ConvertTo3Channels()
            random_pad = ssd_aug.RandomPadFixedAR(patch_aspect_ratio=img_width / img_height, labels_format=self.labels_format_dict)
            resize = ssd_aug.Resize(img_height, img_width, labels_format=self.labels_format_dict)

            if pad_in_eval_mode:
                transformations = [convert_to_3channels, random_pad, resize]
            else:
                transformations = [convert_to_3channels, resize]

        self.transformations = transformations

        if self.labels is not None:
            for transform in transformations:
                transform.labels_format = self.labels_format_dict

        self.box_filter = box_filter

        self.built = True


    '''
    Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement
    `on_epoch_end`. The method `__getitem__` should return a complete batch.
    '''
    def shuffle_idxs(self):
        if self.shuffle:
            self._idxs_order = np.random.permutation(self._dataset_size)
        elif self.file_order:
            self._idxs_order = np.argsort(self.image_ids)
        else:
            self._idxs_order = np.arange(self._dataset_size)

    def on_epoch_end(self):
        self.shuffle_idxs()

    def __len__(self):
        # Return number of complete batches (run build() first with batch_size)
        num_batches_tot = int( np.ceil( self._dataset_size / self.batch_size ) )
        if self.max_num_batches is None:
            num_batches = num_batches_tot
        else:
            num_batches = min(num_batches_tot, self.max_num_batches)

        return num_batches

    def __getitem__(self, item):
        return self.load_batch(item)


    def flow(self,
             batch_size=32,
             shuffle=True,
             file_order=False, # iterate in file order (overwrites shuffle)
             seed=None,

             max_num_samples=None,
             max_num_batches=None,
             max_num_classes=None,  # ignored (included to match flow_directory)
             returns=('processed_images', 'encoded_labels'),

             static=False):

        """
        batch_size (int, optional): The size of the batches to be generated.
        shuffle (bool, optional): Whether or not to shuffle the dataset before each pass.
            This option should always be `True` during training, but it can be useful to turn shuffling off
            for debugging or if you're using the generator for prediction.
        ssd_box_encoder (SSDBoxEncoder, optional): Only required if `train = True`. An SSDBoxEncoder object
            to encode the ground truth labels to the required format for training an SSD model.
        returns (set, optional): A set of strings that determines what outputs the generator yields. The generator's output
            is always a tuple with the processed images as its first element and, if in training mode, the encoded
            labels as its second element. Apart from that, the output tuple can contain additional outputs according
            to the keywords in `returns`. The possible keyword strings and their respective outputs are:
            * 'processed_images': An array containing the processed images. Will always be in the outputs, so it doesn't
                matter whether or not you include this keyword in the set.
            * 'encoded_labels': The encoded labels tensor. This is an array of shape `(batch_size, n_boxes, n_classes + 12)`
                that is the output of `SSDBoxEncoder.encode_y()`. Will always be in the outputs if in training mode,
                so it doesn't matter whether or not you include this keyword in the set if in training mode.
            * 'matched_anchors': The same as 'encoded_labels', but containing anchor box coordinates for all matched
                anchor boxes instead of ground truth coordinates. The can be useful to visualize what anchor boxes
                are being matched to each ground truth box. Only available in training mode.
            * 'processed_labels': The processed, but not yet encoded labels. This is a list that contains for each
                batch image a Numpy array with all ground truth boxes for that image. Only available if ground truth is available.
            * 'filenames': A list containing the file names (full paths) of the images in the batch.
            * 'image_ids': A list containing the integer IDs of the images in the batch. Only available if there
                are image IDs available.
            * 'inverse_transform': An array of shape `(batch_size, 4, 2)` that contains two coordinate conversion values for
                each image in the batch and for each of the four coordinates. These these coordinate conversion values makes
                it possible to convert the box coordinates that were predicted on a transformed image back to what those coordinates
                would be in the original image. This is mostly relevant for evaluation: If you want to evaluate your model on
                a dataset with varying image sizes, then you are forced to transform the images somehow (by resizing or cropping)
                to make them all the same size. Your model will then predict boxes for those transformed images, but for the
                evaluation you will need the box coordinates to be correct for the original images, not for the transformed
                images. This means you will have to transform the predicted box coordinates back to the original image sizes.
                Since the images have varying sizes, the function that transforms the coordinates is different for every image.
                This array contains the necessary conversion values for every coordinate of every image in the batch.
                In order to convert coordinates to the original image sizes, first multiply each coordinate by the second
                conversion value, then add the first conversion value to it. Note that the conversion will only be correct
                for the `resize`, `random_crop`, `max_crop_and_resize` and `random_pad_and_resize` transformations.
            * 'original_images': A list containing the original images in the batch before any processing.
            * 'original_labels': A list containing the original ground truth boxes for the images in this batch before any
                processing. Only available if ground truth is available.
            The order of the outputs in the tuple is the order of the list above. If `returns` contains a keyword for an
            output that is unavailable, that output will simply be skipped and not be part of the yielded tuple.


        limit_boxes (bool, optional): If `True`, limits box coordinates to stay within image boundaries
            post any transformation. This should always be set to `True`, even if you set `include_thresh`
            to 0. I don't even know why I made this an option. If this is set to `False`, you could
            end up with some boxes that lie entirely outside the image boundaries after a given transformation
            and such boxes would of course not make any sense and have a strongly adverse effect on the learning.
        include_thresh (float, optional): Only relevant if `limit_boxes` is `True`. Determines the minimum
            fraction of the area of a ground truth box that must be left after limiting in order for the box
            to still be included in the batch data. If set to 0, all boxes are kept except those which lie
            entirely outside of the image bounderies after limiting. If set to 1, only boxes that did not
            need to be limited at all are kept.
        keep_images_without_gt (bool, optional): If `True`, images for which there are no ground truth boxes
            (either because there weren't any to begin with or because random cropping cropped out a patch that
            doesn't contain any objects) will be kept in the batch. If `False`, such images will be removed
            from the batch.
        """

        if self.dataset_size == 0:
            raise ValueError("Cannot generate batches because you did not load a dataset.")

        #############################################################################################
        # Warn if any of the set returns aren't possible.
        #############################################################################################

        if self.labels is None:
            if any([ret in returns for ret in ['original_labels', 'processed_labels', 'encoded_labels', 'matched_anchors', 'evaluation-neutral']]):
                warnings.warn("Since no labels were given, none of 'original_labels', 'processed_labels', 'evaluation-neutral', 'encoded_labels', and 'matched_anchors' " +
                              "are possible returns, but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif self.ssd_box_encoder is None:
            if any([ret in returns for ret in ['encoded_labels', 'matched_anchors']]):
                warnings.warn("Since no label encoder was given, 'encoded_labels' and 'matched_anchors' aren't possible returns, " +
                              "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif self.ssd_box_encoder is None:
            if 'matched_anchors' in returns:
                warnings.warn("`label_encoder` is not an `SSDInputEncoder` object, therefore 'matched_anchors' is not a possible return, " +
                              "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))


        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_order = file_order
        self.seed = seed
        self.max_num_samples = max_num_samples
        self.max_num_batches = max_num_batches
        self.max_num_classes = max_num_classes  # ignored (included to match flow_directory)
        self.static = static
        self.returns = returns

        if not shuffle and self.split == 'train':
            utl.cprint('**************** warning: shuffle = False *************************', color=utl.Fore.RED)
        if file_order and self.split == 'train':
            utl.cprint('**************** warning: file_order = True *************************', color=utl.Fore.RED)

        if max_num_batches is not None:
            utl.cprint('**************** warning: max_num_batches = %s *************************' % (str(max_num_batches)), color=utl.Fore.RED)

        self.shuffle_idxs()



        return self


    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 file_order=False,
                 seed=None,
                 returns=('processed_images', 'encoded_labels'),
                 ):
        """
        Generate batches of samples and corresponding labels indefinitely from
        lists of filenames and labels.

        Returns two Numpy arrays, one containing the next `batch_size` samples
        from `filenames`, the other containing the corresponding labels from
        `labels`.

        """
        self.flow(batch_size=batch_size,
                 shuffle=shuffle,
                 file_order=file_order,
                 seed=seed,
                 returns=returns)

        current_batch_id = 0
        while True:
            if current_batch_id >= len(self):
                current_batch_id = 0
                self.shuffle_idxs()

            cur_batch = self.load_batch(current_batch_id)
            current_batch_id += 1
            yield cur_batch




    def load_batch(self, batch_id):
        # Get the image filepaths for this batch.

        if not self.built:
            #self.build( resize=(self.img_width, self.img_height) ) # set augmentations to default (ie. no augmentations)
            self.build( ) # set augmentations to default (ie. no augmentations)

        idx_start = batch_id * self.batch_size
        idx_end = min( (batch_id + 1) * self.batch_size, self._dataset_size)
        batch_sample_idxs = self._idxs_order[ np.arange(idx_start, idx_end) ]

        # Find out the indices of the box coordinates in the label data
        batch_filenames = [self.filenames[i] for i in batch_sample_idxs]

        batch_X = []
        # Load the images for this batch.
        for filename in batch_filenames:
            with Image.open(filename) as img:
                batch_X.append(np.array(img))

        # Get the labels for this batch (if there are any).
        if not (self.labels is None):
            batch_y = [deepcopy(self.labels[i]) for i in batch_sample_idxs]
        else:
            batch_y = None

        if not (self.eval_neutral is None):
            batch_eval_neutral = [self.eval_neutral[i] for i in batch_sample_idxs]
        else:
            batch_eval_neutral = None

        # Get the image IDs for this batch (if there are any).
        if not self.image_ids is None:
            batch_image_ids = [self.image_ids[i] for i in batch_sample_idxs]
        else:
            batch_image_ids = None

        # Create the array that is to contain the inverse coordinate transformation values for this batch.
        #batch_inverse_coord_transform = np.array([[[0, 1]] * 4] * self.batch_size,
        #                                         dtype=np.float)  # Array of shape `(batch_size, 4, 2)`, where the last axis contains an additive and a multiplicative scalar transformation constant.

        #if 'original_images' in self.returns:
        #    batch_original_images = deepcopy(batch_X)  # The original, unaltered images (edit: done after converting to 3-channels)
        #if 'original_labels' in self.returns and not batch_y is None:
        #    batch_original_labels = np.array(deepcopy(batch_y))  # The original, unaltered labels

        #current += self.batch_size
        batch_original_images = [None] * self.batch_size
        batch_original_labels = [None] * self.batch_size

        batch_items_to_remove = []  # In case we need to remove any images from the batch because of failed random cropping, store their indices in this list.
        batch_inverse_transforms = []


        if 'original_images' in self.returns:
            batch_original_images = deepcopy(batch_X)
            for i in range(self.batch_size):
                if (batch_original_images[i].ndim == 2):
                    if self.convert_to_3_channels:
                        # Convert the 1-channel image into a 3-channel image.
                        batch_X[i] = np.stack([batch_X[i]] * 3, axis=-1)
                    else:
                        # batch_X[i].ndim must always be 3, even for single-channel images.
                        batch_X[i] = np.expand_dims(batch_X[i], axis=-1)



        for i in range(len(batch_X)):

            #img_height, img_width = batch_X[i].shape[0], batch_X[i].shape[1]

            if batch_y is not None:
                # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
                if (len(batch_y[i]) == 0) and not self.keep_images_without_gt:
                    batch_items_to_remove.append(i)
                # Convert labels into an array (in case it isn't one already), otherwise the indexing below breaks.
                batch_y[i] = np.array(batch_y[i])

                # For COCO dataset, map the coco annotation class labels [1..80] to the
                # original class idxs for the network [1..90]
                # Edit: moved to do this immediately on loading, during post_parsing()
                #if self.class_idx_mapping is not None and len(batch_y[i]) > 0:
                #    orig_ids = batch_y[i][:, 0].astype(np.int32)
                #    mapped_class_ids = self.class_idx_mapping[ orig_ids ]
                #    batch_y[i][:, 0] = mapped_class_ids

                if 'original_labels' in self.returns and not batch_y is None:
                    batch_original_labels[i] = np.array(deepcopy(batch_y[i]))  # The original, unaltered labels
                                                                    # after remapping, but before cropping/augmentation


        #########################################################################################
        # Maybe perform image transformations.
        #########################################################################################

        for i in range(len(batch_X)):

            if not (self.labels is None):
                # Convert the labels for this image to an array (in case they aren't already).
                batch_y[i] = np.array(batch_y[i])
                # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
                if (batch_y[i].size == 0) and not self.keep_images_without_gt:
                    batch_items_to_remove.append(i)
                    batch_inverse_transforms.append([])
                    continue

            # Apply any image transformations we may have received.
            if self.transformations:

                inverse_transforms = []

                for transform in self.transformations:

                    if self.labels is not None:

                        if ('inverse_transform' in self.returns) and (
                                'return_inverter' in inspect.signature(transform).parameters):
                            batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i],
                                                                                  return_inverter=True)
                            inverse_transforms.append(inverse_transform)
                        else:
                            batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])

                        if batch_X[i] is None:  # In case the transform failed to produce an output image, which is possible for some random transforms.
                            batch_items_to_remove.append(i)
                            batch_inverse_transforms.append([])
                            continue

                    else:

                        if ('inverse_transform' in self.returns) and (
                                'return_inverter' in inspect.signature(transform).parameters):
                            batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                            inverse_transforms.append(inverse_transform)
                        else:
                            batch_X[i] = transform(batch_X[i])

                batch_inverse_transforms.append(inverse_transforms[::-1])



            #########################################################################################
            # Check for degenerate boxes in this batch item.
            #########################################################################################

            if not (self.labels is None):

                xmin = self.labels_format_dict['xmin']
                ymin = self.labels_format_dict['ymin']
                xmax = self.labels_format_dict['xmax']
                ymax = self.labels_format_dict['ymax']

                if (batch_y[i].size == 0):
                    if not self.keep_images_without_gt:
                        batch_items_to_remove.append(i)

                elif np.any(batch_y[i][:,xmax] - batch_y[i][:,xmin] <= 0) or np.any(batch_y[i][:,ymax] - batch_y[i][:,ymin] <= 0):
                    if self.degenerate_box_handling == 'warn':
                        warnings.warn("Detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, batch_y[i]) +
                                      "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. " +
                                      "This could mean that your dataset contains degenerate ground truth boxes, or that any image transformations you may apply might " +
                                      "result in degenerate ground truth boxes, or that you are parsing the ground truth in the wrong coordinate format." +
                                      "Degenerate ground truth bounding boxes may lead to NaN errors during the training.")
                    elif self.degenerate_box_handling == 'remove':
                        batch_y[i] = self.box_filter(batch_y[i])
                        if (batch_y[i].size == 0) and not self.keep_images_without_gt:
                            batch_items_to_remove.append(i)

        #########################################################################################
        # Remove any items we might not want to keep from the batch.
        #########################################################################################


        if batch_items_to_remove:
            for j in sorted(batch_items_to_remove, reverse=True):
                # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
                batch_X.pop(j)
                batch_filenames.pop(j)
                if batch_inverse_transforms: batch_inverse_transforms.pop(j)
                if not (self.labels is None): batch_y.pop(j)
                if not (self.image_ids is None): batch_image_ids.pop(j)
                if not (self.eval_neutral is None): batch_eval_neutral.pop(j)
                if 'original_images' in self.returns: batch_original_images.pop(j)
                if 'original_labels' in self.returns and not (self.labels is None): batch_original_labels.pop(j)

        '''
        if not self.keep_images_without_gt:
            # If any batch items need to be removed because of failed random cropping, remove them now.
            batch_inverse_coord_transform = np.delete(batch_inverse_coord_transform, batch_items_to_remove, axis=0)
            batch_X = np.delete(batch_X, batch_items_to_remove, axis=0)
            for j in sorted(batch_items_to_remove, reverse=True):
                # This isn't efficient, but it hopefully should not need to be done often anyway.
                batch_filenames.pop(j)
                if not batch_y is None: batch_y.pop(j)
                if not batch_image_ids is None: batch_image_ids.pop(j)
                if 'original_images' in self.returns: batch_original_images.pop(j)
                if 'original_labels' in self.returns and not batch_y is None: batch_original_labels.pop(j)
        '''

        # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes
        #          or varying numbers of channels. At this point, all images must have the same size and the same
        #          number of channels.
        batch_X = np.array(batch_X)
        if (batch_X.size == 0):
            raise ValueError(
                "You produced an empty batch. This might be because the images in the batch vary " +
                "in their size and/or number of channels. Note that after all transformations " +
                "(if any were given) have been applied to all images in the batch, all images " +
                "must be homogenous in size along all axes.")


        #########################################################################################
        # If we have a label encoder, encode our labels.
        #########################################################################################

        have_encoder = False
        batch_y_encoded, batch_matched_anchors = None, None
        if self.ssd_box_encoder is not None and self.labels is not None:
            have_encoder = True
            if 'matched_anchors' in self.returns:
                batch_y_encoded, batch_matched_anchors = self.ssd_box_encoder.encode_y(batch_y, diagnostics=True)
            else:
                batch_y_encoded = self.ssd_box_encoder.encode_y(batch_y, diagnostics=False)


        if self._lambda_weights is not None:
            batch_y_encoded = self._lambda_weights.AugmentOutput(batch_y_encoded)

        # Compile the output.
        ret = []
        for ret_name in self.returns:
            if ret_name == 'original_images':
                ret.append(batch_original_images)
            elif ret_name == 'original_labels':
                ret.append(batch_original_labels if batch_y is not None else '[orig labels unavailable]')
            elif ret_name == 'processed_images':
                ret.append(batch_X)
            elif ret_name == 'encoded_labels':
                ret.append(batch_y_encoded if have_encoder else '[encoded_labels unavailable]')
            elif ret_name == 'evaluation-neutral':
                ret.append(batch_eval_neutral if self.eval_neutral is not None else '[evaluation-neutral unavailable]')
            elif ret_name == 'matched_anchors':
                ret.append(batch_matched_anchors if have_encoder else '[matched_anchors unavailable]' )
            elif ret_name == 'processed_labels':
                ret.append(batch_y if batch_y is not None else '[processed labels unavailable]')
            elif ret_name == 'filenames':
                ret.append(batch_filenames)
            elif ret_name == 'image_ids':
                ret.append(batch_image_ids if batch_image_ids is not None else '[image_ids unavailable]')
            elif ret_name == 'inverse_transform':
                ret.append(batch_inverse_transforms)
            else:
                raise ValueError('Unrecognized return identifier : %s' % ret_name)

        return ret

    def get_filenames_labels(self):
        '''
        Returns:
            The list of filenames, the list of labels, and the list of image IDs.
        '''
        return self.filenames, self.labels, self.image_ids

    def get_n_samples(self):
        '''
        Returns:
            The number of image files in the initialized dataset.
        '''
        return len(self.filenames)

    def label_to_name(self, id):
        return self._label_to_name[id]




