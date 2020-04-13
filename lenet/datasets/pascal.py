import os
import glob
import imageio
import numpy as np
import xmltodict
from utility_scripts import misc_utl as utl
from datasets.dataset_loaders import DatasetLoader, DownloadDataset
import ssd.ssd_info as ssd

# Keep the Pascal_VOC path as the root, as the downloaded tar files unzip from this level.
_pascal_root = utl.datasets_root() + '/Pascal_VOC/'

# All the data is below this folder
_pascal_data_subdir = '/VOCdevkit/'

from datasets.ssd_data_loader import BatchGenerator

class GTCPascalLoader(BatchGenerator):

    def __init__(self, years=(2007, 2012), split='train', imsize=None,
                 condensed_labels=None,  # not used. Just here for compatibility with COCO Loader
                 ssd_box_encoder=None, verbose=True):

        DownloadPascalDataset(verbose=False)
        if not isinstance(years, list):
            years=[years]
        if split == 'test':
            if list(years) == [2007, 2012]:
                years = [2007] # only have test data for 2007

        dataset_type = ssd.SSDDatasetType(name='pascal', years=years, split=split)

        datasets_info = ssd.get_datasets_info(dataset_type.spec_list())
        images_dirs = datasets_info['image_dirs']
        annotations_dirs = datasets_info['annot_dirs']
        image_set_filenames = datasets_info['id_files']

        class_names = ssd.get_pascal_class_names(include_bckg_class=True)
        class_idx_mapping = np.arange(0, 21)

        self.dataset_name = str(dataset_type)

        super(GTCPascalLoader, self).__init__(
            imsize=imsize, split=split, ssd_box_encoder=ssd_box_encoder,
            class_names=class_names, class_idx_mapping=class_idx_mapping)

        self.parse_xml(
            images_dirs=images_dirs,
            image_set_filenames=image_set_filenames,
            annotations_dirs=annotations_dirs,
            classes=class_names,
            include_classes='all',
            exclude_truncated=False,
            exclude_difficult=False,ret=False, verbose=verbose)


class PascalLoader_old(DatasetLoader):
    def __init__(self, split='val', batch_size=1, year=2007, imsize=None, max_num_images=None, shuffle=False, root_dir=None, **kwargs):

        if root_dir is None:
            root_dir = _pascal_root  # replace with dir on your machine

        pascal_data_dir = root_dir + _pascal_data_subdir

        self._pascal_voc_root_dir = root_dir
        self._pascal_year_subdir = pascal_data_dir + 'VOC%d/' % year
        valid_splits = ['train', 'val', 'test']
        if not split in valid_splits:
            raise ValueError('Invalid split : %s' % split)

        self._imdir = self._pascal_year_subdir + 'JPEGImages/'
        self._annotdir = self._pascal_year_subdir + 'Annotations/'

        self._max_num_images = max_num_images

        split_ids_file = self._pascal_year_subdir + 'ImageSets/Main/' + split + '.txt'
        with open(split_ids_file, 'r') as f:
            split_ids = f.read().splitlines()

        self._file_ext = '.jpg'
        all_filenames = glob.glob(self._imdir + '*' + self._file_ext)
        all_file_ids = [ os.path.splitext( os.path.basename(f))[0] for f in all_filenames]
        file_idxs_use = [i for i,id in enumerate(all_file_ids) if id in split_ids]

        assert(len(file_idxs_use) == len(split_ids))

        num_images = len(file_idxs_use)

        idx_use = list(range(len(all_filenames)))

        if max_num_images is not None and num_images > max_num_images:
            idx_random_order = np.random.permutation(np.arange(num_images))

            idx_use = np.sort( idx_random_order[:max_num_images] )
            num_images = max_num_images
            all_file_ids = [all_file_ids[i] for i in idx_use]

        self._all_file_ids = all_file_ids

        kwargs['num_classes'] = 20

        super(PascalLoader_old, self).__init__(
            num_images=num_images, batch_size=batch_size, shuffle=shuffle, imsize=imsize, **kwargs)

    def load(self, index=None):
        return self.load_image_and_annotation(index)

    def load_image_and_annotation(self, index=None):

        if index is None:
            index = self._cur_idx

        idx = self._idx_order[index]

        im_file = self._imdir + self._all_file_ids[idx] + self._file_ext
        annot_file = self._annotdir + self._all_file_ids[idx] + '.xml'

        im = self.load_image(im_file)
        annot = self.load_annotation(annot_file)

        self._cur_idx += 1
        return im, annot


    def load_annotation(self, annot_file):
        with open(annot_file) as f:
            s = f.read()
        xml_data = xmltodict.parse(s)
        all_objects = xml_data['annotation']['object']
        if not isinstance(all_objects, list):
            all_objects = [all_objects]
        return all_objects




def DownloadPascalDataset(verbose=True):

    delete_downloaded_tar_files = False

    all_pascal_archive_files = [
        dict(url='http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
             sample_file='VOCdevkit/VOC2012/ImageSets/Main/aeroplane_trainval.txt'),
        dict(url='http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
             sample_file='VOCdevkit/VOC2007/ImageSets/Main/aeroplane_trainval.txt'),
        dict(url='http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
             sample_file='VOCdevkit/VOC2007/ImageSets/Main/aeroplane_test.txt')]

    DownloadDataset(all_pascal_archive_files, _pascal_root,
                    delete_downloaded_tar_files, verbose=verbose)



if __name__ == "__main__":

    DownloadPascalDataset(verbose=True)
    import ssd.ssd_utils as ssd_utl
    import networks.net_types as net
    import matplotlib.pyplot as plt
    import cv2
    import xxhash
    import hickle as hkl
    hasher = lambda x: xxhash.xxh64(x).hexdigest()

    a = 1
    ssd_net_type = net.VGGSSDType(style='keras')
    ssd_config = ssd_utl.SSDConfig(net_type=ssd_net_type)
    predictor_sizes = ssd_utl.predictor_sizes_vgg_300

    ssd_encoder = ssd_utl.SSDBoxEncoder(ssd_config, predictor_sizes=predictor_sizes)
    do_augmentations = True
    if do_augmentations:
        augmentations_dict = dict(photometric=True, expand=True, crop=True, flip=True)
    else:
        augmentations_dict = {}

    returns = ['processed_images', 'encoded_labels', 'processed_labels',
               'original_images', 'original_labels', 'filenames', 'matched_anchors']

    loader = GTCPascalLoader(years=[2007], split='trainval', ssd_box_encoder=ssd_encoder)
    loader.build(**augmentations_dict, img_height=300, img_width=300)
    train_gen = loader.flow(batch_size=32, returns=returns)

    class_names = loader.class_names
    #x,y = train_gen[0]
    a = 1
    path_base = 'F:/SRI/bitnet/test_ssd/'

    x, y_encoded, y_labels, x_orig, y_orig, filenames, matched_anchors = train_gen[0]
    for j,fn in enumerate(filenames):
        filebase = os.path.basename(fn).replace('.jpg', '_gtc.png')

        text_thickness = 1

        y_pred_i = y_labels[j]
        labels_dict = ssd_utl.boxarray2dict(y_pred_i, ssd_config=ssd_config)

        im2 = ssd_utl.drawLabelsOnImage(x[j], labels_dict, class_names=class_names)
        imageio.imwrite(path_base + filebase, im2)





    y_encoded = np.round(y_encoded, 10) # ignore floating point differences
    matched_anchors = np.round(matched_anchors, 10)
    #y_pred = model.predict(x)
    print('x         : ', hasher(x))
    print('y_encoded : ', hasher(y_encoded))
    #print('y_pred    : ', hasher(y_pred))
    print('y_labels  : ', y_labels)
    print('y_labels(hash) : ', hasher(np.array(y_labels)))
    print('m_anchors : ', hasher(matched_anchors))

    S = hkl.load('F:/SRI/bitnet/ssd_y_pred_true300_vgg.hkl')

    #it = train_gen.generate()
    #x = next(it)
    #for x in it:
    #    print(len(x))
    #    pass


    loader2 = PascalLoader_old(split='val')
    val_im, val_annot = loader2.load_batch(batch_index=0)

    #category_labels = ssd.coco_labels

    for i in range(len(val_im)):
        image_i = val_im[i].copy()
        annot_i = val_annot[i]

        for i, annot in enumerate(annot_i):
            xmin, ymin, xmax, ymax = tuple([int(z) for z in annot['bndbox'].values()])
            cv2.rectangle(image_i, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(image_i, '%s ' % annot['name'], (xmin, ymax - 15), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (255, 0, 0), thickness=1)


        plt.imshow(image_i)
        a = 1

    print('Loaded %s images/labels ' % (len(val_im)))
    a = 1

    """
    boxes = val_annot.boxes
    classes = val_annot.classes
    # category_index = 0
    vis_utl.visualize_boxes_and_labels_on_image_array(
        image_i,
        boxes,  # output_dict['boxes'],
        classes,  # output_dict['classes'],
        None,  # output_dict['scores'],
        category_labels,
        instance_masks=None,
        use_normalized_coordinates=True,
        line_thickness=2)
    """