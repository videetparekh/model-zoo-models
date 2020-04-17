#  Copyright (c) 2020 by LatentAI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "LatentAI Commercial Software License". Please see the LICENSE
#  file that should have been included as part of this package.

import argparse
import os


def save_config(base_dir, args):
    # [print(vals) for vals in vars(args)]
    path = base_dir + '/' + args.name_of_experiment + '/'
    with open(path + 'config.txt', 'w') as f:
        for vals in vars(args):
            f.write(str(vals) + ' ' + str(getattr(args, vals)))
            f.write('\n')
    f.close()


def config(base_dir, name_of_experiment='lenet_on_mnist',
           optimizer='sgd',
           batch_size=32,
           num_epochs=1,
           learning_rate=.0002,
           max_number_bb_per_gt=10,
           image_size=(32, 32),
           num_channels=1,
           num_classes=10,
           lambda_bit_loss=1e-5,
           lambda_distillation_loss=0.01,
           weight_decay=.0002,
           learning_rate_decay=None,
           resume_training=False,
           path_to_pretrained_model=None,
           loading_pretrained_weights=False,
           path_to_pretrained_weights=None):
    if image_size is None or (not isinstance(image_size, tuple)):
        raise ValueError('pass correct image size of the data')

    parser = argparse.ArgumentParser()
    name_of_experiment = name_of_experiment + '_' + optimizer + '_weight_decay_' + str(
        weight_decay) + '_lam_bl_' + str(lambda_bit_loss) + '_lam_dl_' + str(
        lambda_distillation_loss)

    parser.add_argument(
        '--name_of_experiment',
        type=str,
        default=name_of_experiment,
        help='Name of the experiment')
    parser.add_argument(
        '--max_number_bb_per_gt',
        type=int,
        default=10,
        help='maximum number of bounding boxes per GT')
    parser.add_argument(
        '--batch_size', type=int, default=batch_size, help='batch size')
    parser.add_argument(
        '--num_epochs', type=int, default=num_epochs, help='num of epochs')
    parser.add_argument(
        '--optimizer', type=str, default=optimizer, help='optimizer')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=weight_decay,
        help='weight_decay for L2 regularization')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=learning_rate,
        help='learning rate')
    parser.add_argument(
        '--learning_rate_decay',
        type=float,
        default=learning_rate_decay,
        help='decay for the learning rate')
    parser.add_argument(
        '--image_size',
        type=tuple,
        default=(image_size[0], image_size[1], num_channels),
        help='image size')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=num_classes,
        help='number of classes in dataset')
    parser.add_argument(
        '--lambda_bit_loss',
        type=float,
        default=lambda_bit_loss,
        help='lambda for bit loss')
    parser.add_argument(
        '--lambda_distillation_loss',
        type=float,
        default=lambda_distillation_loss,
        help='lambda for distillation loss')
    parser.add_argument(
        '--resume_training',
        type=bool,
        default=resume_training,
        help='Continue training')
    parser.add_argument(
        '--path_to_pretrained_model',
        type=str,
        default=str(path_to_pretrained_model),
        help='path to pretrained_model')
    parser.add_argument(
        '--loading_pretrained_weights',
        type=bool,
        default=loading_pretrained_weights,
        help='Want to load weights from pretrained model')
    parser.add_argument(
        '--path_to_pretrained_weights',
        type=path_to_pretrained_weights,
        default=str(path_to_pretrained_weights),
        help='path to pretrained weight files')
    #gtc related cli parameters
    parser.add_argument("--verbose_output", default="None",
                        help="verbose output for all the gtc parameters")
    parser.add_argument("--basedirectory", default = base_dir,
                        help="base directory for the output model")
    parser.add_argument("--save_weights", default=True, type=bool, help="save the weights?")
    parser.add_argument("--print_layers_bits", default=True, type=bool, help="print bits per layer?")
    parser.add_argument("--save_model_checkpoint", default=True, type=bool, help="Save model checkpoint?")
    parser.add_argument("--dataset_name", default="mnist", help="Which dataset to use")
    parser.add_argument("--network_name", default="lenet", help="Which network to use")
    parser.add_argument("--savemodel", default=True, type=bool,
                        help="Save all the models at the end")
    args = parser.parse_args()
    path_expt = base_dir + '/' + args.name_of_experiment + '/'
    model_path = path_expt + 'model/'
    logs_path = path_expt + 'logs/'
    if not os.path.exists(path_expt):
        os.mkdir(path_expt)
        os.mkdir(model_path)
        os.mkdir(logs_path)

    save_config(base_dir, args)
    return args
