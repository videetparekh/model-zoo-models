import os
import tensorflow as tf
import imageio
import logging
import argparse
import numpy as np

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

def evaluate(args):
    input_model = args.input_path
    ###############################
    #   Load the model
    ###############################
    print("Graph file Dir: {}".format(input_model))

    # Start the Session
    sess = tf.compat.v1.InteractiveSession()
    logging.debug(sess)
    try:
        tf.compat.v1.saved_model.loader.load(sess, tags=['train'],
                                                    export_dir=input_model)
    except RuntimeError:
        try:
            tf.compat.v1.saved_model.loader.load(sess, tags=['serve'],
                                                    export_dir=input_model)
        except RuntimeError:
            raise RuntimeError("The Saved Model has no tags, \
                                ['train'] or ['serve']")

    print('Model is loaded')
    ###############################
    #   Create list of images
    ###############################
    image_paths = []

    if os.path.isdir(args.test_path):
        for inp_file in os.listdir(args.test_path):
            if os.path.isdir(args.test_path + '/' + inp_file):
                for inp in os.listdir(args.test_path + '/' + inp_file):
                    image_paths += [args.test_path + '/' + inp_file + '/' + inp]
            else:
                image_paths += [args.test_path + inp_file]
    else:
        image_paths += [args.test_path]
    
    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
    if not os.path.exists(args.basedirectory):
        os.mkdir(args.basedirectory)
    file_object = open(args.basedirectory +'/EvalIndex.txt', 'a')
    input_tensor = sess.graph.get_tensor_by_name('Placeholder:0')
    # the main loop
    output = sess.graph.get_tensor_by_name('dense_2/MatMul:0')
    for image_path in image_paths:
        image = imageio.imread(image_path)
        image = np.expand_dims(image,axis=2)
        image = np.expand_dims(image,axis=0)
        print(image_path)
        dict_eval = {input_tensor : image}
        prediction = sess.run(tf.compat.v1.math.softmax(output), feed_dict = dict_eval)[0]
        prediction = np.nonzero(prediction)[0][0]
        file_object.write(image_path + ' ' + str(prediction) + '\n')
    file_object.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='Path to pb model.')
    parser.add_argument('--test_path', default='mnist_examples/images/', help='Path to a directory with images.')
    parser.add_argument('--basedirectory', help='Directory for the output')
    args = parser.parse_args()

    print('\n')
    print(args)
    print('\n')


    print('Evaulating...')
    evaluate(args)
    print('Evaluation complete.')
