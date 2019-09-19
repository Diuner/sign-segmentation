import os
import numpy as np
import tensorflow as tf
import cv2
import random
from tensorflow.keras.models import load_model
import argparse

this_dir = "/".join(os.path.realpath(__file__).split('/')[:-1]) + '/'
parent_dir = '/'.join(this_dir.split('/')[:-2]) + '/'

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, 
                    default=this_dir+'0_input/1_input_maps/',
                    help='Folder with input maps.')
parser.add_argument('-o', '--output', type=str, 
                    default=this_dir+'0_input/2_masks/',
                    help='Folder with masks.')
parser.add_argument('-m', '--model', type=str, 
                    default='',
                    help='Path to model.')
parser.add_argument('--cuda', type=str, default='0',
                    help='Graphic card use. "-1" - only CPU, \
                    "0" - only one card, "0,1" - two cards, etc.')
parser.add_argument('-w', '--width', type=int, default=512,
                    help='Image width during model train.')
parser.add_argument('-e', '--height', type=int, default=512,
                    help='Image height during model train.')
parser.add_argument('-s', '--seed', type=int, default=42,
                    help='Seed for random operations.')
parser.add_argument('-ch', '--channel', type=int, default=3,
                    help='Number of channels in image.')
parser.add_argument('-t', '--treshold', type=float, default=0.5,
                    help='Threshold for prediction from network')
FLAGS = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda

IMG_WIDTH = FLAGS.width
IMG_HEIGHT = FLAGS.height
IMG_CHANNELS = FLAGS.channel
seed = FLAGS.seed
random.seed = seed
np.random.seed = seed

def list_of_files(working_directory, extension):
    '''
    Get list of paths to files for further OCR with certain extension
    :param working_directory: directory to search for files
    :param extension: single extension or list of extensions to search for
    :result: list containing files with given extensions
    
    >>>list_of_files(masks_path, ['.jpg', '.JPEG', '.png'])
    ['/home/maski/K18_232_2_1-023.png',
     '/home/maski/K18_236A_2_1-002.png']
    '''
    file_to_check = []
    for file in os.listdir(working_directory):
        if type(extension) == list:
            for end in extension:
                if file.endswith(str(end)):
                    file_to_check.append('{}/{}'.format(working_directory, file))
        elif file.endswith(str(extension)):
            file_to_check.append('{}/{}'.format(working_directory, file))
    return(file_to_check)

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def main():
    # Model load
    model = load_model(FLAGS.model, custom_objects={'mean_iou': mean_iou})
    full_test_images = list_of_files(FLAGS.input, 
                                     ['.jpg', '.JPG', '.png', '.PNG'])
    out_folder = FLAGS.output
    
    for i in range(len(full_test_images)):
        try:
            im = cv2.imread(full_test_images[i])
            saved_shape = im.shape[:2]

            im_resized = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT), 
                                    interpolation = cv2.INTER_LINEAR)
            im_in = np.zeros((1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), 
                             dtype=np.uint8)
            im_in[0] = im_resized
            pred = model.predict(im_in, verbose=1, batch_size=1)
            pred_t = np.squeeze((pred > FLAGS.treshold).astype(np.uint8))

            true_size_pred = cv2.resize(pred_t, (saved_shape[1], saved_shape[0]), 
                                        interpolation = cv2.INTER_LINEAR)
            cv2.imwrite(out_folder + '/' + full_test_images[i].split('/')[-1], 
                        true_size_pred.astype('uint8') * 255)
        except:
            continue
    
    print("Prediction done")

if __name__ == '__main__':
    main()

