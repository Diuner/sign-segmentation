import os
import numpy as np
import tensorflow as tf
import random
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import concatenate, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse

this_dir = "/".join(os.path.realpath(__file__).split('/')[:-1]) + '/'

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train_dir', type=str, 
                    default=this_dir+'0_input/1_input_maps/', 
                    help='Folder with train images.')
parser.add_argument('-m', '--mask_dir', type=str, 
                    default=this_dir+'0_input/2_masks/', 
                    help='Folder with train masks.')
parser.add_argument('-ht', '--image_height', type=int, default=512, 
                    help='Height of train image.')
parser.add_argument('-wd', '--image_width', type=int, default=512, 
                    help='Width of train image.')
parser.add_argument('-sd', '--seed', type=int, default=42, 
                    help='Random seed.')
parser.add_argument('--cuda', type=str, default='0', 
                    help='Graphic card use. "-1" - only CPU, \
                    "0" - only one card, "0,1" - two cards, etc.')
parser.add_argument('--model', type=str, default='model.h5', 
                    help='Model output name.')
parser.add_argument('--epochs', type=int, default=1000, 
                    help='Max number of epochs.')
parser.add_argument('--patience', type=int, default=25, 
                    help='Patience.')
parser.add_argument('--batch', type=int, default=3, 
                    help='Batch size.')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda

def list_of_files(working_directory, extension):
    '''Get list of paths to files for further OCR with certain extension'''
    file_to_check = []
    if type(extension) == list:
        extension = tuple(extension)    
    for file in os.listdir(working_directory):
        if file.endswith(extension):
            file_to_check.append('{}/{}'.format(working_directory,file))
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
    #general variables
    train_dir = FLAGS.train_dir
    mask_dir = FLAGS.mask_dir
    IMG_WIDTH = FLAGS.image_height
    IMG_HEIGHT = FLAGS.image_width
    IMG_CHANNELS = 3
    seed = FLAGS.seed
    random.seed = seed
    np.random.seed = seed

    images_list = list_of_files(train_dir, ('.jpg', '.JPG', '.png', '.PNG'))
    mask_list = list_of_files(mask_dir, ('.jpg', '.JPG', '.png', '.PNG'))
    
    #X and Y sets creation
    X_t = np.zeros((len(images_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                   dtype=np.uint8)
    Y_t = np.zeros((len(mask_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    
    X_position = 0
    for img in images_list:
        X_read = imread(img)[:,:,:IMG_CHANNELS]
        X_resized = resize(X_read, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                           preserve_range=True)
        X_t[X_position] = X_resized
        X_position += 1
        
    Y_position = 0
    for img in mask_list:
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        mask_ = imread(img, as_gray=True)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), 
                               mode='constant', preserve_range=True), 
                               axis=-1)
        mask = np.maximum(mask, mask_)
        Y_t[Y_position] = mask
        Y_position += 1
    
    #train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_t, Y_t, 
                                                        test_size=0.1, 
                                                        random_state=seed)
    
    # Model implementation
    input_image = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    VGG16_1 = Convolution2D(64,(3,3), padding='same')(input_image)
    VGG16_1 = BatchNormalization(momentum=0.997, epsilon=1e-5, 
                                 scale=True)(VGG16_1)
    VGG16_1 = Activation('elu')(VGG16_1)
    VGG16_1 = Convolution2D(64,(3,3), padding='same')(VGG16_1)
    VGG16_1 = BatchNormalization(momentum=0.997, epsilon=1e-5, 
                                 scale=True)(VGG16_1)
    VGG16_1 = Activation('elu')(VGG16_1)
    VGG16_1 = MaxPooling2D((2,2), strides=(2,2))(VGG16_1)

    VGG16_2 = Convolution2D(128,(3,3), padding='same')(VGG16_1)
    VGG16_2 = BatchNormalization(momentum=0.997, epsilon=1e-5, 
                                 scale=True)(VGG16_2)
    VGG16_2 = Activation('elu')(VGG16_2)
    VGG16_2 = Convolution2D(128,(3,3), padding='same')(VGG16_2)
    VGG16_2 = BatchNormalization(momentum=0.997, epsilon=1e-5, 
                                 scale=True)(VGG16_2)
    VGG16_2 = Activation('elu')(VGG16_2)
    VGG16_2 = MaxPooling2D((2,2), strides=(2,2))(VGG16_2)

    VGG16_3 = Convolution2D(256,(3,3), padding='same')(VGG16_2)
    VGG16_3 = BatchNormalization(momentum=0.997, epsilon=1e-5, 
                                 scale=True)(VGG16_3)
    VGG16_3 = Activation('elu')(VGG16_3)
    VGG16_3 = Convolution2D(256,(3,3), padding='same')(VGG16_3)
    VGG16_3 = BatchNormalization(momentum=0.997, epsilon=1e-5, 
                                 scale=True)(VGG16_3)
    VGG16_3 = Activation('elu')(VGG16_3)
    VGG16_3 = Convolution2D(256,(3,3), padding='same')(VGG16_3)
    VGG16_3 = BatchNormalization(momentum=0.997, epsilon=1e-5, 
                                 scale=True)(VGG16_3)
    VGG16_3 = Activation('elu')(VGG16_3)
    VGG16_3 = MaxPooling2D((2,2), strides=(2,2))(VGG16_3)

    VGG16_4 = Convolution2D(256,(3,3), padding='same')(VGG16_3)
    VGG16_4 = BatchNormalization(momentum=0.997, epsilon=1e-5, 
                                 scale=True)(VGG16_4)
    VGG16_4 = Activation('elu')(VGG16_4)
    VGG16_4 = Convolution2D(256,(3,3), padding='same')(VGG16_4)
    VGG16_4 = BatchNormalization(momentum=0.997, epsilon=1e-5, 
                                 scale=True)(VGG16_4)
    VGG16_4 = Activation('elu')(VGG16_4)
    VGG16_4 = Convolution2D(256,(3,3), padding='same')(VGG16_4)
    VGG16_4 = BatchNormalization(momentum=0.997, epsilon=1e-5, 
                                 scale=True)(VGG16_4)
    VGG16_4 = Activation('elu')(VGG16_4)
    VGG16_4 = MaxPooling2D((2,2), strides=(2,2))(VGG16_4)

    VGG16_5 = Convolution2D(256,(3,3), padding='same')(VGG16_4)
    VGG16_5 = BatchNormalization(momentum=0.997, epsilon=1e-5, 
                                 scale=True)(VGG16_5)
    VGG16_5 = Activation('elu')(VGG16_5)
    VGG16_5 = Convolution2D(256,(3,3), padding='same')(VGG16_5)
    VGG16_5 = BatchNormalization(momentum=0.997, epsilon=1e-5, 
                                 scale=True)(VGG16_5)
    VGG16_5 = Activation('elu')(VGG16_5)
    VGG16_5 = Convolution2D(256,(3,3), padding='same')(VGG16_5)
    VGG16_5 = BatchNormalization(momentum=0.997, epsilon=1e-5, 
                                 scale=True)(VGG16_5)
    VGG16_5 = Activation('elu')(VGG16_5)
    VGG16_5 = MaxPooling2D((2,2), strides=(2,2))(VGG16_5)

    FMB_1 = UpSampling2D((2,2))(VGG16_5)
    FMB_1 = concatenate([FMB_1, VGG16_4])
    FMB_1 = Convolution2D(128,(1,1), padding='same')(FMB_1)
    FMB_1 = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(FMB_1)
    FMB_1 = Activation('elu')(FMB_1)
    FMB_1 = Convolution2D(128,(3,3), padding='same')(FMB_1)
    FMB_1 = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(FMB_1)
    FMB_1 = Activation('elu')(FMB_1)

    FMB_2 = UpSampling2D((2,2))(FMB_1)
    FMB_2 = concatenate([FMB_2, VGG16_3])
    FMB_2 = Convolution2D(64,(1,1), padding='same')(FMB_2)
    FMB_2 = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(FMB_2)
    FMB_2 = Activation('elu')(FMB_2)
    FMB_2 = Convolution2D(64,(3,3), padding='same')(FMB_2)
    FMB_2 = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(FMB_2)
    FMB_2 = Activation('elu')(FMB_2)

    FMB_3 = UpSampling2D((2,2))(FMB_2)
    FMB_3 = concatenate([FMB_3, VGG16_2])
    FMB_3 = Convolution2D(32,(1,1), padding='same')(FMB_3)
    FMB_3 = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(FMB_3)
    FMB_3 = Activation('elu')(FMB_3)
    FMB_3 = Convolution2D(32,(3,3), padding='same')(FMB_3)
    FMB_3 = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(FMB_3)
    FMB_3 = Activation('elu')(FMB_3)

    FMB_4 = UpSampling2D((2,2))(FMB_3)
    FMB_4 = Convolution2D(16,(1,1), padding='same')(FMB_4)
    FMB_4 = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(FMB_4)
    FMB_4 = Activation('elu')(FMB_4)
    FMB_4 = Convolution2D(16,(3,3), padding='same')(FMB_4)
    FMB_4 = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(FMB_4)
    FMB_4 = Activation('elu')(FMB_4)

    FMB_5 = UpSampling2D((2,2))(FMB_4)
    FMB_5 = Convolution2D(8,(1,1), padding='same')(FMB_5)
    FMB_5 = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(FMB_5)
    FMB_5 = Activation('elu')(FMB_5)
    FMB_5 = Convolution2D(8,(3,3), padding='same')(FMB_5)
    FMB_5 = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(FMB_5)
    FMB_5 = Activation('elu')(FMB_5)

    FMB_6 = Convolution2D(4,(3,3), padding='same')(FMB_5)
    FMB_6 = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(FMB_6)
    FMB_6 = Activation('elu')(FMB_6)

    inside_score = Convolution2D(1, (1,1), padding='same', 
                                 name='inside_score')(FMB_6)

    model = Model(input_image, inside_score)
    
    #Optimizer and loss function selection
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    earlystopper = EarlyStopping(patience=FLAGS.patience, verbose=1)
    checkpointer = ModelCheckpoint(FLAGS.model, verbose=1, save_best_only=True)
    
    #Model fitting
    results = model.fit(X_train, Y_train, validation_split=0.1, 
                        batch_size=FLAGS.batch, epochs=FLAGS.epochs,
                        callbacks=[earlystopper, checkpointer])

if __name__ == '__main__':
    main()

