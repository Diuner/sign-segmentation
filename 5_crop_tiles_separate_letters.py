import os
import cv2
import numpy as np
import json
from skimage.filters import threshold_local
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', type=str, default='', 
                    help='Input dir with images.')
parser.add_argument('-j', '--json_dir', type=str, default='', 
                    help='Dir with jsons with bounding boxes coordinates.')
parser.add_argument('-o', '--output_dir', type=str, default='', 
                    help='Output dir for separate signs.')
FLAGS = parser.parse_args()

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

def main():
    in_dir = FLAGS.input_dir
    box_dir = FLAGS.json_dir
    out_dir = FLAGS.output_dir
    out_dir = out_dir.replace('\\', '/')
    
    images = list_of_files(in_dir, '.jpg')
    images.sort()
    boxes = list_of_files(box_dir, '.json')
    boxes.sort()
    
    for img, js in tqdm(zip(images, boxes)):
        img = img.replace('\\', '/')
        out_name = img.split('/')[-1].split('.')[0]
        directory = out_dir + '/' + out_name
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        im = cv2.imread(img)
        y_max, x_max = im.shape[:2]

        with open(js, 'r') as f:
            bboxes = json.load(f)

        blank_img = np.zeros(shape=im.shape).astype(im.dtype)

        contours = [np.array([box[0], box[1], 
                             box[2], box[3]]) for box in bboxes]
        color = [255, 255, 255]
        cv2.fillPoly(blank_img, contours, color)
        result = cv2.bitwise_and(im, blank_img)

        for box in bboxes:
            x = [i[0] for i in box]
            y = [i[1] for i in box]
            x1 = min(x)
            x2 = max(x)
            y1 = min(y)
            y2 = max(y)

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > x_max:
                x2 = x_max
            if y2 > y_max:
                y2 = y_max

            crop = im[y1:y2+1, x1:x2+1]

            hsv_crop = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))[2]
            thresh_l = threshold_local(hsv_crop, 39, offset=15, method="gaussian")

            thresh = (hsv_crop > thresh_l).astype("uint8") * 255
            thresh = cv2.bitwise_not(thresh)

            to_pred = []
            for y_i in range(thresh.shape[0]):
                for x_i in range(thresh.shape[1]):
                    if thresh[y_i, x_i] == 255:
                        to_pred.append((x_i, y_i))
            
            if to_pred == []:
                print('No words found on image')
                continue
            
            db = DBSCAN(eps=1, min_samples=1).fit(to_pred)
            labels = db.labels_

            separate_objects = {}
            for lab in list(set(labels)):
                if lab != -1:
                    separate_object = []
                    for pixel, label in zip(to_pred, labels):
                        if label == lab:
                            separate_object.append(pixel)
                    if len(separate_object) > 20:
                        x_obj = [i[0] for i in separate_object]
                        y_obj = [i[1] for i in separate_object]
                        x1_obj = min(x_obj)
                        x2_obj = max(x_obj)
                        y1_obj = min(y_obj)
                        y2_obj = max(y_obj)

                        blank_obj = np.zeros(shape=thresh.shape).astype(im.dtype)
                        for obj in separate_object:
                            blank_obj[obj[1], obj[0]] = 255

                        cropped_object = blank_obj[y1_obj:y2_obj+1, x1_obj:x2_obj+1]
                        
                        y_extension = int(cropped_object.shape[0]*0.2)
                        x_extension = int(cropped_object.shape[1]*0.2)
                        
                        ext_blank_y = cropped_object.shape[0] + y_extension
                        ext_blank_x = cropped_object.shape[1] + x_extension

                        blank_extended = np.zeros(shape=(ext_blank_y,
                                        ext_blank_x)).astype(cropped_object.dtype)

                        y_cropped, x_cropped = cropped_object.shape
                        y_extended ,x_extended = blank_extended.shape

                        y_diff = int((y_extended - y_cropped)/2)
                        x_diff = int((x_extended - x_cropped)/2)

                        blank_extended[y_diff:y_diff+y_cropped, 
                                       x_diff:x_diff+x_cropped] = cropped_object
                        
                        x_in = str(x1+x1_obj)
                        y_in = str(y1+y1_obj)
                        
                        tile_name = out_name + "_x_" + x_in + '_y_' + y_in

                        cv2.imwrite(directory + '/' + tile_name + '.jpg', 
                                    blank_extended)
                    
if __name__ == '__main__':
    main()

