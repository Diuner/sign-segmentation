import os
import ntpath
import cv2
import pandas as pd
import numpy as np
import csv
import argparse

this_dir = "/".join(os.path.realpath(__file__).split('/')[:-1]) + '/'

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train_dir', type=str, 
            default=this_dir+'0_input/1_input_maps/', 
            help='Input dir with images to train and txt files with annotations.')
parser.add_argument('-o', '--mask_dir', type=str, 
            default=this_dir+'0_input/2_masks/', 
            help='Output dir for masks created by script.')
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


def path_leaf(path_list):
    '''Get names of files in specific directory without extensions'''
    file_name_list = []
    for path in path_list:
        head, tail = ntpath.split(path)
        file_name = tail.split('.')[0]
        file_name_list.append(file_name)
    return file_name_list

def shrink_poly(poly, r):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generating the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) >                     np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][0][1] - poly[0][0][1]), 
                           (poly[1][0][0] - poly[0][0][0]))
        poly[0][0][0] += R * r[0] * np.cos(theta)
        poly[0][0][1] += R * r[0] * np.sin(theta)
        poly[1][0][0] -= R * r[1] * np.cos(theta)
        poly[1][0][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][0][1] - poly[3][0][1]), 
                           (poly[2][0][0] - poly[3][0][0]))
        poly[3][0][0] += R * r[3] * np.cos(theta)
        poly[3][0][1] += R * r[3] * np.sin(theta)
        poly[2][0][0] -= R * r[2] * np.cos(theta)
        poly[2][0][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0][0] - poly[0][0][0]), 
                           (poly[3][0][1] - poly[0][0][1]))
        poly[0][0][0] += R * r[0] * np.sin(theta)
        poly[0][0][1] += R * r[0] * np.cos(theta)
        poly[3][0][0] -= R * r[3] * np.sin(theta)
        poly[3][0][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0][0] - poly[1][0][0]), 
                           (poly[2][0][1] - poly[1][0][1]))
        poly[1][0][0] += R * r[1] * np.sin(theta)
        poly[1][0][1] += R * r[1] * np.cos(theta)
        poly[2][0][0] -= R * r[2] * np.sin(theta)
        poly[2][0][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0][0] - poly[0][0][0]), 
                           (poly[3][0][1] - poly[0][0][1]))
        poly[0][0][0] += R * r[0] * np.sin(theta)
        poly[0][0][1] += R * r[0] * np.cos(theta)
        poly[3][0][0] -= R * r[3] * np.sin(theta)
        poly[3][0][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0][0] - poly[1][0][0]), 
                           (poly[2][0][1] - poly[1][0][1]))
        poly[1][0][0] += R * r[1] * np.sin(theta)
        poly[1][0][1] += R * r[1] * np.cos(theta)
        poly[2][0][0] -= R * r[2] * np.sin(theta)
        poly[2][0][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][0][1] - poly[0][0][1]), 
                           (poly[1][0][0] - poly[0][0][0]))
        poly[0][0][0] += R * r[0] * np.cos(theta)
        poly[0][0][1] += R * r[0] * np.sin(theta)
        poly[1][0][0] -= R * r[1] * np.cos(theta)
        poly[1][0][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][0][1] - poly[3][0][1]), 
                           (poly[2][0][0] - poly[3][0][0]))
        poly[3][0][0] += R * r[3] * np.cos(theta)
        poly[3][0][1] += R * r[3] * np.sin(theta)
        poly[2][0][0] -= R * r[2] * np.cos(theta)
        poly[2][0][1] -= R * r[2] * np.sin(theta)
    return poly

def main():
    train_dir = FLAGS.train_dir
    mask_dir = FLAGS.mask_dir
    
    images_list = list_of_files(train_dir, '.jpg')
    images_names = path_leaf(images_list)
    annot_list = list_of_files(train_dir, '.txt')
    
    number_of_images = len(images_list)
    number_of_annot = len(annot_list)
    
    #check if number of annotations matches number of images
    if number_of_images != number_of_annot:
        print("Number of annotations doesn't match number of images")
    else:
        print("Number of annotations and images is the same")
    
    #check if names of images matches names of annotations
    check_sum = 0
    for img, annot, name in zip(images_list, annot_list, images_names):
        if (path_leaf([img])[0] == path_leaf([annot])[0][3:] and 
            path_leaf([img])[0] == name):
            check_sum += 1
    if check_sum == number_of_images:
        print("Every image has it's annotation file")
    else:
        print("Check if every image has appropriate annotation file")
    
    #create mask files for train
    for img, annot, name in zip(images_list, annot_list, images_names):
        im = cv2.imread(img)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_annots = pd.read_csv(annot, header=None, quoting=csv.QUOTE_NONE,
                              names=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 
                                     'x4', 'y4', 'text'])
        t = im.copy()
        mask_img = np.zeros((t.shape[0], t.shape[1], 3), dtype=np.int32)

        for item in im_annots.itertuples():
            pts = np.array([[item.x1,item.y1],[item.x2,item.y2],
                            [item.x3,item.y3],[item.x4,item.y4]], np.int32)
            pts = pts.reshape((-1,1,2))
            r = [None, None, None, None]
            for i in range(4):
                r[i] = min(np.linalg.norm(pts[i] - pts[(i + 1) % 4]),
                           np.linalg.norm(pts[i] - pts[(i - 1) % 4]))
            shrink_pts = shrink_poly(pts, r)
            cv2.polylines(mask_img, [shrink_pts], True, (255,255,255))
            cv2.fillPoly(mask_img, np.int_([shrink_pts]), (255, 255, 255))

        if mask_dir[-1] == '/':
            cv2.imwrite("".join((mask_dir, name, '.jpg')), mask_img)
        else:
            cv2.imwrite("".join((mask_dir, '/', name, '.jpg')), mask_img)
    
    print('Mask creation done.')
    
if __name__ == '__main__':
    main()

