import cv2
import numpy as np
import os
from tqdm import tqdm
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import argparse

this_dir = "/".join(os.path.realpath(__file__).split('/')[:-1]) + '/'
parent_dir = '/'.join(this_dir.split('/')[:-2]) + '/'

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, 
                    default=this_dir+'0_input/1_input_maps/',
                    help='Input folder.')
parser.add_argument('-m', '--masks', type=str, 
                    default=this_dir+'0_input/2_masks/',
                    help='Masks folder.')
parser.add_argument('-o', '--output', type=str, 
                    default=this_dir+'0_input/3_output/',
                    help='Output folder.')
parser.add_argument('-x', '--x_mod', type=float, default=0.9,
                    help='Parameter for box stretching in x direction')
parser.add_argument('-y', '--y_mod', type=float, default=0.7,
                    help='Parameter for box stretching in y direction')
parser.add_argument('-b', '--min_box_dim', type=int, default=10,
                    help='Minimum bounding box dimension')
FLAGS = parser.parse_args()

def list_of_files(working_directory, extension):
    '''Get list of paths to files for further OCR with certain extension'''
    file_to_check = []
    if type(extension) == list:
        extension = tuple(extension)    
    for file in os.listdir(working_directory):
        if file.endswith(extension):
            file_to_check.append('{}/{}'.format(working_directory,file))
    return(file_to_check)

def main():
    before_pred = list_of_files(FLAGS.input, '.jpg')
    after_pred = list_of_files(FLAGS.masks, '.jpg')
    out_folder = FLAGS.output
    
    #Rectangle pred
    min_dim = FLAGS.min_box_dim
    for before in tqdm(before_pred[:]):
        for after in after_pred:
            if before.split('/')[-1] == after.split('/')[-1]:
                true_img = cv2.imread(before)

                shape_x = true_img.shape[1]
                shape_y = true_img.shape[0]
                x_multiplier = shape_x / 512
                y_multiplier = shape_y / 512

                im = cv2.imread(after)
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                contours, _ = cv2.findContours(gray, cv2.RETR_TREE, 
                                               cv2.CHAIN_APPROX_SIMPLE)
                area_list = [cv2.contourArea(cnt) for cnt in contours]
                contours_filtered = []
                for cnt, area in zip(contours, area_list):
                     if area > min_dim * x_multiplier * min_dim * y_multiplier:
                        contours_filtered.append(cnt)

                true_pred = true_img.copy()
                boxes_list = []
                for cnt in contours_filtered:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    boxes_list.append(box)

                mid_points = []
                length_width = []
                for cnt in boxes_list:
                    x = [point[0] for point in cnt]
                    y = [point[1] for point in cnt]
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX = (max(x) + min(x))/2
                        cY = (max(y) + min(y))/2

                    mid_points.append((cX, cY))

                    length = max(x) - min(x)
                    width = max(y) - min(y)
                    length_width.append((length, width))


                expanded_contours = []
                x_mod = FLAGS.x_mod
                y_mod = FLAGS.y_mod
                for cnt, mid, dims in zip(boxes_list, mid_points, length_width):
                    new_cnt = []
                    x_mid, y_mid = mid
                    length, width = dims
                    for point in cnt:
                        if point[0] < x_mid and point[1] < y_mid:
                            new_point = [[int(point[0]-(((1/x_mod)-1)/2)*length), 
                                          int(point[1]-(((1/y_mod)-1)/2)*width)]]
                        elif point[0] < x_mid and point[1] > y_mid:
                            new_point = [[int(point[0]-(((1/x_mod)-1)/2)*length), 
                                          int(point[1]+(((1/y_mod)-1)/2)*width)]]
                        elif point[0] > x_mid and point[1] > y_mid:
                            new_point = [[int(point[0]+(((1/x_mod)-1)/2)*length), 
                                          int(point[1]+(((1/y_mod)-1)/2)*width)]]
                        elif point[0] > x_mid and point[1] < y_mid:
                            new_point = [[int(point[0]+(((1/x_mod)-1)/2)*length), 
                                          int(point[1]-(((1/y_mod)-1)/2)*width)]]
                        elif point[0] == x_mid and point[1] > y_mid:
                            new_point = [[int(point[0]), 
                                          int(point[1]+(((1/y_mod)-1)/2)*width)]]
                        elif point[0] == x_mid and point[1] < y_mid:
                            new_point = [[int(point[0]), 
                                          int(point[1]-(((1/y_mod)-1)/2)*width)]]
                        elif point[0] > x_mid and point[1] == y_mid:
                            new_point = [[int(point[0]+(((1/x_mod)-1)/2)*length), 
                                          int(point[1])]]
                        elif point[0] < x_mid and point[1] == y_mid:
                            new_point = [[int(point[0]-(((1/x_mod)-1)/2)*length), 
                                          int(point[1])]]

                        new_cnt.append(new_point)
                    new_cnt = np.array(new_cnt, dtype = 'int32')
                    expanded_contours.append(new_cnt)

                all_rects = [Polygon([tuple(r[0][0]), 
                                      tuple(r[1][0]), tuple(r[2][0]),
                                      tuple(r[3][0])]) for r in expanded_contours]


                boxes_to_remove = []
                for rect in expanded_contours:
                    p1 = Point(tuple(rect[0][0]))
                    p2 = Point(tuple(rect[1][0]))
                    p3 = Point(tuple(rect[2][0]))
                    p4 = Point(tuple(rect[3][0]))

                    for poly in all_rects:
                        check = [poly.contains(p1), poly.contains(p2), 
                                 poly.contains(p3), poly.contains(p4)]
                        if sum(check) == 4:
                            p1_coords = list(p1.coords)
                            p2_coords = list(p2.coords)
                            p3_coords = list(p3.coords)
                            p4_coords = list(p4.coords)

                            box_to_remove = np.array([[list(p1_coords[0])], 
                                                      [list(p2_coords[0])], 
                                                      [list(p3_coords[0])], 
                                                      [list(p4_coords[0])]], 
                                                      dtype = 'int32')

                            boxes_to_remove.append(box_to_remove)

                if boxes_to_remove == []:
                    filtered_boxes = expanded_contours
                else:
                    filtered_boxes = []
                    for i in expanded_contours:
                        checker = [np.array_equal(i, j) for j in boxes_to_remove]
                        if sum(checker) == 0:
                            filtered_boxes.append(i)

                json_out = []
                for rect in filtered_boxes:
                    p1 = list(rect[0][0])
                    p2 = list(rect[1][0])
                    p3 = list(rect[2][0])
                    p4 = list(rect[3][0])

                    json_out.append((list(map(int, p1)), list(map(int, p2)), 
                                     list(map(int, p3)), list(map(int, p4))))
                 
                new_file = before.split('/')[-1].split('.')[0]
                with open(out_folder + '/' + new_file + '.json', 'w') as f:
                    json.dump(json_out, f)

                dd = cv2.drawContours(true_pred, filtered_boxes, -1, (0,0,255), 3)
                cv2.imwrite(out_folder + '/' + before.split('/')[-1], true_pred)  

if __name__ == '__main__':
    main()