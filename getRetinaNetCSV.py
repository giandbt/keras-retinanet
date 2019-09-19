import os
import json
import pandas as pd
import cv2
import math

def get_label_csv(data_dir, data_type, output_dir, padding_perc = 0.0):

    images_dir = os.path.join(data_dir,'images', data_type)
    annotations = os.path.join(data_dir, 'annotations', '%s_400.json' %data_type)

    with open(annotations, 'r') as f:
        annon_dict = json.loads(f.read())

    # Initializes variables
    avail_imgs = annon_dict.keys()
    x1_list = []
    x2_list = []
    y1_list = []
    y2_list = []
    image_paths = []

    # Gets path for all images
    images = [os.path.join(images_dir, image) for image in os.listdir(images_dir) if 'jpg' or 'png' in image]
    for image in images:
        # read image (to determine size later)
        img = cv2.imread(image)

        # gets images Id
        img_id = os.path.basename(image)[:-4].lstrip('0')

        # ensures the image is in the dictionary key
        if not img_id in avail_imgs:
            continue

        for idx, annon in enumerate(annon_dict[img_id].keys()):

            # ensures we have a face detected
            if not annon_dict[img_id][annon]['age_gender_pred']:
                continue
            bbox = annon_dict[img_id][annon]['age_gender_pred']['detected']
            x1 = bbox['x1']
            x2 = bbox['x2']
            y1 = bbox['y1']
            y2 = bbox['y2']

            # add padding to face
            upper_y = int(max(0, y1 - (y2 - y1) * padding_perc))
            lower_y = int(min(img.shape[0], y2 + (y2 - y1) * padding_perc))
            left_x = int(max(0, x1 - (x2 - x1) * padding_perc))
            right_x = int(min(img.shape[1], x2 + (x2 - x1) * padding_perc))


            image_paths.append(image)
            x1_list.append(left_x)
            x2_list.append(right_x)
            y1_list.append(upper_y)
            y2_list.append(lower_y)

    # saves data in RetinaNet format
    data = {'image_path': image_paths,
            'x1': x1_list,
            'y1': y1_list,
            'x2': x2_list,
            'y2': y2_list,
            'class_name': ['human']*len(image_paths)}

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, '%s_labels.csv' %data_type), index=False, header=False)

def get_classes_csv(output_dir):
    data = {'class_name': ['human'],
            'id': [0]}

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'training_classes.csv'), index=False, header=False)

def wider_csv(data_dir, output_dir):

    # initializes lists
    x1_list = []
    x2_list = []
    y1_list = []
    y2_list = []
    image_paths = []

    annotations = os.path.join(data_dir, 'wider_face_split', 'wider_face_val_bbx_gt.txt')
    file = open(annotations,'r')
    image_path = os.path.join(data_dir, 'WIDER_val', 'images', file.readline())
    while True:

        num_imgs = int(file.readline())

        if num_imgs == 0:
            file.readline()

        for idx in range(num_imgs):
            face = file.readline().split(' ')
            x1 = int(face[0])
            y1 = int(face[1])
            x2 = int(face[0]) + int(face[2])
            y2 = int(face[1]) + int(face[3])

            if (x1 - x2)*(y1 - y2) < 400:
                continue

            image_paths.append(image_path.rstrip('\n'))
            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)

        image_path = os.path.join(data_dir,'WIDER_val', 'images', file.readline())
        if image_path == os.path.join(data_dir,'WIDER_val', 'images', ''):
            break

    # saves data in RetinaNet format
    data = {'image_path': image_paths,
            'x1': x1_list,
            'y1': y1_list,
            'x2': x2_list,
            'y2': y2_list,
            'class_name': ['human'] * len(image_paths)}

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'wider_validation_labels.csv'), index=False, header=False)

def fddb_csv(data_dir, output_dir):

    # initializes lists
    x1_list = []
    x2_list = []
    y1_list = []
    y2_list = []
    image_paths = []

    for fold in os.listdir(os.path.join(data_dir, 'FDDB-folds')):

        if not 'ellipseList' in fold:
            continue

        annotations = os.path.join(data_dir, 'FDDB-folds', fold)
        file = open(annotations,'r')
        path = file.readline().split('/')
        image_path = os.path.join(data_dir, 'originalPics', path[0], path[1], path[2], path[3], '%s.jpg' % path[4][:-1])

        while True:

            num_imgs = int(file.readline())

            for idx in range(num_imgs):
                face = file.readline().split(' ') # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
                x1 = int(float(face[3]) - float(face[1]))
                y1 = int(float(face[4]) - float(face[0]))
                x2 = int(float(face[3]) + float(face[1]))
                y2 = int(float(face[4]) + float(face[0]))

                if (x1 - x2)*(y1 - y2) < 400:
                    continue

                if not os.path.isfile(image_path):
                    continue

                image_paths.append(image_path.rstrip('\n'))
                x1_list.append(x1)
                y1_list.append(y1)
                x2_list.append(x2)
                y2_list.append(y2)

            path = file.readline().split('/')
            if path[0] == '':
                break
            image_path = os.path.join(data_dir, 'originalPics', path[0], path[1], path[2], path[3], '%s.jpg' % path[4][:-1])



    # saves data in RetinaNet format
    data = {'image_path': image_paths,
            'x1': x1_list,
            'y1': y1_list,
            'x2': x2_list,
            'y2': y2_list,
            'class_name': ['human'] * len(image_paths)}

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'fddb_validation_labels.csv'), index=False, header=False)

if __name__ == '__main__':
    #data_dir = '/home/giancarlo/Documents/custom_dataset_final_results/data'
    #data_type = 'train2017'
    output_dir = '/home/giancarlo/Documents/Face_Detection_test'
    #get_label_csv(data_dir, data_type, output_dir, padding_perc = 0.0)
    #get_classes_csv(output_dir)


    #data_dir = '/home/giancarlo/Documents/Face_Detection_test/WIDER'
    #wider_csv(data_dir, output_dir)

    data_dir = '/home/giancarlo/Documents/Face_Detection_test/FDDB'
    fddb_csv(data_dir, output_dir)