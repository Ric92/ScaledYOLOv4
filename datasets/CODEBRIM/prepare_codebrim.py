import os
from os import listdir, getcwd
from os.path import join
from voc_to_yolo import (convert_annotation, getImagesInDir)
from dataset import (test_set_check, split_train_test_by_id, check_annotations)
import shutil
from shutil import copyfile

codebrim_path = "/content/gdrive/MyDrive/datasets/PILOTING/CODEBRIM_original/original_dataset"
output_path = "/content/gdrive/MyDrive/datasets/PILOTING/CODEBRIM_original/modified_dataset"
classes = ['defect']

if __name__ == '__main__':
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    images_path = codebrim_path + "/images"
    annotations_path = codebrim_path + "/annotations"
    # get all images from CODEBRIM
    image_list = getImagesInDir(images_path)

    # split images labeled and unlabeled
    image_list, unused_image_list = check_annotations(codebrim_path, image_list, annotations_path)

    # split into train val and test  
    val, train = split_train_test_by_id(image_list, 0.2)

    print('Images ' + str(len(image_list)) + ' train ' + str(len(train)) + ' test ' + str(len(val)) + ' unused '+ str(len(image_list) - len(unused_image_list)))

    i = 0
    for dataset in [train, val]:
        s = 'val'
        if len(dataset) > len(val):
            s = 'train'
        
        label_path = codebrim_path + '/annotations'
        new_labels_path = output_path + '/labels/'+ s + '/'
        new_image_path = output_path + '/images/'+ s + '/'
        # create train val txt
        list_file = open(output_path + '/'+ s + '.txt', 'w')
        for image_path in dataset:
            # write image path in train val txt

            if not os.path.exists(new_labels_path):
                os.makedirs(new_labels_path)

            # convert VOC to yolo format
            convert_annotation(label_path, new_labels_path, image_path)
            
            if not os.path.exists(new_image_path):
                os.makedirs(new_image_path)
            shutil.copy(image_path, new_image_path)

            basename = os.path.basename(image_path)
            basename_no_ext = os.path.splitext(basename)[0]
            list_file.write(new_image_path + basename_no_ext + '.jpg\n')
            
            
            i = i + 1
            percentage_completed = 100 * i / len(image_list)
            if percentage_completed % 1 == 0:
                print('Progress completed -- ' + str(percentage_completed) +'%')


        list_file.close()

    list_file = open(output_path + '/unused.txt', 'w')
    for unused_image in unused_image_list:
        list_file.write(unused_image + '\n')
    list_file.close()