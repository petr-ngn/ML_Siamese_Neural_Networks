#Importing relevant libraries
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import cv2
from time import time

#Creating paths for saving cropped images.
if not os.path.exists('./cropped_images/'):
    os.makedirs('./cropped_images/')

#Defining path to the data
annotation_file = './data/Anno/'
identity_file = annotation_file + 'identity_CelebA.txt'
bbox_file = annotation_file + 'list_bbox_celeba.txt'

#Loading the image names (image) and labels (image_id)
identity = pd.read_csv(identity_file, sep = " ", header = None,
                        names = ['image', 'image_id'])

#Loading the bounding boxes of images
bbox = pd.read_csv(bbox_file, delim_whitespace = True)

#Function for cropping and resizing pictures with further exporting into .jpg format.
def face_crop_export(id_df, bbox_df, crop_all = False):

    #Start time initialization (for execution time measurement)
    start_time = time()

    #Function for cropping image
    def face_crop(image_name, bbox_df):

            #Loading Image
            image_path = './data/Img/img_celeba/' + image_name
            img = cv2.imread(image_path)

            #Setting bounding box coordinates
            startX = bbox_df[bbox_df['image_id'] == image_name]['x_1'].values[0]
            startY = bbox_df[bbox_df['image_id'] == image_name]['y_1'].values[0]
            endX = startX + bbox_df[bbox_df['image_id'] == image_name]['width'].values[0]
            endY = startY + bbox_df[bbox_df['image_id'] == image_name]['height'].values[0]
    
            #Cropping
            crop_img = img[startY:endY, startX:endX]
            output_img = crop_img
            
            #Resizing
            output_img = cv2.resize(crop_img, (224, 224))

            #Exporting the cropped image
            cv2.imwrite(f'./cropped_images/{image_name}', output_img)

    #List of images to crop (whether to crop all the images in the identity file or only those images which haven't been cropped yet)
    imgs_to_crop = id_df['image'] if crop_all == True else list(set(id_df['image'].tolist()).difference(os.listdir("./cropped_images/")))

    #Initialization for counting number of pictures which cannot be cropped.
    k = 0

    #Crop each image
    for ind, img in enumerate(imgs_to_crop):
        try:
            #Cropping image
            face_crop(img, bbox)

            #Print statement
            no_imgs = f'{ind + 1 - k}/{len(imgs_to_crop)}' #How many pictures have been cropped so far
            speed = (ind + 1- k)/(time() - start_time) * 60 #How many pictures have been cropped per minute on average
            eta = (len(imgs_to_crop) - (ind + 1 - k)) / ((ind + 1 - k)/(time() - start_time)) / 60 #Estimated remaining execution time
        
            print(f"{no_imgs} images cropped ... {speed:.2f} images/min | ETA: {eta:.2f} minutes", end = '\r')

        #Skip if the picture has no applicable bounding boxes.    
        except cv2.error as e:
            print(f"Image {img} has no applicable bounding boxes.")
            k += 1
            continue

#Cropping, resizing and exporting the images
face_crop_export(identity, bbox)