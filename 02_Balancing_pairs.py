#Importing relevant libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from time import time
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import os

#Creating paths for saving generated data frames.
if not os.path.exists('./csv/'):
    os.makedirs('./csv/')

#Defining path to the data
annotation_file = './data/Anno/'
identity_file = annotation_file + 'identity_CelebA.txt'

#Loading the image names (image) and labels (image_id)
identity = pd.read_csv(identity_file, sep = " ", header = None,
                        names = ['image', 'image_id'])

#List for dropping pictures which could not be cropped.
imgs_to_drop = list(set(identity['image'].tolist()).difference(os.listdir("./cropped_images/")))

#Dropping the pictures which could not be dropped from the data frame.
identity = identity[~identity['image'].isin(imgs_to_drop)]

#Parameters' initialization
random_seed = 123
test_size = 0.2
validation_size = 0.2

#Sampling from the dataset of whole pictures (from initial 202599 images, we sample only 70838 images)
labels_annot = pd.DataFrame(identity.image_id.value_counts(ascending = True)).query('image_id > 29').index.tolist()
identity_filtered = identity[identity['image_id'].isin(labels_annot)]

#Extracting images' names and their labels
imgs = identity_filtered['image']
labels = identity_filtered['image_id']

#Stratified split of the images into training, validation and test sets.
_, test_imgs, __, test_labels = train_test_split(imgs, labels,
                                               test_size = test_size,
                                               random_state = random_seed,        
                                               stratify = labels)

train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(_, __,
                                               test_size = validation_size/(1-test_size),
                                               random_state = random_seed,        
                                               stratify = __)

#Exporting data frames with the split images and their labels.
pd.concat((train_imgs, train_labels), axis = 1).to_csv('./csv/train_imgs_list.csv', index = False)
pd.concat((valid_imgs, valid_labels), axis = 1).to_csv('./csv/valid_imgs_list.csv', index = False)
pd.concat((test_imgs, test_labels), axis = 1).to_csv('./csv/test_imgs_list.csv', index = False)

#Function for generating balanced pairs of images (50% positive pairs, 50% negative pairs)
def pairs_generator(labels, image_names, export_name = None):

    #Start time initialization (for execution time measurement)
    start_time = time()

    #List for storing the pair labels (1 = positive pair | 0 = negative pair)
    pair_labels = []
    #List for storing the pair image names.
    pair_images_names = []

    #Array with all the labels
    unique_classes = np.unique(labels)

    #Dictionary for storing images' indices for each label.
    dict_idx = {i:np.where(labels == i)[0] for i in unique_classes}

    #Count initialization
    i = 0

    #For each image, find its positive pair and negative pair
    for idx_a in range(len(image_names)):

        #Current image - its label (person) and photo name
        label = labels[idx_a]
        current_image_name = image_names[idx_a]

        #Positive image - random picture of the same person
        idx_b = np.random.choice(dict_idx[label])
        
        #If the photos are the same (if the indices are the same), then again randomly select another picture.
        while True:
            if idx_b != idx_a:
                break
            else:
                idx_b = np.random.choice(dict_idx[label])
        
        #If the pair is existing in the list, select randomly another picture again.
        while True:
            positive_image_name = image_names[idx_b]
            pair_names = [sorted(pair) for pair in pair_images_names]
            current_pos_pair = sorted([current_image_name, positive_image_name])

            if current_pos_pair not in pair_names:
                break
            else:
                idx_b = np.random.choice(dict_idx[label])
        
        pair_images_names.append([current_image_name, positive_image_name])
        pair_labels.append([1])

        i += 1

        negative_index = np.random.choice(dict_idx[np.random.choice([i for i in dict_idx.keys()
                                                                        if i != label])])
        #If the pair is existing in the list, select randomly another picture again.
        while True:
            negative_image_name = image_names[negative_index]
            pair_names = [sorted(pair) for pair in pair_images_names]
            current_neg_pair = sorted([current_image_name, negative_image_name])

            if current_neg_pair not in pair_names:
                break
            else:
                negative_index = np.random.choice(dict_idx[np.random.choice([i for i in dict_idx.keys()
                                                                                if i != label])])

        pair_images_names.append([current_image_name, negative_image_name])
        pair_labels.append([0])
        
        i += 1
        
        #Print statement
        runtime = time()
        no_pairs = f'{i +1 }/{2*(len(image_names))}' #How many pairs have been created so far
        speed = (i + 1)/(runtime - start_time) * 60 #How many pairs have been created per minute on average
        eta = (2*(len(image_names)) - (i + 1)) / ((i + 1)/(runtime - start_time)) / 60 #Estimated remaining execution time

        print(f'{no_pairs} pairs created ... {speed:.2f} pairs/min | ETA: {eta:.2f} minutes', end = '\r')

    #Data frame storing all negative and positive pairs (with the image names) and their pair label (1 = positive | 0 = negative).
    final_df = pd.concat((pd.DataFrame((pair_images_names), columns = ['img_1', 'img_2']),
                            pd.DataFrame(pair_labels, columns = ['label'])),
                            axis = 1)

    if export_name != None:
        final_df.to_csv(f'./csv/{export_name}_pairs.csv', index  = False)

    return final_df

train_pairs = pairs_generator(train_labels, train_imgs, 'train')
valid_pairs = pairs_generator(valid_labels, valid_imgs, 'valid')
test_pairs = pairs_generator(test_labels, test_imgs, 'test')

#Function for plotting an image and its positive and negative pair
def plot_pairs(pairs_df, resize = True):
    
    #Choose random picture as a baseline
    base_img = np.random.choice(pairs_df['img_1'])
    
    #Filter pairs which include the baseline picture
    filtered_df = pairs_df[pairs_df['img_1'] == base_img]

    #Filter a positive pair of given baseline picture
    positive_img = filtered_df.query('label == 1')['img_2'].values[0]

    #Filter a negative pair of given baseline picture
    negative_img = filtered_df.query('label == 0')['img_2'].values[0]

    #Folder path definition
    folder_path = 'cropped_images' if resize == True else 'data/Img/img_celeba'

    #Plot the baseline picture
    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot(131)
    ax.set_title(f"Base picture - {base_img}")
    ax.imshow(cv2.cvtColor(cv2.imread(f'./{folder_path}/{base_img}'),
            cv2.COLOR_BGR2RGB))

    #Plot the positive picture
    ax = plt.subplot(132)
    ax.set_title(f"Positive picture - {positive_img}")
    ax.imshow(cv2.cvtColor(cv2.imread(f'./{folder_path}/{positive_img}'),
            cv2.COLOR_BGR2RGB))

    #Plot the negative picture
    ax = plt.subplot(133)
    ax.set_title(f"Negative picture - {negative_img}")
    ax.imshow(cv2.cvtColor(cv2.imread(f'./{folder_path}/{negative_img}'),
            cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()

plot_pairs(test_pairs, resize = False)