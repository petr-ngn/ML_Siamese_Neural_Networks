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

#Parameters' initialization
random_seed = 123
test_size = 0.2
validation_size = 0.2

#Function for reading and subsampling the images.
def images_subsampling(identity_file = './data/Anno/identity_CelebA.txt'):

        #Reading the images' names and labels.
        identity = pd.read_csv(identity_file, sep = " ", header = None,
                            names = ['image', 'image_id'])
        
        #Dropping the images which could not be cropped.
        imgs_to_drop = list(set(identity['image'].tolist()).difference(os.listdir("./cropped_images/")))
        identity = identity[~identity['image'].isin(imgs_to_drop)]

        #Filtering/subsampling only a part of images due to the computional and capacity limits.
        labels_annot = pd.DataFrame(identity.image_id.value_counts(ascending = True)).query('image_id > 29').index.tolist()
        identity_filtered = identity[identity['image_id'].isin(labels_annot)]

        return identity_filtered

identity_filtered = images_subsampling()


#Function for splitting the images into training set, validation set, and test set.
def images_split(identity_filtered, validation_size, test_size, export = False):

    #Extracting the images' names and their labels.
    imgs = identity_filtered['image']
    labels = identity_filtered['image_id']

    #Stratified split - in order to preserve the same labels' distribution across the samples.
    _, test_imgs, __, test_labels = train_test_split(imgs, labels,
                                               test_size = test_size,
                                               random_state = random_seed,        
                                               stratify = labels)

    train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(_, __,
                                               test_size = validation_size/(1-test_size),
                                               random_state = random_seed,        
                                               stratify = __)
    #Exporting the samples.
    if export:
        pd.concat((train_imgs, train_labels), axis = 1).to_csv('./csv/train_imgs_list.csv', index = False)
        pd.concat((valid_imgs, valid_labels), axis = 1).to_csv('./csv/valid_imgs_list.csv', index = False)
        pd.concat((test_imgs, test_labels), axis = 1).to_csv('./csv/test_imgs_list.csv', index = False)

    return train_imgs, train_labels, valid_imgs, valid_labels, test_imgs, test_labels

train_imgs, train_labels, valid_imgs, valid_labels, test_imgs, test_labels = images_split(identity_filtered, test_size, test_size)


#Function for generating balanced pairs of images (50% positive pairs, 50% negative pairs)
def pairs_generator(labels, image_names, seed, export_name = None):
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
    no_pairs_generated = 0

    #For each image, find its positive pair and negative pair
    for idx_a in range(len(image_names)):

        #Current anchor image - its label (person) and photo name
        label = labels[idx_a]
        current_image_name = image_names[idx_a]

        #Positive image - random image of the same person
        np.random.seed(seed)
        idx_b = np.random.choice(dict_idx[label])
        
        #Increment for chaning the random seed.
        delta_seed = 1

        #If the pair is existing in the list, select randomly another image again.
        #If the photos are the same (if the indices are the same), then again randomly select another image.

        #Creating a list of all posible positive pairs for given image.
        all_pos_combos = [sorted([current_image_name, image_names[b]]) for b in dict_idx[label]]

        #While loop as a constraint for generating unique positive pairs only (no duplicated such as [a,b] and [b,a])
            #Another constraint - such positive pair has to include the different images..
            #If there is not existing any combination of two positive images which is not included in the list of generated pairs - skip and proceed with the next image.
        while True:
            positive_image_name = image_names[idx_b] #Positive image name
            pair_names = [sorted(pair) for pair in pair_images_names] #List of pairs generated so far
            current_pos_pair = sorted([current_image_name, positive_image_name]) #Current generated positive pair

            #If the (1) current generated pair is not included in the list of all generated pairs so far and at the same time the positive pair includes 2 different photos or (2) there is no existing positive pair left - skip.
            if ((current_pos_pair not in pair_names) & (len(set(current_pos_pair)) != 1)) or (len(all_pos_combos) == 0):
                break
            
            #If is the positive pair is duplicated or has 2 same images or there are still some combinations left:
            else:
                #Change the random seed to sample different image name
                np.random.seed(seed + delta_seed) #Change the random seed to sample different image name
                idx_b = np.random.choice(dict_idx[label])

                delta_seed += 1 #Change the increment of the random seed in order to sample a different image name.
                try:
                    all_pos_combos.remove(current_pos_pair) #Remove the genereted pair from the list of all possible pairs.
                except ValueError:
                    continue
        
        #If the previous iteration has been break because there were not any combination of two images left - exit this iteration (do not look for the negative pair) and proceed with the next image.
            #In order to acheive balanced distribution of pairs (positive/negative).
        if (len(all_pos_combos) != 0):
            
            #Randomly sampling a different label (person) and its image name's index.
            negative_index = np.random.choice(dict_idx[np.random.choice([i for i in dict_idx.keys()
                                                                        if i != label])])

           #While loop as a constraint for generating unique positive pairs only (no duplicated such as [a,b] and [b,a]).
            while True:
                negative_image_name = image_names[negative_index] #Negative image name
                pair_names = [sorted(pair) for pair in pair_images_names] #List of pairs generated so far
                current_neg_pair = sorted([current_image_name, negative_image_name]) #Current generated negative pair
                
                 #If the negative pair is already existing in the list, select randomly another image again.
                if (current_neg_pair not in pair_names):
                    break
                
                #If is the negative pair:
                else:
                    #Change the random seed to sample different image name
                    np.random.seed(seed + delta_seed)
                    negative_index = np.random.choice(dict_idx[np.random.choice([i for i in dict_idx.keys()
                                                                                if i != label])])
                    delta_seed += 1

            #Appending the positive pair's images' names and its label.
            pair_images_names.append([current_image_name, positive_image_name])
            pair_labels.append([1])
            no_pairs_generated += 1
            
            #Appending the negative pair's images' names and its label.
            pair_images_names.append([current_image_name, negative_image_name])
            pair_labels.append([0])
            no_pairs_generated += 1

        #Print statement
            runtime = time() #Break point for measuring the run time of generating pairs with respect to the anchor image
            no_pairs = f'{no_pairs_generated}/{2*(len(image_names))}' #How many pairs have been created so far
            speed = (no_pairs_generated)/(runtime - start_time) * 60 #How many pairs have been created per minute on average
            eta = (2*(len(image_names)) - (no_pairs_generated)) / ((no_pairs_generated)/(runtime - start_time)) / 60 #Estimated remaining execution time

            print(f'{no_pairs} pairs created ... {speed:.2f} pairs/min | ETA: {eta:.2f} minutes   ', end = '\r')


    #Data frame storing all negative and positive pairs (with the image names) and their pair label (1 = positive | 0 = negative).
    final_df = pd.concat((pd.DataFrame((pair_images_names), columns = ['img_1', 'img_2']),
                            pd.DataFrame(pair_labels, columns = ['label'])),
                            axis = 1)

    #Exporting the generated pairs and their labels.
    if export_name != None:
        final_df.to_csv(f'./csv/{export_name}_pairs.csv', index  = False)
    
    #Print statement
    print('                                                                                                ', end = '\r') #Removing the previous statements

    #Final statement
    print(f'{no_pairs_generated} unique balanced pairs generated', '\n')
    print(f'Total Run Time: {(time() - start_time)/60:.2f} minutes', '\n')
    
    return final_df


train_pairs = pairs_generator(train_labels, train_imgs, 'train')
valid_pairs = pairs_generator(valid_labels, valid_imgs, 'valid')
test_pairs = pairs_generator(test_labels, test_imgs, 'test')


#Function for plotting an image and its positive and negative pair
def plot_pairs(pairs_df, base_img = None, resize = True):
        
        if base_img == None:
                base_img = np.random.choice(pairs_df['img_1'])
                
        #Filter pairs which include the baseline image
        filtered_df = pairs_df[pairs_df['img_1'] == base_img]

        #Filter a positive pair of given baseline image
        positive_img = filtered_df.query('label == 1')['img_2'].values[0]

        #Filter a negative pair of given baseline image
        negative_img = filtered_df.query('label == 0')['img_2'].values[0]

        #Folder path definition
        folder_path = 'cropped_images' if resize == True else 'data/Img/img_celeba'

        #Plot the baseline image
        fig = plt.figure(figsize=(15,15))
        ax = plt.subplot(131)
        ax.set_title(f"Anchor image - {base_img}")
        ax.imshow(cv2.cvtColor(cv2.imread(f'./{folder_path}/{base_img}'),
        cv2.COLOR_BGR2RGB))

        #Plot the positive image
        ax = plt.subplot(132)
        ax.set_title(f"Positive image - {positive_img}")
        ax.imshow(cv2.cvtColor(cv2.imread(f'./{folder_path}/{positive_img}'),
                  cv2.COLOR_BGR2RGB))

        #Plot the negative image
        ax = plt.subplot(133)
        ax.set_title(f"Negative image - {negative_img}")
        ax.imshow(cv2.cvtColor(cv2.imread(f'./{folder_path}/{negative_img}'),
                  cv2.COLOR_BGR2RGB))

        plt.tight_layout()
        plt.show()

plot_pairs(test_pairs, resize = False)