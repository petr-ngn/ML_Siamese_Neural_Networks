import os
import pandas as pd
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tensorflow.keras.models import Model
from tensorflow.keras.backend import epsilon
from tensorflow.keras.optimizers import Adam
from tensorflow.math import square, maximum, reduce_mean, sqrt, reduce_sum
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda



#Function for reading the images' names, labels and bounding boxes
def reading_crop_inputs(identity_file = './data/Anno/identity_CelebA.txt', bbox_file = './data/Anno/list_bbox_celeba.txt'):
    #Loading the image names (image) and labels (image_id)
    identity = pd.read_csv(identity_file, sep = " ", header = None,
                        names = ['image', 'image_id'])
    #Loading the bounding boxes of images
    bbox = pd.read_csv(bbox_file, delim_whitespace = True)

    return identity, bbox



#Function for cropping, resizing and exporting a single image in .jpg format.
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



#Function for cropping all the images
def face_crop_export(id_df, crop_all = False):

    #Start time initialization (for execution time measurement)
    start_time = time()

    #List of images to crop (whether to crop all the images in the identity file or only those images which haven't been cropped yet)
    imgs_to_crop = id_df['image'] if crop_all == True else list(set(id_df['image'].tolist()).difference(os.listdir("./cropped_images/")))

    #Initialization for counting number of images which cannot be cropped.
    k = 0

    #Crop each image
    for ind, img in enumerate(imgs_to_crop):
        try:
            #Cropping image
            face_crop(img, bbox)

            #Print statement
            no_imgs = f'{ind + 1 - k}/{len(imgs_to_crop)}' #How many images have been cropped so far
            speed = (ind + 1- k)/(time() - start_time) * 60 #How many images have been cropped per minute on average
            eta = (len(imgs_to_crop) - (ind + 1 - k)) / ((ind + 1 - k)/(time() - start_time)) / 60 #Estimated remaining execution time
        
            print(f"{no_imgs} images cropped ... {speed:.2f} images/min | ETA: {eta:.2f} minutes", end = '\r')

        #Skip if the image has no applicable bounding boxes.    
        except cv2.error as e:
            print(f"Image {img} has no applicable bounding boxes.")
            k += 1
            continue



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



#Function for reading the generated balanced pairs and their labels saved in .csv format and outputing them as two separate numpy arrays
def read_pairs(sample_name):
    
    final_df = pd.read_csv(f'./csv/{sample_name}_pairs.csv')

    #Attaching a path folder name to each image.
    for col in ['img_1', 'img_2']:
        final_df[col] =  [f'./cropped_images/{i}'for i in final_df[col]]

    #Extracting separately the pairs of images' names and the labels
    imgs = final_df[['img_1', 'img_2']]
    labels = final_df[['label']]

    return np.array(imgs), np.array(labels)



#Function for processing the both anchor image (left image) and its comaprison image (right image)
def tf_img_pipeline(anchor, comparison):
    
    #Function for processing the the image (reading, decoding, resizing and converting to tensors)
    def tf_img_processing(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels = 3)
        img = tf.image.resize(img, [224,224], method = 'bilinear')
        img = tf.image.convert_image_dtype(img, tf.float32)

        return img

    return tf_img_processing(anchor), tf_img_processing(comparison)



#Function for converting the labels from numpy arrays to tensors
def tf_label_pipeline(label):
    return tf.cast(label, tf.float32)



#Function for creating a pipeline for both processing images and labels and generating a TensorFlow dataset as an input for modelling
def tf_data_processing_pipeline(images, labels):

    images_tf = tf.data.Dataset.from_tensor_slices((images[:, 0] , images[:, 1])).map(tf_img_pipeline)
    labels_tf = tf.data.Dataset.from_tensor_slices(labels).map(tf_label_pipeline)

    dataset = tf.data.Dataset.zip((images_tf,
                                    labels_tf)).batch(64,
                                                      num_parallel_calls = AUTOTUNE).cache().prefetch(buffer_size = AUTOTUNE)
    return dataset



#Function for calculation an Euclidean distance between the two feature vectors
def euclidean_distance(vectors):

    x, y = vectors
    sum_square = reduce_sum(square(x - y), axis = 1, keepdims = True)

    return sqrt(maximum(sum_square, epsilon()))



#Function for a calculation of a contrastive loss
def contrastive_loss(margin = 1):

    def contrastive__loss(y_true, y_pred):

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive__loss



#Function for building the Siamese Networks model within a hyperparameter optimization.
def model_building(hp):

    #Input layer
    inputs = Input(shape = (224, 224, 3))
    x = inputs

    #Tuning a number of convolutional blocks
    for i in range(hp.Int('conv_blocks', min_value = 2, max_value = 5, default = 3)):
    
        #Tuning the number of convolution's output filters
        filters = hp.Int('filters_' + str(i), min_value = 32,
                        max_value = 1000, step = 32) 

        #Within each block, perform 2 convolutions and batch normalization
        for _ in range(2):

        #Tuning the number of convolution's output filters
            x = Conv2D(filters, kernel_size=(3, 3), padding = 'same',
                 activation = 'relu')(x)
            x = BatchNormalization()(x)

        #Tuning the pooling type in the convolutional block
        if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
            x = MaxPooling2D()(x)
        else:
            x = AveragePooling2D()(x)
    
    #Tuning the dropout rate in the dropout layer in the convolutional block
        x = Dropout((hp.Float('dropout', 0, 0.5, step = 0.05, default = 0.5)), seed = 123)(x)

    #Flatten the output
    x = Flatten()(x)

    #Tuning the number of units in the dense layer
    x = Dense(hp.Int('Dense units' ,min_value = 50,
                   max_value = 100, step = 10, default = 50),
                  activation='relu')(x)

    #Tuning the dropout rate in the dropout layer - the final feature vector layer
    feature_layer = Dropout((hp.Float('dropout', 0, 0.5, step = 0.05, default = 0.5)), seed = 123)(x)

    #Mapping a embedding model
    embedding_network = Model(inputs, feature_layer)
    
    #Setting an input layer for the image pairs
    input_1 = Input((224, 224, 3))
    input_2 = Input((224, 224, 3))
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    #Layers for calculation of the Euclidean distance between the two feature vectors, with further normalization
    merge_layer = Lambda(euclidean_distance)([tower_1, tower_2])
    normal_layer = BatchNormalization()(merge_layer)

    #Final output layer (classification whether the images are of the same label/person)
    output_layer = Dense(1, activation="sigmoid")(normal_layer)

    #Final model mapping
    model = Model(inputs=[input_1, input_2], outputs = output_layer)

    #Model compilation:
        #Tuning the learning rate of the stochastic gradient method in the Adam optimizer.
        #Minimizing a binary cross entropy loss function and maximizing an accuracy.
        #We compute the binary cross entropy for each label separately and then sum them up for the complete loss.
        
    model.compile(optimizer = Adam(hp.Float('learning_rate', min_value = 1e-4,
                                                            max_value = 1e-2,
                                                            sampling = 'log')), 
                    loss = contrastive_loss(margin = 1))

    return model
