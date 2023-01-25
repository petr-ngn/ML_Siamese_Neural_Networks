import os
import pandas as pd
import numpy as np
from time import time
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tensorflow.keras.models import Model
from tensorflow.keras.backend import epsilon
from tensorflow.keras.optimizers import RMSprop
from tensorflow.math import square, maximum, reduce_mean, sqrt, reduce_sum
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, Lambda
from keras.callbacks import EarlyStopping, TensorBoard
import seaborn as sns

#Function for reading the images' names, labels and bounding boxes
def reading_crop_inputs(identity_file = './data/Anno/identity_CelebA.txt', bbox_file = './data/Anno/list_bbox_celeba.txt'):

    #Loading the image names (image) and labels (image_id)
    identity = pd.read_csv(identity_file, sep = " ", header = None,
                        names = ['image', 'image_id'])

    #Loading the bounding boxes of images
    bbox = pd.read_csv(bbox_file, delim_whitespace = True)

    return identity, bbox



#Function for cropping all the images
def face_crop_export(id_df, bbox, crop_all = False):

    #Function for cropping, resizing and exporting a single image in .jpg format.
    def face_crop(image_name, bbox):

        #Loading Image
        image_path = './data/Img/img_celeba/' + image_name
        img = cv2.imread(image_path)

        #Setting bounding box coordinates
        startX = bbox[bbox['image_id'] == image_name]['x_1'].values[0]
        startY = bbox[bbox['image_id'] == image_name]['y_1'].values[0]
        endX = startX + bbox[bbox['image_id'] == image_name]['width'].values[0]
        endY = startY + bbox[bbox['image_id'] == image_name]['height'].values[0]
    
        #Cropping
        crop_img = img[startY:endY, startX:endX]
        output_img = crop_img
            
        #Resizing
        output_img = cv2.resize(crop_img, (224, 224))

        #Exporting the cropped image
        cv2.imwrite(f'./cropped_images/{image_name}', output_img)

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
        
            print(f"{no_imgs} images cropped ... {speed:.2f} images/min | ETA: {eta:.2f} minutes                 ", end = '\r')

        #Skip if the image has no applicable bounding boxes.    
        except cv2.error as e:
            k += 1
            continue



#Function for reading and subsampling the images.
def images_subsampling(identity_file = './data/Anno/identity_CelebA.txt', atts_file =  "./data/Anno/list_attr_celeba.txt", bad_imgs_file = 'final_bad_imgs.csv'):

        #Reading the images' names and labels.
        identity = pd.read_csv(identity_file, sep = " ", header = None,
                            names = ['image', 'image_id'])

        #Reading a csv list of manually picked bad images
        bad_imgs = pd.read_csv(bad_imgs_file)

        print(f'Original number of images: {identity.shape[0]}')

        #Dropping the images which could not be cropped and the bad images.
        imgs_to_drop = list(set(identity['image'].tolist()).difference(os.listdir("./cropped_images/")))
        identity = identity[~identity['image'].isin(imgs_to_drop)]
        identity = identity[~identity['image'].isin(bad_imgs['image'].tolist())]

        print(f"Number of images after the 1st exclusion: {identity.shape[0]}")

        #Reading attributes file
        atts = pd.read_csv(atts_file, delim_whitespace = True).reset_index().rename(columns = {'index':'image'})

        #Filtering images based on subsampling on identity file and replace -1 with 0
        atts = atts[atts['image'].isin(identity['image'])].replace(-1, 0)

        #Exclude images which fullill at least on the following images (in order to exclude noisy images)
        exclude_imgs = (atts['w5_o_Clock_Shadow'] == 1) | (atts['Wearing_Hat'] == 1) | (atts['Eyeglasses'] == 1) | (atts['Smiling'] == 1) | (atts['Narrow_Eyes'] == 1) | (atts['Pale_Skin'] == 1) | (atts['Blurry'] == 1) | (atts['Mouth_Slightly_Open'] == 1)
        
        #filtered images based o excluding images which meet at least of the above mentioned conditions
        atts_filtered = atts[~exclude_imgs]
        identity_filtered = identity[~exclude_imgs]

        print(f"Number of images after the 2nd exclusion: {identity_filtered.shape[0]}")

        #If a celebrity has more photos where he/she was young than photos where he/she was old, the "old" photos were excluded (and vice versa)
        young_old_ids = identity_filtered.merge(atts_filtered[['image','Young']], on = 'image').groupby('image_id')['Young'].mean().reset_index().rename(columns = {'Young':'Young_photo'})
        young_old_ids['Young_photo'] = [0 if i <= 0.5 else 1 for i in young_old_ids['Young_photo']]
        young_old_filter = identity_filtered.merge(young_old_ids[['image_id','Young_photo']], on ='image_id').merge(atts[['image','Young']], on ='image').query('Young_photo == Young')['image'].tolist()

        atts_filtered = atts_filtered[atts_filtered['image'].isin(young_old_filter)]
        identity_filtered = identity[identity['image'].isin(young_old_filter)]

        print(f"Number of images after the 3rd exclusion: {identity_filtered.shape[0]}")

        #If a celebrity has more photos where he/she had grey hair color than photos where he/she had not, the photos with no grey hair color were ecluded (and vice versa)
        gray_hair_ids = identity_filtered.merge(atts_filtered[['image','Gray_Hair']], on = 'image').groupby('image_id')['Gray_Hair'].mean().reset_index().rename(columns = {'Gray_Hair':'Gray_Hair_photo'})
        gray_hair_ids['Gray_Hair_photo'] = [0 if i <= 0.5 else 1 for i in gray_hair_ids['Gray_Hair_photo']]
        gray_hair_filter = identity_filtered.merge(gray_hair_ids[['image_id','Gray_Hair_photo']], on ='image_id').merge(atts[['image','Gray_Hair']], on ='image').query('Gray_Hair_photo == Gray_Hair')['image'].tolist()

        atts_filtered = atts_filtered[atts_filtered['image'].isin(gray_hair_filter)]
        identity_filtered = identity[identity['image'].isin(gray_hair_filter)]

        print(f"Number of images after the 4th exclusion: {identity_filtered.shape[0]}")

        #Final filter - choose only such images whose classes (celebrities) had at least 5 images.
        final_imgs = identity_filtered['image_id'].value_counts().reset_index().rename(columns = {'image_id':'count','index':'image_id'}).query('count >= 5')['image_id']
        final_identity = identity_filtered[identity_filtered['image_id'].isin(final_imgs)]
        final_atts = atts_filtered[atts_filtered['image'].isin(final_identity['image'])]

        print(f"Final number of images after all the exclusions: {final_identity.shape[0]}")
        
        return final_identity, final_atts



#Function for generating balanced pairs of images (50% positive pairs, 50% negative pairs)
def pairs_generator(identity_file, atts_df, seed, target_number, exclude_imgs_list, export_name = None):

    #Start time initialization (for execution time measurement)
    start_time = time()

    #List for storing the pair labels (1 = positive pair | 0 = negative pair)
    pair_labels = []

    #List for storing the pair image names.
    pair_images_names = []

    #If a list of images for exclusion is defined, then the images included in such list will not be considered within generating pairs.
      #This is relevant when generating pairs for validation set, thus the images within training pairs will not be considered in such case.
      #This is relevant when generating pairs for test set, thus the images within training and validation pairs will not be considered in such case.

    if len(exclude_imgs_list) !=0: #case for generating training pairs
        id_df = identity_file[~identity_file['image'].isin(exclude_imgs_list)]
        atts = atts_df[~atts_df['image'].isin(exclude_imgs_list)]
        labels = id_df['image_id'].reset_index(drop = True)
        image_names = id_df['image'].reset_index(drop = True)

    else: #case for generating validation or test pairs
        atts = atts_df.copy()
        labels = identity_file['image_id'].reset_index(drop = True)
        image_names = atts_df['image'].reset_index(drop = True)

    #Array with all the labels
    unique_classes = np.unique(labels)

    #Dictionary for storing images' indices for each label.
    dict_idx = {i:np.where(labels == i)[0] for i in unique_classes}
    
    #Separating indices of male and female images.
    male_indices = pd.DataFrame(image_names).merge(atts[['image', 'Male']], on ='image').query('Male == 1').index.to_list()
    female_indices = pd.DataFrame(image_names).merge(atts[['image', 'Male']], on ='image').query('Male == 0').index.to_list()

    #Random shuffle of those indices
    np.random.seed(seed)
    np.random.shuffle(male_indices)
    np.random.shuffle(female_indices)
                  
    #Count of pairs initialization
    no_pairs_generated = 0

    #count initialization of generated pairs (with male image as an anchor)
    i_male = 0
    #count initialization of generated pairs (with female image as an anchor)
    i_female = 0

    #Increment for changing the random seed.
    delta_seed = 1

    #For each image, find its positive pair and negative pair
    for _ in range(len(image_names)):
        
        #The first half of the pairs will have male images as anchor
        if no_pairs_generated < target_number/2:
            current_img_list = male_indices #male images list for selecting an anchor
            idx_a = current_img_list[i_male] #male anchor

            i_male += 1 #increase an male index in order to choose another male anchor within the next iteration
    
        #The second half will have female images as anchor.
        else:
            current_img_list = female_indices #female images list for selecting an anchor
            idx_a = current_img_list[i_female] #female anchor

            i_female += 1 #increase an female index in order to choose another female anchor within the next iteration

        #Current anchor image - its label (person) and photo name
        label = labels[idx_a]
        current_image_name = image_names[idx_a]

        #Positive image - random image of the same person
        np.random.seed(seed + delta_seed)
        idx_b = np.random.choice(dict_idx[label])

        #Creating a list of all posible positive pairs for given image.
        all_pos_combos = [sorted([current_image_name, image_names[b]]) for b in dict_idx[label]]

        #Generating positive pair
        while True:
            positive_image_name = image_names[idx_b] #Positive image name
            pair_names = [sorted(pair) for pair in pair_images_names] #List of pairs generated so far
            current_pos_pair = sorted([current_image_name, positive_image_name]) #Current generated positive pair

             #In order to "minimize" the difference between the images, the person should have the same hair color in both images
              #Check whether the the person has the same color hair on both images - he/she should have (hence we would expect the value to be 0)
            color_hair_indicator = (atts.loc[atts['image'] == current_image_name,['Blond_Hair', 'Black_Hair','Brown_Hair']].reset_index(drop = True) !=\
                                    atts.loc[atts['image'] == positive_image_name,['Blond_Hair', 'Black_Hair','Brown_Hair']].reset_index(drop = True)).sum().sum()

            #In order to "minimize" the difference between the images, the person should have either (no) make up on both photos.
              #Check whether the the person has make up on both images or not - either he/she would have make up on both images or no make up on both images (hence we would expect the value to be 1 (or higher than 0))
            makeup_indicator = (atts.loc[atts['image'] == current_image_name, ['Heavy_Makeup']].reset_index(drop = True) ==\
                                atts.loc[atts['image'] == positive_image_name, ['Heavy_Makeup']].reset_index(drop = True)).sum().sum()

            #In order to "minimize" the difference between the images, the person should have either (no) facial hair on both photos.
              #Check whether the the person has facial hair on both images or not - the images attributes should match at least at 2 of the 3 attributes (No_Beard, Mustache, Goatee) -  (hence we would expect the value to be 2 (or higher than 1))
            facial_hair_indicator = (atts.loc[atts['image'] == current_image_name, ['No_Beard', 'Mustache', 'Goatee']].reset_index(drop = True) ==\
                                    atts.loc[atts['image'] == positive_image_name, ['No_Beard', 'Mustache', 'Goatee']].reset_index(drop = True)).sum().sum()
            
            #If:
            # (1) current generated pair is not included in the list of all generated pairs so far and at the same time, the positive pair includes 2 different photos (not a pair one identical photo), and
            # (2) has the same hair color, and
            # (3) do/doesn't have make up on both images, and
            # (4) do/doesn't have facial hair on both images, or
            # (5) there is at least one positive pair combination left (which was not generated before) out of all possible positive pair combinations (with respect to the anchor)
                  #It could happen that there are no other possible positive pair combinations left for given anchor. Meaning all the possible positive pair combinations were already generated before.
            # Then exit the while loop and proceed next.
            if ((current_pos_pair not in pair_names) & (len(set(current_pos_pair)) != 1)) & (color_hair_indicator == 0) & (makeup_indicator > 0) & (facial_hair_indicator > 1) or (len(all_pos_combos) == 0):
                break
            
            #If one of the conditions mentioned above is not met, then try to search for another positive pair combination (that's why, we increase a random seed in order to choose different photo)
            else:
                #Change the random seed to sample different image name
                np.random.seed(seed + delta_seed) #Change the random seed to sample different image name
                idx_b = np.random.choice(dict_idx[label]) #New photo index for a positive pair

                delta_seed += 1 #Change the increment of the random seed in order to sample a different image name in the next iteration

                #If one of the conditions was not met, remove such positive pair combination from list of all the possible positive pair combinations
                try:
                    all_pos_combos.remove(current_pos_pair) #Remove the genereted pair from the list of all possible pairs.
                except ValueError: #if there are no other possible positive pair combinations left for given anchor, then continue with a next iteration
                    continue
        

        #If the previous iteration has been broken because there were not any combination of two images left - exit this iteration (do not look for the negative pair) and proceed with the next anchor image.
            #In order to acheive balanced distribution of pairs (positive/negative).
        #If the previous iteration has not been broken because there was at least on combination of two images left - then continue with looking for the negative image with respect to the current anchor.

        if (len(all_pos_combos) != 0):
            
            #Randomly sampling a different label (person) and its image name's index.
            np.random.seed(seed + delta_seed)
            negative_index = np.random.choice(dict_idx[np.random.choice([i for i in dict_idx.keys()
                                                                        if i != label])])

            #Generating negative pair
            while True:
                negative_image_name = image_names[negative_index] #Negative image name
                pair_names = [sorted(pair) for pair in pair_images_names] #List of pairs generated so far
                current_neg_pair = sorted([current_image_name, negative_image_name]) #Current generated negative pair

                #In order to "maximimize" the difference between the images, we look for such negative image which has an opposite gender with respect to the anchor
                  #if the anchor is female, then we look for a male image and vice versa.
                genders = [atts.loc[atts['image'] == current_image_name,'Male'].values[0],
                               atts.loc[atts['image'] == negative_image_name,'Male'].values[0]]

                #In order to "maximimize" the difference between the images, the celebrities should have different hair color
                  #Check whether the the persons have the same color hair on the images - they should not have (hence we would expect the value to be 1 (or higher than 0))
                color_hair_indicator = (atts.loc[atts['image'] == current_image_name,['Blond_Hair', 'Black_Hair','Brown_Hair']].reset_index(drop = True) !=\
                                        atts.loc[atts['image'] == negative_image_name,['Blond_Hair', 'Black_Hair','Brown_Hair']].reset_index(drop = True)).sum().sum()
                
                #If:
                # (1) current generated pair is not included in the list of all generated pairs so far and at the same time, and
                # (2) both images are of opposite gender (pair Male-Female or pair Female-Male)
                # (3) both images have different hair color 
                # Then exit the while loop and proceed with the next anchor.
                if (current_neg_pair not in pair_names) and (genders[0] != genders[1]) and (color_hair_indicator > 0):
                    break
                
                #If one of the conditions mentioned above is not met, then try to search for another negative pair combination
                else:
                    #Change the random seed to sample different image name
                    np.random.seed(seed + delta_seed)
                    negative_index = np.random.choice(dict_idx[np.random.choice([i for i in dict_idx.keys()
                                                                                if i != label])])
                    delta_seed += 1

              #Note - we did not set a constraint regarding of all possible negative combinations.
                #since we assume that there is a low probability that such situation could arise, as there are a lot of combinations possible (because we can sample from different labels and photos as well)

            #Appending the positive pair's images' names and its label.
            pair_images_names.append([current_image_name, positive_image_name])
            pair_labels.append([1])
            no_pairs_generated += 1
            
            #Appending the negative pair's images' names and its label.
            pair_images_names.append([current_image_name, negative_image_name])
            pair_labels.append([0])
            no_pairs_generated += 1

        #Print statement
            no_pairs = f'{no_pairs_generated}/{target_number}' #How many pairs have been created so far
            print(f'{no_pairs} pairs created', end = '\r')
            
            if no_pairs_generated >= target_number:
                break

    #Data frame storing all negative and positive pairs (with the image names) and their pair label (1 = positive | 0 = negative).
    final_df = pd.concat((pd.DataFrame((pair_images_names), columns = ['img_1', 'img_2']),
                            pd.DataFrame(pair_labels, columns = ['label'])),
                            axis = 1)

    #Shuffle the indices/order of the generated pairs
    final_df = final_df.sample(frac = 1,
                                random_state = seed).sample(frac = 1,
                                random_state = seed)

    #Exporting the generated pairs and their labels.
    if export_name != None:
        final_df.to_csv(f'./csv/{export_name}_pairs.csv', index  = False)
    
    #Print statement
    print('                                                                                                 ', end = '\r') #Removing the previous statements

    #Final statement
    print(f'{no_pairs_generated} unique balanced pairs generated', '\n')
    print(f'Total Run Time: {(time() - start_time)/60:.2f} minutes', '\n')
    
    return final_df



#Function for checking descriptive information about the generated pairs in given sample
def pairs_check(pairs_df, atts):
    
    #Accessing unique labels [0, 1] and their frequencies
    labels = pairs_df['label'].replace(1, 'Positive').replace(0, 'Negative').value_counts().index
    freqs = pairs_df['label'].value_counts().values

    #Print the label distribution
    print(f"Label distribution ... {labels[0]}: {freqs[0]} ({freqs[0]/sum(freqs)*100:.0f}%) | {labels[1]}: {freqs[1]} ({freqs[1]/sum(freqs)*100:.0f}%)")

    #Acessing the unique pairs from given sample
    unique_pairs = set([str(sorted([i,j])) for i,j in zip(pairs_df['img_1'], pairs_df['img_2'])])

    #Check whether the number of unique pairs match the number of generated pairs in withn sample
    if len(unique_pairs):
        print(f'Number of unique pairs ... {len(unique_pairs)}')
    else:
        print(f"Number of unique pars doesn't match the number of pairs in given sample ({len(unique_pairs)} vs {pairs_df.shape[0]})")

    #Print whether the are any pairs which contains a single picture only
    print(f"Number of pairs containing the same image ... {(pairs_df['img_1'] == pairs_df['img_2']).sum()}")

    #Print number of images included in the generated pair sample
    num_unique_imgs = len(list(set(pairs_df['img_1'].tolist() + pairs_df['img_2'].tolist())))
    print(f'Number of images ... {num_unique_imgs}')

    #Print the distribution with respect to the gender - it should be also balanced as well
    gender = pairs_df.merge(atts[['image','Male']].replace(1, 'Male').replace(0, 'Female'), left_on ='img_1', right_on = 'image')['Male'].value_counts().index
    gender_freqs = pairs_df.merge(atts[['image','Male']].replace(1, 'Male').replace(0, 'Female'), left_on ='img_1', right_on = 'image')['Male'].value_counts().values

    print(f"Gender distribution ... {gender[0]}: {gender_freqs[0]} ({gender_freqs[0]/sum(freqs)*100:.0f}%) | {gender[1]}: {gender_freqs[1]} ({gender_freqs[1]/sum(gender_freqs)*100:.0f}%)")



#Function for plotting an image and its positive and negative pair
def plot_pairs(pairs_df, base_img = None, resize = True):
        
        #if an anchor image name is not provided, it will randomly choose an anchor from provided pair sample
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

        #Plot the anchor image
        fig = plt.figure(figsize=(15,15))
        ax = plt.subplot(131)
        ax.set_title(f"Anchor image - {base_img}")
        ax.imshow(cv2.cvtColor(cv2.imread(f'./{folder_path}/{base_img}'),
                  cv2.COLOR_BGR2RGB))
        ax.set_axis_off()

        #Plot the positive image
        ax = plt.subplot(132)
        ax.set_title(f"Positive image - {positive_img}")
        ax.imshow(cv2.cvtColor(cv2.imread(f'./{folder_path}/{positive_img}'),
                  cv2.COLOR_BGR2RGB))
        ax.set_axis_off()

        #Plot the negative image
        ax = plt.subplot(133)
        ax.set_title(f"Negative image - {negative_img}")
        ax.imshow(cv2.cvtColor(cv2.imread(f'./{folder_path}/{negative_img}'),
                  cv2.COLOR_BGR2RGB))
        ax.set_axis_off()

        plt.tight_layout()
        plt.show()



#Function for reading pairs and load either as data frame or as arrays.
def read_pairs(sample_name, separate = True):
    
    final_df = pd.read_csv(f'./csv/{sample_name}_pairs.csv')

    #If separate, then output the pairs and labels separately as numpy arrays
    if separate:
        for col in ['img_1', 'img_2']:
            final_df[col] =  [f'./cropped_images/{i}'for i in final_df[col]]

        imgs = final_df[['img_1', 'img_2']]
        labels = final_df[['label']]

        return np.array(imgs), np.array(labels)
    
    #Otherwise, output the pairs and labels together in a data frame
    else:
        return final_df



#Function for processing pairs of images
def tf_img_pipeline(anchor, comparison):
    
    #Function for processing an image (reading, decoding, resizing and tensor conversion)
    def tf_img_processing(img_path):
        img = tf.io.read_file(img_path) #reading
        img = tf.image.decode_jpeg(img, channels = 3) #jpg decoding
        img = tf.image.resize(img, [224,224], method = 'bilinear') #resizing
        img = tf.image.convert_image_dtype(img, tf.float32) /  tf.constant(255, dtype = tf.float32) #normalization

        return img

    return tf_img_processing(anchor), tf_img_processing(comparison) #API processing of both anchor and comparison image



#Function for processing labels
def tf_label_pipeline(label):
    return tf.cast(label, tf.float32) #just a conversion to TF tensors



#Function for tensorflow dataset creation - input for modelling
def tf_data_processing_pipeline(images, labels):

    images_tf = tf.data.Dataset.from_tensor_slices((images[:, 0] , images[:, 1])).map(tf_img_pipeline) #API processing of both anchor and comparison images
    labels_tf = tf.data.Dataset.from_tensor_slices(labels).map(tf_label_pipeline) #API processing of labels

    dataset = tf.data.Dataset.zip((images_tf,
                                    labels_tf)).batch(16,
                                                      num_parallel_calls = AUTOTUNE).cache().prefetch(buffer_size = AUTOTUNE) #API processing of TF dataset creation with 16 batch size, cache and prefetch
    return dataset


#Function for calculating the Euclidean distance between the two vectors
def euclidean_distance(vects):

    # unpack the vectors into separate lists
    x, y = vects

    # compute the sum of squared distances between the vectors
    sum_square = tf.math.reduce_sum(tf.math.square(x - y),
                                    axis = -1, keepdims = True)
    
    # return the euclidean distance between the vectors
    return tf.math.sqrt(tf.math.maximum(sum_square,
                                        tf.keras.backend.epsilon()))



#Contrastive loss function
def contrastive_loss(margin = 1):

    #D - predicted distances (feature vectors)
    #Y - True labels (0 or 1)

    def contrastive__loss(Y, D):

        return tf.math.reduce_mean(
            Y * tf.math.square(D) + (1 - Y) * tf.math.square(tf.math.maximum(margin - D, 0))
                                  )

    return contrastive__loss



#Function for hyperparameter tuning input
def model_building(hp):

  #Input layer
  inputs = Input(shape = (224, 224, 3))
  x = inputs

  #Tuning a number of convolutional blocks
  for i in range(hp.Int('conv_blocks', min_value = 1, max_value = 5,
                        default = 3)):
    
    #Normalization layer
    x = BatchNormalization()(x)

    #Tuning the number of convolution's output filters
    filters = hp.Int(f'filters_{i}', min_value = 32,
                     max_value = 256, step = 32)

    #Tuning the activation function within convolution
    activations = hp.Choice(f'activation_{i}',
                            ['relu','tanh'])

    #Tuning the kernel size within convolution
    kernel_sizes = hp.Choice(f'kernelsize_{i}',
                             [j for j in range(1,11)])

    #Convolution layer
    x = Conv2D(filters, kernel_size= (kernel_sizes,kernel_sizes),
                 activation = activations)(x)

    #Tuning the pool size within pooling
    pool_sizes = hp.Choice(f'poolsizes_{i}',
                           [j for j in range(1,9)])

    #Tuning the strides within pooling
    strides_pool = hp.Choice(f'strides_pool_{i}',
                             [j for j in range(1,6)])

    #Tuning the pooling type in the convolutional block
    if hp.Choice(f'pooling_{i}', ['avg', 'max']) == 'max':

        x = MaxPooling2D(pool_size = (pool_sizes,pool_sizes),
                         strides = (strides_pool,strides_pool))(x)
    else:
      
        x = AveragePooling2D(pool_size = (pool_sizes,pool_sizes),
                         strides = (strides_pool,strides_pool))(x)

  #Global Average Pooling / Flatten/ Normalization layers
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)
  x = BatchNormalization()(x)

  #Tuning the number of units and activation function in feature vector layer
  feature_layer = Dense(hp.Int('Dense_units_0', min_value = 10,
                                max_value = 360, step = 10, default = 50),
                                activation=  hp.Choice(f'activation_{0}',
                                         ['relu','tanh']),
                       name = 'feature_layer')(x)

  #Embedding Convolutional neural network model
  embedding_network = Model(inputs, feature_layer, name = 'CNN')

  #Setting an input layer for the image pairs
  input_1 = Input((224, 224, 3), name = 'left_tower')
  input_2 = Input((224, 224, 3), name = 'right_tower')

  tower_1 = embedding_network(input_1)
  tower_2 = embedding_network(input_2)

  #Layers for calculation of the Euclidean distance between the two feature vectors
  merge_layer = Lambda(euclidean_distance,
                       name = 'lambda_layer')([tower_1,
                                              tower_2])
  
  #Normalization layer
  normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)


  #Final output layer (classification whether the images are of the same label/person)
  output_layer = Dense(1, activation = "sigmoid",
                         name = 'output_layer')(normal_layer)

  #Final Siamese neural networks model
  model = Model(inputs=[input_1, input_2], outputs = output_layer, name = 'SNN')

  #Model compilation with root mean square propagation optimizer (RMSProp), contrastive loss and accuracy
  model.compile(optimizer = RMSprop(hp.Float('learning_rate',
                                            min_value = 1e-4, max_value = 0.5,
                                            sampling = 'log')),
                
                  loss = contrastive_loss(hp.Float('margin',
                                            min_value = 0.1, max_value = 1.5,
                                            sampling = 'log')))

  return model


#Function for plotting the validation and training loss
def plot_val_train_loss(history_model, export = True):
  
  plt.figure(figsize = (12, 10))

  plt.plot(history_model.history['loss'])
  plt.plot(history_model.history['val_loss'])

  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.legend(['train', 'validation'])

  plt.grid()
  plt.tight_layout()

  if export:
    plt.savefig('validation_train_loss.png')

  plt.show()



#Function for processing of a single part of a images' pair (just left or right)
def tf_single_prep(images):

    def _tf_img_pipeline_(pic):
    
      def _tf_img_processing_(img_path):
          img = tf.io.read_file(img_path)
          img = tf.image.decode_jpeg(img, channels = 3)
          img = tf.image.resize(img, [224,224], method = 'bilinear')
          img = tf.image.convert_image_dtype(img, tf.float32) /  tf.constant(255, dtype = tf.float32)

          return img

      return _tf_img_processing_(pic)

    images_tf = tf.data.Dataset.from_tensor_slices(images).map(_tf_img_pipeline_)

    dataset = images_tf.batch(16,num_parallel_calls = AUTOTUNE).cache().prefetch(buffer_size = AUTOTUNE)

    return dataset



#Function for calculation of predicted feature vector
def distance_derivation(imgs_, CNN_model):

  #TF API processing of an images
  tf_ = tf_single_prep(imgs_)

  #Predict the distance/extract the predicted feature vector based on the processed images and trained model
  feat_vecs_ = CNN_model.predict(tf_)

  return feat_vecs_



#Function for calculation of the optimal threshold for a classification
def cutoff_derivation(imgs, labels, CNN_model):

  #Calculation of positive feature vectors and negative feature vectors
  def pos_neg_distances(imgs_, labels_, CNN_model):
    tf_ = tf_single_prep(imgs_) #TF API processing of an images
    feat_vecs_ = CNN_model.predict(tf_) #Predict the distance/extract the predicted feature vectors based on the processed images and trained model
    pos_feat_vecs_ = feat_vecs_[[i[0] for i in np.argwhere(labels_  == [1])]] #Filter such feature vectors which correspond to the positive pairs
    neg_feat_vecs_ = feat_vecs_[[i[0] for i in np.argwhere(labels_  == [0])]] #Filter such feature vectors which correspond to the negative pairs

    return (pos_feat_vecs_, neg_feat_vecs_) #return positive feature vectors and negative feature vectors

  #Extract the positive and negative feature vectors for anchor images
  left_pos_feat_vecs, left_neg_feat_vecs = pos_neg_distances(imgs[:,0], labels, CNN_model)

  #Extract the positive and negative feature vectors for comparison images
  right_pos_feat_vecs, right_neg_feat_vecs = pos_neg_distances(imgs[:,1], labels, CNN_model)

  #Calculate the average of the Euclidean distances between the anchor's and comparison's positive feature vectors
  pos_dis_mean = np.mean(euclidean_distance((left_pos_feat_vecs, right_pos_feat_vecs)).numpy().flatten())

  #Calculate the average of the Euclidean distances between the anchor's and comparison's negative feature vectors
  neg_dis_mean = np.mean(euclidean_distance((left_neg_feat_vecs, right_neg_feat_vecs)).numpy().flatten())

  #Calculate the cut-off based on the computed averages
  cutoff = (pos_dis_mean + neg_dis_mean)/2

  return cutoff



#Function for computing the accuracy and returning the predicted feature vectors (distances)
def compute_accuracy(y_true, left_feat_vecs, right_feat_vecs, cutoff):

  #First take the continuous distance predictions based on Frobenius/Euclidean vector norm of the anchor's and comparison's feature vectors
  pred_distances = np.linalg.norm(left_feat_vecs - right_feat_vecs, axis=1)

  #Classify the predicted class based on the cut-off and the continuous distance predictions
    #Positive pair (1) - if distance < cut-off
    #Negative pair (0) - if distance >= cut-off
  pred_classes = pred_distances.flatten() < cutoff

  #Return:
  # (1) accuracy
  # (2) predicted classes (0 or 1)
  # (3) predicted continuous distances

  return np.mean(pred_classes == y_true), pred_classes, pred_distances



#Function for ploting single images
def plot_single_images(photo_names = None, photo_path = None, photos_dict = None):

  fig, axs = plt.subplots(nrows = 1,ncols = 5, figsize = (15, 30))

  #Plot the photos based on provided image names and their paths.
  if (photo_names != None) & (photo_path != None) & (photos_dict == None):

    for name, ax in zip(photo_names, axs.ravel()):
        ax.imshow(cv2.cvtColor(cv2.imread(photo_path + name), cv2.COLOR_BGR2RGB))
        ax.set_title(name)
        ax.set_axis_off()

  #Plot the photos based on provided dictionary (key = photo name; value = numpy array with photo's pixels)
  elif (photo_names == None) & (photo_path == None) & (photos_dict != None):

    for photo_item, ax in zip(photos_dict.items(), axs.ravel()):
      ax.imshow(cv2.cvtColor(photo_item[1], cv2.COLOR_BGR2RGB))
      ax.set_title(photo_item[0])
      ax.set_axis_off()

  plt.tight_layout()
  plt.show()



#Function for plotting the pairs
def plot_pairs_live_demo(photo_names_1, photo_names_2, photo_path_1, photo_path_2, no_rows, no_cols, figure_size):

  fig, axs = plt.subplots(nrows = no_rows, ncols = no_cols, figsize = figure_size)
  col_ind=0
  axis_count = 0

  for ax in axs.ravel():
      
      #Left images
      if axis_count % 2 == 0:
          ax.imshow(cv2.cvtColor(cv2.imread([f'{photo_path_1}{i}' for i in photo_names_1][col_ind]), cv2.COLOR_BGR2RGB))
          ax.set_title([f'{photo_path_1}{i}' for i in photo_names_1][col_ind])
          ax.set_axis_off()
      
      #Right images
      else:
          ax.imshow(cv2.cvtColor(cv2.imread([f'{photo_path_2}{i}' for i in photo_names_2][col_ind]), cv2.COLOR_BGR2RGB))
          ax.set_title([f'{photo_path_2}{i}' for i in photo_names_2][col_ind])
          ax.set_axis_off()

          col_ind += 1
        
      axis_count +=1
    

  plt.tight_layout()
  plt.show()



#Function for detection of face with subsequent cropping based on detected bouding boxes
def cropping_engine(path, photo_name_list):

    def bbox_engine_img_input(path, img_name, m1_scale_factor = 1.1, m1_min_neighbors = 13):
    
        img = cv2.imread(path + img_name)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #we use Cascade Classifier for generating bounding boxes
        faces = face_cascade.detectMultiScale(image = img, scaleFactor = m1_scale_factor, minNeighbors = m1_min_neighbors)

        #If the face on an image cannot be detected nor cropped (len(face) == 0) regardless the m1_min_neighbors hyperparameter value, then exit the loop - there is not chance that the image can be cropped
        while len(faces) < 1:
            m1_min_neighbors -= 1
            faces = face_cascade.detectMultiScale(image = img, scaleFactor = m1_scale_factor, minNeighbors = m1_min_neighbors)
            if m1_min_neighbors < 0:
                break
    
        if len(faces) > 0:

            #Coordinates of a bounding box
            bbox = {"x_1" : faces[0][0],
                    "y_1" : faces[0][1],
                    "width" : faces[0][2],
                    "height" : faces[0][3],
                    'x_end' : faces[0][0] + faces[0][2],
                    'y_end' : faces[0][1] + faces[0][3]}

            return bbox

    #Dictionary for storing cropped images based on generated bounding boxes
    photos_dict = {}

    bbox_generated = pd.DataFrame(columns= ['image_id', 'x_1', 'y_1', 'width', 'height', 'x_end', 'y_end'])

    for photo_name in photo_name_list:

        bbox_coordinates = bbox_engine_img_input(path, photo_name)
        bbox_coordinates['image_id'] = photo_name

        bbox_generated = bbox_generated.append(bbox_coordinates, ignore_index = True)

        startX = bbox_generated[bbox_generated['image_id'] == photo_name]['x_1'].values[0]
        startY = bbox_generated[bbox_generated['image_id'] == photo_name]['y_1'].values[0]
        endX = bbox_generated[bbox_generated['image_id'] == photo_name]['x_end'].values[0]
        endY = bbox_generated[bbox_generated['image_id'] == photo_name]['y_end'].values[0]

        img =  cv2.imread(path+ photo_name)
        crop_img = cv2.resize(img[startY:endY, startX:endX], (224, 224))

        photos_dict[photo_name] = crop_img

    return photos_dict



#Function for predicting a person class based on a minimum of Euclidean distances between the on-site image(s) and reference images
def live_demo_preds(ref_feat_vecs, onsite_feat_vecs):

  #Dictionaries for storing the predicted Euclidean distances and predicted team members' names.
  euclidean_distance_dict = {}
  predicted_pairs_dict = {}

  #For each on-site photo, calculate the Euclidan distance with each reference photo.
  for ons in onsite_feat_vecs.columns:

    #Dictionary for storing all the computed Euclidean distances of current on-site photo with respect to the all reference photos
    euc_dis_ons = {}

    for ref in ref_feat_vecs.columns:
      euc_dis = euclidean_distance((onsite_feat_vecs[ons].values,
                                    ref_feat_vecs[ref].values)).numpy().flatten()
      euc_dis_ons[ref] = euc_dis[0]

    #Assign all the computed Euclidean distances to the respective on-site photo key.
    euclidean_distance_dict[ons] = euc_dis_ons

  #For each on-site photo, select the reference photo name which has the smallest Euclidean distance.
  for ons in euclidean_distance_dict.keys():

    pred_person = min(euclidean_distance_dict[ons],
                     key = euclidean_distance_dict[ons].get)

    predicted_pairs_dict[ons] = pred_person

  #Return both dictionary of predicted pairs and dictionary of predicted Euclidean distances
  return predicted_pairs_dict, euclidean_distance_dict



#Function for plotting the predicted pairs
def plot_predicted_pairs(predict_pairs):

  fig, axs = plt.subplots(5, 1, figsize=(25, 15))

  for ax, pair in zip(axs.ravel(), predict_pairs.items()):
    
    #Joined the on-site photo with the predicted reference photo into one image
    ax.imshow(tf.concat([cv2.cvtColor(cv2.imread(f'./cropped_live_demo/{pair[0]}'), 
                                      cv2.COLOR_BGR2RGB),
                         cv2.cvtColor(cv2.imread(f'./cropped_live_demo/{pair[1]}'),
                                      cv2.COLOR_BGR2RGB)
                        ],axis = 1))
  
    ax.set_title(f'On-site person: {pair[0].split("_")[0]} \n Predicted (reference) person: {pair[1].split("_")[0]}',
               size = 14)
    ax.set_axis_off()
  
  plt.tight_layout()
  plt.show()