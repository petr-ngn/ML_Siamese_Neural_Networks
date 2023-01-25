# Siamese Convolutional Neural Networks

_**Author:**_ [**Petr Nguyen**](https://www.linkedin.com/in/petr-ngn) (_**Lead Data Scientist**_)

_**Other members of ML team:**_ [**Daniel Malinovsky**](https://www.linkedin.com/in/daniel-malinovsky-88b162198) (_**Team Leader**_), [Petr Hollmann](https://www.linkedin.com/in/petr-hollmann-3583aa208) (_Regular member_), [Roman Pavlata](https://www.linkedin.com/in/roman-pavlata-a3b602161) (_Regular member_), [Natalie Musilova](https://www.linkedin.com/in/natálie-musilová-3b98287a) (_Regular member_)

Within the course __*Agile Project of Machine Learning Applications (4IZ481)*__ at Faculty of Informatics and Statistics, Prague University of Economics and Business, our task was to develop a Machine Lerning model for face image recognition and detection, using Tensorflow and Keras.

This course was surpervised by the Data Scientists, AI & ML Engineers and Scrum Masters at [**CN Group CZ**](https://www.linkedin.com/company/cngroup-dk), namely by [Petr Polak](https://www.linkedin.com/in/87petrpolak), [Patrik Tison](https://www.linkedin.com/in/patriktison), Viktor Stepanuk, Marek Hronek and [Tomas Kliment](https://www.linkedin.com/in/tomáš-kliment-b74120196).

Particularly, I have developed a Siamese Convolutional Neural Networks with Contrastive Loss, which was further tuned with Bayesian Optimization while minimizing a loss function on validation set, and was built on the generated balanced pairs of images from the cleaned and processed [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Such model had an accuracy over 70% on the test set.



This repository contains following notebooks:
1) [Cropping](https://github.com/petr-ngn/ML_Siamese_Neural_Networks/blob/main/01_Cropping.ipynb) - this notebook focuses on a cropping of images' faces based on the provided bounding boxes' coordinates.
2) [Balancing pairs](https://github.com/petr-ngn/ML_Siamese_Neural_Networks/blob/main/02_Balanced_pairs.ipynb) - this notebook focuses on subsampling the CelebA dataset based on several constraints defined within custom functions and further, it generates balanced pairs of images (50% of positive pairs = two photos of the same person, and 50% of negative pairs = two photos of two different persons) for training, validation and test set as an input for modelling.
3) [Modelling](https://github.com/petr-ngn/ML_Siamese_Neural_Networks/blob/main/03_Modelling.ipynb) - this notebook focuses on processing the images' pairs into TensorFlow dataset, on a model building of Siamese Convolutional Neural Networks with Contrastive Loss, whose hyperparameters are tuned with Bayesian Optimization. Last but not least, it depicts an evaluation of the built model based on a classification cut-off point derived from the predicted feature vectors of distances.
4) [Live demo](https://github.com/petr-ngn/ML_Siamese_Neural_Networks/blob/main/04_Live_demo.ipynb) - this notebook focuses on a live demo using the team member's reference photos and on-site photos. The goal is to compute the feature vectors of the reference photos and then the feature vectors of the on-site photos, both using the built model. Then for each on-site photo, it predicts a person class (particular team member) by comparing the feature vectors to the reference photos' feature vectors and looking for such reference photo which results in the lowest Euclidean distance.
