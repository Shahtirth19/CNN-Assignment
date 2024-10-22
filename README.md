# Project Name - Melanoma Skin Cancer Detection Using Convolutional Neural Networks (CNNs)

## Abstract

This project aims to build an automated skin cancer classification system using dermatoscopic images. The model achieved 84% training accuracy and 81% validation accuracy without batch normalization. Both training and validation loss decreased steadily, demonstrating effective learning. The minimal gap between training and validation accuracy indicates strong generalisation and low overfitting, making the model a reliable tool for early skin cancer detection.

## Problem statement
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

## Table of Contents
* [General Info](#general-information)
* [Model Architecture](#model-architecture)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

## General Information
This project utilises a Convolutional Neural Network (CNN) to accurately classify melanoma from dermatoscopic images, aiming to assist dermatologists in early skin cancer detection. The dataset is augmented to address class imbalance, using techniques such as image flipping and rotation to improve model generalisation. By leveraging deep learning, this system automates the classification process, potentially reducing diagnostic time and effort.

The dataset consists of 2357 images of malignant and benign oncological diseases, sourced from the International Skin Imaging Collaboration (ISIC). All images were categorized based on the ISIC classification, with balanced subsets for each condition, except for melanomas and moles, which are slightly dominant.

The data set contains the following diseases:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

In order to address the challenge of class imbalance, the Augmentor Python package [Augmentor](https://augmentor.readthedocs.io/en/master/) was employed to augment the dataset. This involved generating additional samples for all classes, ensuring that none of the classes had insufficient representation.

## Model Architecture
The CNN consists of multiple convolutional and pooling layers, followed by fully connected layers to predict skin cancer types. The model incorporates key techniques, such as:

- Data Augmentation: To increase dataset variety and reduce overfitting.
- Convolutional Layers (16, 32, 64 filters): For feature extraction from images.
- Max Pooling: To reduce feature map size while retaining critical information.
- Dropout (0.2): Applied to prevent overfitting.
- Dense Layers: Used for final classification into skin cancer types.
  
The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss function to handle multi-class classification. Callbacks such as EarlyStopping and ModelCheckpoint ensure optimal model training without overfitting.

## Conclusions
- The model successfully achieved 84% training accuracy and 81% validation accuracy, indicating effective learning.
- The small gap between training and validation accuracy highlights strong generalisation, with minimal overfitting.
- Data augmentation significantly improved model performance, enhancing its robustness in predictions.

## Technologies Used
* Python 3.11.5
* Pandas - 2.0.3
* Numpy - 1.24.3
* Matplotlib - 3.7.2
* Seaborn - 0.12.2
* Tensorflow - version 2.15.0
* Google colab

## Acknowledgements
* References
  - Upgrad Course - https//learn.upgrad.com


## Contact
Created by [@Shahtirth19](https://github.com/Shahtirth19)
