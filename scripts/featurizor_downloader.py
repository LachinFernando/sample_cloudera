from firebase import Firebase
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model

from keras.layers import Dense,GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax, Ftrl, Nadam, RMSprop, SGD
import pandas as pd

class Image_Featurizor_and_Uploader():

  def __init__(self, path, config_file, number_of_classes = 2):
    """
    This class creates the featurized csv which will be used to calculate the neighbor images
    and while doing that process, images will be uploaded to the firebase.

    To run this file, app/serviceAccount.json must be in your default environment.

    path: (str) -> Path of the images used in training file
    config_file: (dict) -> Firebase configuration details. (Attched below for reference)
    number_of_classes: (int) -> Number of prediction classes

    example config_file:

    config = {
        "apiKey": "AIzaSyDvKxjPAxyfb8RcGxn7bIRff4eH2xlmohE",
        "authDomain": "test2-72fd2.firebaseapp.com",
        "projectId": "test2-72fd2",
        "storageBucket": "test2-72fd2.appspot.com",
        "messagingSenderId": "851610909928",
        "appId": "1:851610909928:web:5087fb5397bf4f4d521d5d",
        "measurementId": "G-ZD52Q3E3G2",
        "serviceAccount":"serviceAccount.json",
        "databaseURL": "https://test2-72fd2-default-rtdb.firebaseio.com/"
    }
    
    
    """
    self.path = path
    self.config_file = config_file
    self.classes = number_of_classes
  
  def __create_dataframe(self, number_of_features, features, labels ):

    print("Generating the dataframe")
    
    feature_names = []
    for a in range(0,number_of_features):
      feature_names.append('feature_' + str(a))
    feature_names.append('label')
    df = pd.DataFrame(data=np.hstack((features,labels)), columns=feature_names)
    return df


  def __convert_tf_dataset(self, PATH, model, firebase_storage):
    print("Converting the images into features")
    # This function passes all images provided in PATH
    # and passes them through the model.
    # The result is a featurized image along with labels
    data = []
    IMG_SIZE = (224, 224)
    file_list = []
    
    # Get the list of subfolders
    sub_dirs = next(os.walk(PATH))[1]
    print(sub_dirs)
    num_images = 0

    # Create a list of lists
    # Number of lists is same as the number of subfolders
    # Number of items in the sub-list is the number of
    # images in each sub-folder
    for category in sub_dirs:
        files = next(os.walk(PATH + '/' + category), (None, None, []))[2]
        filenames = [PATH + '/' + category + '/' + file for file in files] 
        num_images += len(filenames)
        file_list.append(filenames)
    
    labels = []
    count_pos = 0
    count_neg = 0
    # Every image is pre-processed and passed thought the model
    # Label is created for every image
    for category in file_list:

        for img_path in category:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            data.append(model.predict(img_preprocessed))
            labels.append(img_path.split('/')[-2])
            label = img_path.split('/')[-2]
            if label == "Negative":
              firebase_storage.child(f"Images/Negative/{count_neg}.jpg").put(img_path)
              count_neg +=1
              print("Count neg", count_neg)
              if count_neg > 100:
                break
            else:
              firebase_storage.child(f"Images/Positive/{count_pos}.jpg").put(img_path)
              count_pos += 1
              print("Count pos",count_pos)
              if count_pos > 100:
                break
              

    # Make sure dimensions are (num_samples, 1280)
    data = np.squeeze(np.array(data))
    labels = np.reshape(labels, (-1,1))
    return data, labels
  
  def featurizer_and_uploader(self, save_path):

    """
    This method save the csv file while uploading the images to the firebase.

    save_path: (str) -> path to save the csv fiel

    return: pd.DataFrame
    
    """

    firebase = Firebase(self.config_file)
    storage = firebase.storage()

    IMG_SIZE = (224, 224)


    # Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
    base_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # Add average pooling to the base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input,outputs=x)

    # Get the transformed features from the dataset
    # TODO: This can be moved to the FE stage of the pipeline
    # label_map is not used anywhere right now. it has information
    # about which label is mapped to which number
    data, labels = self.__convert_tf_dataset(self.path, model_frozen, storage)
    num_classes = self.classes
    # Shuffle the dataset for training
    shuffler = np.random.permutation(len(data))
    data_shuffled = data[shuffler]
    labels_shuffled = labels[shuffler]

    num_features = data_shuffled.shape[1]

    dataframe = self.__create_dataframe(num_features, data_shuffled, labels_shuffled)
    save_path = save_path + "/featureized_data.csv"
    dataframe.to_csv(save_path, index = False)
    print("CSV is saved to the {}".format(save_path))
    return dataframe

if __name__ == "__main__":
    PATH = 'path of the training images'

    #your firebase credentials
    config = {
        "apiKey": "AIzaSyDvKxjPAxyfb8RcGxn7bIRff4eH2xlmohE",
        "authDomain": "test2-72fd2.firebaseapp.com",
        "projectId": "test2-72fd2",
        "storageBucket": "test2-72fd2.appspot.com",
        "messagingSenderId": "851610909928",
        "appId": "1:851610909928:web:5087fb5397bf4f4d521d5d",
        "measurementId": "G-ZD52Q3E3G2",
        "serviceAccount":"serviceAccount.json",
        "databaseURL": "https://test2-72fd2-default-rtdb.firebaseio.com/"
    }

    test = Image_Featurizor_and_Uploader(path = PATH, config_file = config)
    #to upload the images to the firbase and save the csv
    path_csv = "Path to save the csv"
    test.featurizer_and_uploader(path_csv)