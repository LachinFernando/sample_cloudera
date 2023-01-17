class Image_classifier():

  def __init__(self, path, epochs, learning_rate, classes):
    """
    This will intialize the main parameters need to train an Image classifier.
    
    path: (str) -> Path of the images
    epochs: (int) -> Number of epochs
    learning_rate: (float) -> Learning rate which will be used in the training
    classes: (int) -> Number of classes

    """
    
    self.path = path
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.classes = classes

  def __create_model(self,base_model,num_classes):
    import tensorflow as tf
    from keras.layers import Dense,GlobalAveragePooling2D
    from keras.models import Model
    # Grab the last layer and add a few extra layers to it
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    # Dense layer 1
    x=tf.keras.layers.Dense(100,activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(), use_bias=True)(x)

    # Final layer with softmax activation
    preds=tf.keras.layers.Dense(num_classes,activation='softmax', kernel_initializer=tf.keras.initializers.VarianceScaling(), use_bias=False)(x) 
    
    # Create the final model
    model=Model(inputs=base_model.input,outputs=preds)
    return model
  
  def __get_optimizer(self,optimizer_name, learning_rate):
    # Import keras optimizers
    from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax, Ftrl, Nadam, RMSprop, SGD
    print('Selected Optimizer', optimizer_name)
    switcher = {
        'Adadelta': Adadelta(learning_rate=learning_rate),
        'Adagrad': Adagrad(learning_rate=learning_rate),
        'Adam': Adam(learning_rate=learning_rate),
        'Adamax': Adamax(learning_rate=learning_rate),
        'FTRL': Ftrl(learning_rate=learning_rate),
        'NAdam': Nadam(learning_rate=learning_rate),
        'RMSprop': RMSprop(learning_rate=learning_rate),
        'Gradient Descent': SGD(learning_rate=learning_rate)
    }
    # If optimizer_name is empty, Adam will be return as default optimizer
    return switcher.get(optimizer_name, Adam(learning_rate=learning_rate))

  def train_model(self, model_save_path):
    
    # Import packages needed to create a imaage classification model
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import tensorflow as tf

    from keras.applications.resnet import preprocess_input
    from keras.preprocessing.image import ImageDataGenerator
    from keras.layers import Dense,GlobalAveragePooling2D
    from keras.models import Model
    from tensorflow.keras import regularizers

    from tensorflow.keras.preprocessing import image_dataset_from_directory
    from keras.callbacks import EarlyStopping
    from tensorflow import keras

    # Initialize hyper params
    epochs = self.epochs 
    base_learning_rate = self.learning_rate 
    optimizer = 'Adam'
    BATCH_SIZE = 32

    IMG_SIZE = (224, 224)

    # Create the data generation pipeline for training and validation
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2) # set validation split

    train_generator = train_datagen.flow_from_directory(self.path,
                                                    target_size=IMG_SIZE,
                                                    color_mode='rgb',
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    subset = 'training')
    validation_generator = train_datagen.flow_from_directory(self.path,
                                                    target_size=IMG_SIZE,
                                                    color_mode='rgb',
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    subset = 'validation')

    # Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet', alpha=0.35)
    for layer in base_model.layers:
        layer.trainable=False

    # Specify the number of classes
    num_classes = self.classes

    # Create the base model
    model = self.__create_model(base_model,num_classes)

    print(len(base_model.layers))

    model.compile(optimizer = self.__get_optimizer(optimizer_name=optimizer,learning_rate=base_learning_rate),loss='CategoricalCrossentropy',metrics=['accuracy'])
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy

    early_stopping_monitor = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=30,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    print("Epochs chosen: {} & Learning_rate chosen: {}".format(self.epochs, self.learning_rate))
    step_size_train = train_generator.n//train_generator.batch_size
    self.history_fine = model.fit(train_generator,
                            epochs=epochs,
                            callbacks=[early_stopping_monitor],
                            validation_data = validation_generator,
                            verbose=1)
    
    #saving the model
    model_name = "/final_model_epochs:{}_lr:{}.h5".format(self.epochs, self.learning_rate)
    model_saving_path = model_save_path + model_name
    model.save(model_saving_path)
    print("Model is saved to {}".format(model_saving_path))

    #visualizing the performance
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame(self.history_fine.history)
    #loss plots
    plt.figure(figsize=(8,8))
    plt.plot(df['loss'], color='red', label = "Training_loss")
    plt.plot(df['val_loss'], color='blue')
    plt.legend(['Training Loss','Validation loss'],loc = 'best' )
    plt.title('Line plot of Training and Validation loss')
    plt.ylim(0,1)
    plt.show()

    #accuracy plots
    plt.figure(figsize=(8,8))
    plt.plot(df['accuracy'], color='red')
    plt.plot(df['val_accuracy'], color='blue')
    plt.legend(['Training acc','Validation acc'],loc = 'best' )
    plt.title('Line plot of Training and Validation Accuracies')
    plt.ylim(0,1)
    plt.show()
