import os
import numpy as np
import pandas as pd
from utils import *
from pathlib import Path
import argparse
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras.applications import resnet

if __name__ in "__main__":

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="IML Task 4"
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Whether to train the model or not."
    )

    args = parser.parse_args()
    train_bool = args.train

    import zipfile
    with zipfile.ZipFile('./food.zip', 'r') as zip_ref: 
        zip_ref.extractall()
        
    folder_path='./food'

    images=load_images_from_folder(folder_path)
    image_count=len(images)

    target_shape = (200,200)

    #loading the images 

    train_file = open("./train_triplets.txt", "r")
    train = np.loadtxt(train_file, dtype=str)
    test_file = open("./test_triplets.txt", "r")
    test = np.loadtxt(test_file)

    if train_bool == 'y':
        #getting file paths for the training dataset
        anchor_train, positive_train, negative_train = get_image_path(train, folder_path)

        #defining the datasets for the training datasets
        anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_train)
        positive_dataset = tf.data.Dataset.from_tensor_slices(positive_train)
        negative_dataset = tf.data.Dataset.from_tensor_slices(negative_train)

        dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
        dataset = dataset.shuffle(buffer_size=1024)
        dataset = dataset.map(preprocess_triplets)

        # Let's now split our dataset in train and validation.
        train_dataset = dataset.take(round(image_count * 0.8))
        val_dataset = dataset.skip(round(image_count * 0.8))

        train_dataset = train_dataset.batch(32, drop_remainder=False)
        train_dataset = train_dataset.prefetch(8)

        val_dataset = val_dataset.batch(32, drop_remainder=False)
        val_dataset = val_dataset.prefetch(8)

        base_cnn = resnet.ResNet50(weights="imagenet", input_shape=target_shape + (3,), include_top=False)

        flatten = layers.Flatten()(base_cnn.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        output = layers.Dense(256)(dense2)

        embedding = Model(base_cnn.input, output, name="Embedding")

        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable


        anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
        positive_input = layers.Input(name="positive", shape=target_shape + (3,))
        negative_input = layers.Input(name="negative", shape=target_shape + (3,))

        distances = DistanceLayer()(
            embedding(resnet.preprocess_input(anchor_input)),
            embedding(resnet.preprocess_input(positive_input)),
            embedding(resnet.preprocess_input(negative_input)),
        )

        siamese_network = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
        siamese_model = SiameseModel(siamese_network)
        siamese_model.compile(optimizer=optimizers.Adam(0.0001))
        siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

        siamese_model.save("model_v1")

        anchor_test, positive_test, negative_test = get_image_path(test)

        #defining the datasets for the training datasets
        anchor_dataset_test = tf.data.Dataset.from_tensor_slices(anchor_test)
        positive_dataset_test = tf.data.Dataset.from_tensor_slices(positive_test)
        negative_dataset_test = tf.data.Dataset.from_tensor_slices(negative_test)

        dataset_test = tf.data.Dataset.zip((anchor_dataset_test, positive_dataset_test, negative_dataset_test))
        dataset_test = dataset_test.shuffle(buffer_size=1024)
        dataset_test = dataset_test.map(preprocess_triplets)

        dataset_test = dataset_test.batch(32, drop_remainder=False)
        dataset_test = dataset_test.prefetch(8)

        pred = siamese_model.predict_classes(dataset_test)

        submission = pd.DataFrame(pred)
        submission.to_csv('./submission_hugo.csv', index=False, header=False)

    else:
        siamese_model = models.load_model('model_v1')

        anchor_test, positive_test, negative_test = get_image_path(test)

        #defining the datasets for the training datasets
        anchor_dataset_test = tf.data.Dataset.from_tensor_slices(anchor_test)
        positive_dataset_test = tf.data.Dataset.from_tensor_slices(positive_test)
        negative_dataset_test = tf.data.Dataset.from_tensor_slices(negative_test)

        dataset_test = tf.data.Dataset.zip((anchor_dataset_test, positive_dataset_test, negative_dataset_test))
        dataset_test = dataset_test.shuffle(buffer_size=1024)
        dataset_test = dataset_test.map(preprocess_triplets)

        dataset_test = dataset_test.batch(32, drop_remainder=False)
        dataset_test = dataset_test.prefetch(8)

        pred = siamese_model.predict_classes(dataset_test)

        submission = pd.DataFrame(pred)
        submission.to_csv('./submission_hugo.csv', index=False, header=False)