
# for loading/processing the images  
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.applications.xception import preprocess_input 

# models 
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
import tensorflow as tf

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random



seed = 123
random.seed(seed)
np.random.seed(seed)

path = '/Users/darialaslo/Documents/CBB/CBB SEM2/IML/GIT/task-4/food'
os.chdir(path)

# this list holds all the image filename
food = []

#add image files to the list
with os.scandir(path) as files:
  # loops through each file in the directory
    for file in files:
        if file.name.endswith('.jpg'):
          # adds only the image files to the flowers list
            food.append(file.name)


def preprocess_images(file):
    # load the image as a 299x299 array
    img = load_img(file, target_size=(299,299))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,299,299,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    return imgx

os.chdir("/Users/darialaslo/Documents/CBB/CBB SEM2/IML/GIT/task-4")

#importing the pickle file containing the dictionary with the embeddings

file_to_read = open("./xception.pkl", "rb")
data = pickle.load(file_to_read)


# get a list of the filenames
filenames = np.array(list(data.keys()))

# reshape so that there are 10000 samples of 2048 features
feat = np.array(list(data.values()))
feat = feat.reshape(-1,2048)
feat.shape


#trying to set up embedding

input_features = layers.Input(name="features", shape = ( 2048))

dense1 = layers.Dense(512, activation="relu")(input_features)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(inputs=[input_features], outputs=[output], name="Embedding")


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape =  (2048))
positive_input = layers.Input(name="positive", shape = (2048))
negative_input = layers.Input(name="negative", shape = (2048))

distances = DistanceLayer()(
    embedding(anchor_input), 
    embedding(positive_input), 
    embedding(negative_input)
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)
        

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]



#function for getting anchor, pos and neg 
def get_image_path(file):
    
    anchor_list=[]
    positive_list=[]
    negative_list=[]
    for i in range(file.shape[0]):
        anchor_name= "{}.jpg".format(file[i,0])
        positive_name= "{}.jpg".format(file[i,1])
        negative_name= "{}.jpg".format(file[i,2])
        anchor_list.append(anchor_name)
        positive_list.append(positive_name)
        negative_list.append(negative_name)
    
    return anchor_list, positive_list, negative_list


#if using another neural net after extracting fetures, the opposite triplets and labels are needed

# use the get image name to know which one is which
train_file = open("./train_triplets.txt", "r")
train = np.loadtxt(train_file, dtype=str)

anchor_list, positive_list, negative_list = get_image_path(train)

#getting validation dataset for train dataset

train_prop = 0.8
train_split = int(np.floor(train.shape[0] * train_prop))

anchor_list_val = anchor_list[train_split:]
positive_list_val = positive_list[train_split:]
negative_list_val = negative_list[train_split:]

#getting train data for trial
anchor_list=anchor_list[:train_split]
positive_list=positive_list[:train_split]
negative_list= negative_list[:train_split:]



def preprocess_features(a_list, p_list, n_list):
  anchors=[]
  positives=[]
  negatives=[]
  for filename in a_list:
      anchors.append(np.asarray(list(data[filename])))
  for filename in p_list:
      positives.append(np.asarray(list(data[filename])))
  for filename in n_list:
      negatives.append(np.asarray(list(data[filename])))
  anchors = np.asarray(anchors)
  anchors = anchors.reshape(anchors.shape[0], 2048)
  positives = np.asarray(positives)
  positives = positives.reshape(positives.shape[0], 2048)
  negatives = np.asarray(negatives)
  negatives = negatives.reshape(negatives.shape[0], 2048)

  return anchors, positives, negatives


#getting the inputs
anchors , positives, negatives = preprocess_features(anchor_list, positive_list, negative_list)

#getting validation dataset

anchors_val, positives_val, negatives_val = preprocess_features(anchor_list_val, positive_list_val, negative_list_val)



siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.001))
siamese_model.fit([anchors, positives, negatives], epochs=12, validation_data=[anchors_val, positives_val, negatives_val])


# use the get image name to know which one is which, for test set
test_file = open("./test_triplets.txt", "r")
test = np.loadtxt(test_file, dtype=str)

anchor_list_test, positive_list_test, negative_list_test = get_image_path(test)

#getting the inputs
anchors_test , positives_test, negatives_test = preprocess_features(anchor_list_test, positive_list_test, negative_list_test)

pred=siamese_model.predict([anchors_test, positives_test, negatives_test])


#getting which one is more likely
first = np.asarray(list(pred[0]))
second = np.asarray(list(pred[1]))

final_predictions=[]

for i in range(0, len(first)):
  if first[i]<second[i]:
    final_predictions.append(1)
  else:
    final_predictions.append(0)


submission=pd.DataFrame(final_predictions)
submission.to_csv("./submission_test.csv", index=False, header=False)