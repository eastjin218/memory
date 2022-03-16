import os, warnings, sys
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow_hub as hub
import tensorflow_addons as tfa

import mlflow
import mlflow.tensorflow

TRAIN_DATASET_PATH = '/Users/lee/Documents/leedj/workplace/globit_ob_reco/fish_siamese_network/dataset/selected/train'
TEST_DATASET_PATH = '/Users/lee/Documents/leedj/workplace/globit_ob_reco/fish_siamese_network/dataset/selected/test'
BATCH_SIZE = 32
def get_generator():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range =360,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=.2,
        zoom_range=.2,
        brightness_range=[0.75,1.25],
        # validation_split=0.2,
    )
    return datagen

def ref_generator():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        # validation_split=0.2
    )
    return datagen

def get_label(train_data):
    class_names = list(train_data.class_indices.keys())
    return class_names

def triplet_pair_generator(anchor_gen, gen):
    while True:
        anchors, y_anc = next(anchor_gen)

        pos = np.empty(anchors.shape)
        neg = np.empty(anchors.shape)

        y_pos = np.empty(y_anc.shape)
        y_neg = np.empty(y_anc.shape)

        pos_assigned = np.zeros(anchors.shape[0])
        neg_assigned = np.zeros(anchors.shape[0])

        while not np.all(pos_assigned) or not np.all(neg_assigned):
            img, y = next(gen)

            for sample_idx in range(anchors.shape[0]):
                positive_idxs = np.argwhere(y==y_anc[sample_idx]).flatten()
                negative_idxs= np.argwhere(y!=y_anc[sample_idx]).flatten()

                if positive_idxs.size > 0 and pos_assigned[sample_idx] ==0:
                    positive_idx = np.random.choice(positive_idxs, 1)[0]
                    pos[sample_idx] = img[positive_idx]
                    y_pos[sample_idx]=y[positive_idx]
                    pos_assigned[sample_idx] =1

                if negative_idxs.size >0 and neg_assigned[sample_idx] == 0:
                    negative_idx = np.random.choice(negative_idxs, 1)[0]
                    neg[sample_idx] = img[negative_idx]
                    y_neg[sample_idx] = y[negative_idx]
                    neg_assigned[sample_idx] = 1
        
        concatenate_img = np.concatenate([anchors, pos, neg], axis = 0)
        concatenate_y = np.concatenate([y_anc, y_pos, y_neg], axis=0).astype(np.int32)

        yield concatenate_img, concatenate_y

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hub_layer = hub.KerasLayer(
            'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/feature_vector/2',trainable=False
        )
        self.Dense_1_layer = tf.keras.layers.Dense(512, activation='relu')
        self.Dense_2_layer = tf.keras.layers.Dense(256, activation='relu')
        self.Dense_3_layer = tf.keras.layers.Dense(128, activation='relu')
        self.Lamda_layer = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    
    def call(self, input_tensors):
        x = self.hub_layer(input_tensors)
        x = self.Dense_1_layer(x)
        x = self.Dense_2_layer(x)
        x = self.Dense_3_layer(x)
        result =self.Lamda_layer(x)
        return result

    
def main():
    mlflow.tensorflow.autolog()
    with mlflow.start_run():
        datagen = get_generator()
        ref_datagen = ref_generator()

        train_anchor = datagen.flow_from_directory(
            TRAIN_DATASET_PATH,
            target_size=(384,384),
            batch_size=BATCH_SIZE,
            class_mode='sparse'
        )
        test_anchor = datagen.flow_from_directory(
            TEST_DATASET_PATH,
            target_size=(384,384),
            batch_size=BATCH_SIZE,
            class_mode='sparse'
        )

        train_ref = ref_datagen.flow_from_directory(
            TRAIN_DATASET_PATH,
            target_size=(384,384),
            batch_size=BATCH_SIZE*2,
            class_mode='sparse',
            subset='training'
        )
        test_ref = ref_datagen.flow_from_directory(
            TEST_DATASET_PATH,
            target_size=(384,384),
            batch_size=BATCH_SIZE*2,
            class_mode='sparse',
        )

        labels = get_label(train_anchor)

        train_triplet = triplet_pair_generator(train_anchor, train_ref)
        test_triplet = triplet_pair_generator(test_anchor, test_ref)

        model = MyModel()
        model.build([None, 384,384,3])
        model.summary()

        model.compile(
            optimizer= tf.keras.optimizers.Adam(0.001),
            loss=tfa.losses.TripletSemiHardLoss()
        )
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            './model/siames_efficientnetv2_best.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_freq='epoch',
            save_weights_only=True
        )

        history = model.fit(
            train_triplet,
            steps_per_epoch=train_anchor.samples // BATCH_SIZE,
            validation_data= test_triplet,
            validation_steps = train_anchor.samples // BATCH_SIZE,
            epochs =5,
            callbacks=[model_checkpoint_callback]
        )

    model.save_weights('./model/fish_siamese_effnetv2s.h5')
if __name__=="__main__":
    main()