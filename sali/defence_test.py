import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import sys
import numpy as np

# Disable TensorFlow eager execution:
import tensorflow as tf

if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

from keras.models import load_model

from art import config
from art.utils import load_dataset, get_file
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import BasicIterativeMethod
from art.defences.trainer import AdversarialTrainer
from art.defences.trainer import AdversarialTrainerFBF

(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')

path = get_file('mnist_cnn_original.h5', extract=False, path=config.ART_DATA_PATH,
                url='https://www.dropbox.com/s/p2nyzne9chcerid/mnist_cnn_original.h5?dl=1')
classifier_model = load_model(path)
classifier = KerasClassifier(clip_values=(min_, max_), model=classifier_model, use_logits=False)

trainer = AdversarialTrainerFBF(classifier_model)
print(trainer)