import cv2
import tensorflow as tf
import os, glob

from sali_tensor import SaliencyMapTf


f_path = glob.glob('./images/0_*')
input_img = cv2.imread(f_path[1])
input_img = tf.convert_to_tensor(input_img, dtype='float32')
sali_tf = SaliencyMapTf()
preprocess_img = sali_tf.preprocess(input_img)
with tf.GradientTape() as t:
    t.watch(preprocess_img)
    result = sali_tf.compute_saliency(preprocess_img)
dz_dx = t.gradient(result, preprocess_img)