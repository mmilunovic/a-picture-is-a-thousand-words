import tensorflow as tf
from tensorflow import keras
import numpy as np
L = keras.layers
K = keras.backend

from utils_package import utils

import zipfile

import os


IMG_SIZE = 299

def get_cnn_encoder():
    K.set_learning_phase(False)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input

    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model


def get_train_features():
    train_img_embeds = utils.read_pickle("./models/train_img_embeds.pickle")
    train_img_fns = utils.read_pickle("./models/train_img_fns.pickle")

    print("Train features shape: ")
    print(train_img_embeds.shape, len(train_img_fns))

    return train_img_embeds, train_img_fns


def get_val_features():
    val_img_embeds = utils.read_pickle("./models/val_img_embeds.pickle")
    val_img_fns = utils.read_pickle("./models/val_img_fns.pickle")

    print("Validation features shape: ")
    print(val_img_embeds.shape, len(val_img_fns))

    return val_img_embeds, val_img_fns



def extract_features():
    encoder, preprocess_for_model = get_cnn_encoder()
 
    # extract train features
    train_img_embeds, train_img_fns = utils.apply_model(
        "./train_data/train2014.zip", encoder, preprocess_for_model, input_shape=(IMG_SIZE, IMG_SIZE))
    utils.save_pickle(train_img_embeds, "./models/train_img_embeds.pickle")
    utils.save_pickle(train_img_fns, "./models/train_img_fns.pickle")
    
    # extract validation features
    val_img_embeds, val_img_fns = utils.apply_model(
        "./test_data/val2014.zip", encoder, preprocess_for_model, input_shape=(IMG_SIZE, IMG_SIZE))
    utils.save_pickle(val_img_embeds, "./models/val_img_embeds.pickle")
    utils.save_pickle(val_img_fns, "./models/val_img_fns.pickle")
    

def sample_zip(fn_in, fn_out, rate=0.01, seed=42):
    np.random.seed(seed)
    with zipfile.ZipFile(fn_in) as fin, zipfile.ZipFile(fn_out, "w") as fout:
        sampled = filter(lambda _: np.random.rand() < rate, fin.filelist)
        for zInfo in sampled:
            fout.writestr(zInfo, fin.read(zInfo))
 
#sample_zip("train2014.zip", "train2014_sample.zip")
#sample_zip("val2014.zip", "val2014_sample.zip")