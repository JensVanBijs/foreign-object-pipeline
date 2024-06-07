#%%
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import os
import diplib as dip
import json
from imageio import imread

def create_training_data(n_objects, object_projection_dir, target_projection_dir, random_seed = 4):
    x_train = list()
    y_train = list()
    for n in range(n_objects):
        for p in range(n_projections):
            obj = imread(os.path.join(object_projection_dir, f"object_{n}", f"projection_{p}.tif")).astype(float)
            obj /= 65535  # NOTE: uint16
            img_max = np.max(obj)
            img_min = np.min(obj)
            normalized = (obj - img_min) / (img_max - img_min)
            x_train.append(np.asarray(dip.ColorSpaceManager.Convert(dip.Image(normalized), 'RGB')))

            target = imread(os.path.join(target_projection_dir, f"targets_{n}", f"projection_{p}.tif")).astype(float)
            target = dip.Image(target)
            target = dip.IsodataThreshold(target)
            y_train.append(np.asarray(target).astype(int))

    X_train, X_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed)
    X_train, X_validate, y_train, y_validate = np.array(X_train), np.array(X_validate), np.array(y_train), np.array(y_validate)
    X_train = X_train.reshape(X_train.shape)
    y_train = y_train.reshape(y_train.shape)
    X_validate = X_validate.reshape(X_validate.shape)
    y_validate = y_validate.reshape(y_validate.shape)

    return X_train, y_train, X_validate, y_validate

def encoder_block(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = tf.keras.layers.Conv2D(
          n_filters,
          3,
          activation = 'relu',
          padding = 'same',
          kernel_initializer = 'HeNormal'
    )(inputs)
    
    conv = tf.keras.layers.Conv2D(
          n_filters,
          3,
          activation = 'relu',
          padding = 'same',
          kernel_initializer = 'HeNormal'
    )(inputs)

    conv = tf.keras.layers.BatchNormalization()(conv, training=False)

    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection

def decoder_block(prev_layer_input, skip_layer_input, n_filters=32):
    up = tf.keras.layers.Conv2DTranspose(
            n_filters,
            (3, 3),
            strides = (2, 2),
            padding = 'same'
    )(prev_layer_input)
    
    merge = tf.keras.layers.concatenate([up, skip_layer_input], axis=3)
    
    conv = tf.keras.layers.Conv2D(
          n_filters,
          3,
          activation = 'relu',
          padding = 'same',
          kernel_initializer = 'HeNormal'
    )(merge)
    
    conv = tf.keras.layers.Conv2D(
          n_filters,
          3,
          activation = 'relu',
          padding = 'same',
          kernel_initializer = 'HeNormal'
    )(conv)
    
    return conv

def build_unet(input_size=(128, 128, 3), n_filters=32, n_classes=3):

    inputs = tf.keras.layers.Input(input_size)
    
    cblock1 = encoder_block(inputs, n_filters,dropout_prob=0, max_pooling=True)
    cblock2 = encoder_block(cblock1[0],n_filters*2,dropout_prob=0, max_pooling=True)
    cblock3 = encoder_block(cblock2[0], n_filters*4,dropout_prob=0, max_pooling=True)
    cblock4 = encoder_block(cblock3[0], n_filters*8,dropout_prob=0.3, max_pooling=True)
    cblock5 = encoder_block(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 

    ublock6 = decoder_block(cblock5[0], cblock4[1],  n_filters * 8)
    ublock7 = decoder_block(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = decoder_block(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = decoder_block(ublock8, cblock1[1],  n_filters)

    conv9 = tf.keras.layers.Conv2D(
            n_filters,
            3,
            activation = 'relu',
            padding = 'same',
            kernel_initializer = 'he_normal'
    )(ublock9)

    conv10 = tf.keras.layers.Conv2D(n_classes, 1, padding='same')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    return model

def train_model(X_train, y_train, X_validate, y_validate, epochs=20):
    unet = build_unet(input_size=(128,128,3), n_filters=32, n_classes=2)
    unet.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy']
    )
    results = unet.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data = (X_validate, y_validate))
    return unet, results

def save_model_results(n_objects, unet, results, save_path):
    save_path = os.path.abspath(save_path)
    save_path = os.path.join(save_path, f"{n_objects}_objects")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    unet.save(os.path.join(save_path, f"unet_{n_objects}_objects.keras"))
    history = results.history
    json.dump(history, open(os.path.join(save_path, f"history_{n_objects}_objects.json"), 'w'))

def train_model_experiment(object_projection_dir, target_projection_dir, included_n_objects, output_dir="./data/saved_models"):
    obj_dir = os.path.abspath(object_projection_dir)
    target_dir = os.path.abspath(target_projection_dir)
    outdir = os.path.abspath(output_dir)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    projection_dir_list = sorted(os.listdir(os.path.abspath(obj_dir)), key=lambda x: int(x.split("_")[0]))
    for n_dir in projection_dir_list:
        n_objects = int(n_dir.split("_")[0])
        if n_objects in included_n_objects:
            obj_proj_dir = os.path.join(obj_dir, n_dir)
            target_proj_dir = os.path.join(target_dir, n_dir)
            X_train, y_train, X_validate, y_validate = create_training_data(n_objects, obj_proj_dir, target_proj_dir)
            model, results = train_model(X_train, y_train, X_validate, y_validate)
            save_model_results(n_objects, model, results, outdir)

if __name__ == "__main__":
    n_projections = 180
    train_model_experiment("./data/object_projections", "./data/target_projections", included_n_objects=[1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])
