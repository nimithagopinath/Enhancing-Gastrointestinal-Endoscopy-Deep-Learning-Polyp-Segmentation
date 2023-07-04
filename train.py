
import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.metrics import Recall, Precision
from data import load_data, tf_dataset
from model import build_model

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


#Load the training and validation dataset
if __name__ == "__main__":
    ## Dataset
    print("")
    path = "CVC-ClinicDB/"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    print(len(train_x), len(valid_x), len(test_x))

    ## Setting up hyper parameters
    batch = 8
    lr = 1e-4
    epochs = 20

    #setup tf.data pipeline for training and validation dataset 
    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    #Building U-Net model
    model = build_model()

    # Defining the loss, optimiser and metrics
    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", Recall(), Precision(), iou]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

    # Defining callbacks
    callbacks = [
        ModelCheckpoint("files/model.h5"), #To save the weight file after each epoch
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4), #After each epoch if no improvement is seen for a 'patience' number of epochs it will reduce the learning rate by monitoring the validation loss
        CSVLogger("files/data.csv"), #It is used to save all the data into a csv file so that it is easy to visualize
        TensorBoard(), #It is only for visualisation
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    ]

    train_steps = len(train_x)//batch
    valid_steps = len(valid_x)//batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    #Start the training
    model.fit(train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks)