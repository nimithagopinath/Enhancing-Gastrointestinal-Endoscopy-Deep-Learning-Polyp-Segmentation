import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm # tqdm is a library in Python which is used for creating Progress Meters or Progress Bars.


#Load dataset, split into training, validation, test set

def load_data(path, split=0.1):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "masks/*")))

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

    print(total_size, valid_size, test_size)



# Read the image and the mask

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    #size is (256,256,3)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    #size is 256,256
    x = np.expand_dims(x, axis=-1)
     #size is (256,256,1)
    return x


#define tf pipeline

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset



#if __name__ == '__main__':
    #path = "CVC-ClinicDB"
    #(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data (path)
    #print (len(train_x), len(valid_x), len(test_x))

    #ds = tf_dataset(test_x, test_y)
    #for x,y in ds:
        #print (x.shape, y.shape)
       # break



# Building the U-net architecture 
def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def build_model():
    size = 256
    num_filters = [16, 32, 48, 64]
    inputs = Input((size, size, 3))

    skip_x = []
    x = inputs
    
    ## Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)

    ## Bridge
    x = conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)


#if __name__ == "__main__":
model = build_model()
model.summary()



def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


#Load the training and validation dataset
#if __name__ == "__main__":
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


def read_image2(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def read_mask2(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

#if __name__ == "__main__":
    ## Load the testing dataset
path = "CVC-ClinicDB/"
batch_size = 8
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    # Setting up the tf.data pipeline for testing dataset
test_dataset = tf_dataset(test_x, test_y, batch=batch_size)
test_steps = (len(test_x)//batch_size)

if len(test_x) % batch_size != 0:
    test_steps += 1

    # Loading the training U-Net models
with CustomObjectScope({'iou': iou}):
    model = tf.keras.models.load_model("files/model.h5")

model.evaluate(test_dataset, steps=test_steps)

    # Make predictions and save the results
for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
    x = read_image2(x)
    y = read_mask2(y)
    y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        #y_pred = y_pred[0] > 0.5
    h, w, _ = x.shape
    white_line = np.ones((h, 10, 3)) * 255.0

    all_images = [
        x * 255.0, white_line,
        mask_parse(y), white_line,
        mask_parse(y_pred) * 255.0
    ]
    image = np.concatenate(all_images, axis=1)
    cv2.imwrite(f"results2/{i}.png", image)