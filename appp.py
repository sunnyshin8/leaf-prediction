import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import cv2
from tensorflow import keras 
from keras.models import Sequential 
from keras import layers
from keras.preprocessing.image import ImageDataGenerator as IDG
from flask import Flask, request
from PIL import Image
import os

app = Flask(__name__)

def train_model():
    data = 'inputs/PlantVillage'

    BATCH_SIZE = 64
    IMG_SIZE = (200, 200)

    print(tf.config.list_physical_devices('GPU'))

    import os
    for dirname, _, filenames in os.walk('/inputs'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    train_datagen = IDG(
        rescale = 1/.225,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=20,
        height_shift_range=20,
        rotation_range=0.2,
        horizontal_flip=True,
    )

    valid_datagen = IDG(
        rescale = 1/.225,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=20,
        height_shift_range=20,
        rotation_range=0.2,
        horizontal_flip=True,
    )

    train_gen = train_datagen.flow_from_directory(
        data,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
    )

    valid_gen = valid_datagen.flow_from_directory(
        data,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    class_count = len(list(train_gen.class_indices.keys()))

    model = Sequential([
        layers.Conv2D(32, (3,3), strides=(2,2), padding='valid', activation='relu', input_shape=(200, 200, 3)),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(32, (3,3), activation='relu', padding='valid', strides=(2,2)),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='valid', strides=(2,2)),
        layers.MaxPooling2D(pool_size=(2,2)),
    #     layers.Conv2D(128, (3,3), activation='relu', padding='valid', strides=(2,2)),
    #     layers.MaxPooling2D(pool_size=(2,2)),
        
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
    #     layers.Dense(256, activation='relu'),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(64, activation='relu'),
        layers.Dense(class_count, activation='softmax'),
    ])

    opt = keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    history=model.fit(train_gen, validation_data=valid_gen, epochs=1, steps_per_epoch = train_gen.samples//BATCH_SIZE, validation_steps = valid_gen.samples//BATCH_SIZE, verbose=1)

    import matplotlib.pyplot as plt

    def plot_loss_acc(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        
    model.save('cnn_model.h5')
    tf.saved_model.save(model, '')


@app.route('/upload', methods=['POST'])
def get_prediction():

    if 'image' not in request.files:
        return 'No image file uploaded', 400

    imageFile = request.files['image']

    image1 = Image.open(imageFile)
    image1 = image1.resize((200, 200), resample=Image.Resampling.BILINEAR)
    input_arr = keras.preprocessing.image.img_to_array(image1)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    input_arr[0]
    model = tf.keras.models.load_model('cnn_model.h5')
    output=model.predict(input_arr)
    
    classname=['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

    print("predicted label:",classname[np.argmax(output[0])])
    # plt.imshow(input_arr[0].astype("uint8"))
    return "predicted label: " + classname[np.argmax(output[0])]

@app.route('/')
def welcome():
    return 'Welcome to leaf detection'

model_path = '2st.h5'
if not (os.path.exists(model_path)):
    train_model()
else:
    app.run()