# Make predictions on new images
import numpy as np
from tensorflow.keras.preprocessing import image

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = (224, 224)


model = tf.keras.models.load_model('image_classifier_model.h5') 


# Make predictions on new images
import os
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            image_path = os.path.join(folder_path, filename)
            img = image.load_img(image_path, target_size=image_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Rescale pixel values
            prediction = model.predict(img_array)
            if prediction[0] > 0.5:
                print("Prediction: " + filename + " is Normal Page " + str(prediction[0]))
            else:
                print("Prediction: " + filename + " is Error Page " + str(prediction[0]))

# Example usage:
folder_path = 'test'
prediction = predict_image(folder_path)
