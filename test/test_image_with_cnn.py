import tensorflow as tf
import numpy as np

# Replace 'model_path' with the path to your saved model
model_path = r'C:\Users\Public\projects\drone-buddy-library\test\model\cnn\model.keras'
model = tf.keras.models.load_model(model_path)


# Replace 'image_path' with the path to your image file
image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\bottle.jpeg"
# image_path = r'C:\Users\Public\projects\drone-buddy-library\test\object_images\malsha_cup\1.jpeg'

# image_path = 'path_to_your_image.jpg'
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))  # Resize the image to 64x64 or whatever your model expects
img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert the image to an array
img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
img_array /= 255.0  # Scale the pixel values in the same way you did for training data

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)  # Assuming your model uses categorical outputs
print(f"Predicted class: {predicted_class}")

# {0: 'malsha_bottle', 1: 'malsha_cup', 2: 'unknown'}
#