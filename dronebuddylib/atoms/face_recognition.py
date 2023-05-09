import os

import cv2

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path to the directory containing the training images
train_dir = 'path/to/training/images'


# Function to read the training images and labels
def get_images_and_labels(train_dir):
    image_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    images = []
    labels = []
    for image_path in image_paths:
        # Load the image in grayscale
        image = cv2.imread(image_path, 0)
        # Extract the label from the image filename
        label = int(os.path.split(image_path)[-1].split(".")[0].replace("subject", ""))
        images.append(image)
        labels.append(label)
    return images, labels


# Get the training images and labels
images, labels = get_images_and_labels(train_dir)

# Train the recognizer
recognizer.train(images, np.array(labels))
