import os
import re

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from djitellopy import Tello
from torch.utils.data import DataLoader, TensorDataset

from dronebuddylib.atoms.objectdetection.yolo_object_detection_impl import YOLOObjectDetectionImpl
from dronebuddylib.atoms.objectidentification.object_recognition_resnet_impl import ObjectRecognitionResnetImpl
from dronebuddylib.atoms.objectidentification.object_recognition_yolo_impl import ObjectRecognitionYOLOImpl
from dronebuddylib.models import EngineConfigurations, AtomicEngineConfigurations


class CustomClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CustomClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 100)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def extract_features(image):
    object_dec = YOLOObjectDetectionImpl(EngineConfigurations({'OBJECT_DETECTION_YOLO_VERSION': 'yolov8n.pt'}))
    results = object_dec.get_detected_objects_temp(image)
    return results


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def create_data_set():
    features_list = []
    labels_list = []
    label_to_index = {}  # Dictionary to map class names to integers
    current_label_index = 0

    file_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images"
    for class_dir in os.listdir(file_path):
        if not os.path.isdir(os.path.join(file_path, class_dir)):
            continue

        # Assign an integer to the class name if it hasn't been assigned already
        if class_dir not in label_to_index:
            label_to_index[class_dir] = current_label_index
            current_label_index += 1

        for img_path in image_files_in_folder(os.path.join(file_path, class_dir)):
            img = cv2.imread(img_path)
            features = extract_features(img)
            if features is not None and len(features) > 0:
                new_features = extract_feature_set(features)
                features_list.append(new_features)  # Assuming new_features is a NumPy array
                labels_list.append(label_to_index[class_dir])  # Use integer label

    # Convert the list of features to a single NumPy array
    features_array = np.array(features_list)
    features_tensor = torch.tensor(features_array, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    return features_tensor, labels_tensor


def load_model(model_path, input_dim, num_classes):
    model = CustomClassifier(input_dim, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_image(img_path):
    # Preprocess the image as required by YOLO model (resize, normalize, etc.)
    img = cv2.imread(img_path)
    # Assuming YOLO model requires images in BGR format, if not convert it accordingly
    return img


def predict_image(model, img_path, object_detector):
    img = preprocess_image(img_path)
    yolo_results = object_detector.get_detected_objects_temp(img)
    features = extract_feature_set(yolo_results)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(features_tensor)
    return output


def extract_feature_set(yolo_results):
    # Flatten the boxes and confidences into a single feature vector
    # Assuming that yolo_results.boxes is a list of 'Boxes' objects.
    all_boxes = np.concatenate([result.boxes.data.cpu().numpy() for result in yolo_results], axis=0)
    all_confs = np.concatenate([result.boxes.conf.cpu().numpy() for result in yolo_results], axis=0)
    # Now you have a 2D array of all boxes and confidences which you can average or process further.
    # If you need a fixed-size feature vector, you might average the boxes and confidences:
    if len(all_boxes) > 0:  # Check if there are any boxes at all
        average_boxes = np.mean(all_boxes, axis=0)
        average_confs = np.mean(all_confs, axis=0)
        features = np.concatenate((average_boxes, average_confs), axis=None)
        return features
    else:
        # Return a zero array if there are no boxes detected
        return np.zeros((5,))  # This should match the expected feature vector size


def train(num_epochs):
    features, labels = create_data_set()
    dataset = TensorDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = CustomClassifier(features.shape[1], 2)  # Adjust number of classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_features, batch_labels in data_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}')
    model.eval()
    return model


def transfer_learning_model():
    num_epochs = 10
    model_path = r'C:\Users\Public\projects\drone-buddy-library\test\model\model.pth'
    # model = train(num_epochs)
    # torch.save(model.state_dict(), model_path)

    # Load the trained model
    input_dim = 2048  # The exact input dimension of your model
    num_classes = 2  # Adjust as needed
    model = load_model(model_path, input_dim, num_classes)

    # Initialize the YOLO object detector
    object_detector = YOLOObjectDetectionImpl(EngineConfigurations({'OBJECT_DETECTION_YOLO_VERSION': 'yolov8n.pt'}))

    # Folder with new images
    test_folder = r'C:\Users\Public\projects\drone-buddy-library\test\object_images\test_images'
    test_images = image_files_in_folder(test_folder)

    # Predict and print results for each new image
    for test_image in test_images:
        prediction = predict_image(model, test_image, object_detector)
        predicted_class = torch.argmax(prediction, dim=1)
        print(f'Image: {test_image}, Predicted class: {predicted_class.item()}')


if __name__ == '__main__':
    # image_path = r'C:\Users\Public\projects\drone-buddy-library\test\object_images\blaaaa.jpeg'
    # image_path = r'C:\Users\Public\projects\drone-buddy-library\test\object_images\malsha_cup\1.jpeg'
    image_path = r'C:\Users\Public\projects\drone-buddy-library\test\object_images\di_cup.jpeg'
    # image_path = r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectrecognition\resources\model\data\training_data\malsha_bottle\malsha_bottle_72.jpg'
    image = cv2.imread(image_path)
    tello = Tello()
    tello.connect()
    tello.streamon()
    tello.get_frame_read().frame
    object_recognition = ObjectRecognitionYOLOImpl(
        EngineConfigurations({AtomicEngineConfigurations.OBJECT_RECOGNITION_YOLO_WEIGHTS_PATH: 'yolov8n.pt',
                              AtomicEngineConfigurations.OBJECT_RECOGNITION_YOLO_DRONE_INSTANCE: tello,
                              }))
    # object_recognition = ObjectRecognitionResnetImpl(
    #     EngineConfigurations({AtomicEngineConfigurations.OBJECT_RECOGNITION_YOLO_WEIGHTS_PATH: 'yolov8n.pt',
    #                           # AtomicEngineConfigurations.OBJECT_RECOGNITION_YOLO_DRONE_INSTANCE: tello,
    #                           }))
    # print("battery: ", tello.get_battery())
    # print("temperature: ", tello.get_temperature())
    # object_recognition.extract_and_plot_features(image, 20, 5)
    object_recognition.remember_object(None, "cup", "dinithi_cup")
    # object_recognition.train(None, "bottle", "hot_water_bottlecv")
    # object_recognition.create_memory("resnet 50 with unknown objects")
    # results = object_recognition.recognize_objects(image)
    # print(results)
    # object_recognition.test_image(image_path)
