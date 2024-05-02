import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from ultralytics import YOLO

from dronebuddylib.atoms.objectidentification.resources.matching.inferenceDataset import InferenceDataset, transform, \
    load_images_from_folder
from dronebuddylib.atoms.objectidentification.resources.matching.model import SiameseModel

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

# tech debt: making it relative path
# model_file_path = r'C:\Users\wangz\drone\matching\src\siamese_model_e19_b4_lr1e-05_num100_emb20.pth'
model_file_path = r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\matching\latest_model\newmodel.pth'


class SiameseNetworkAPI():
    def __init__(self, obj_tensor):
        self.obj_tensor = obj_tensor
        # self.obj_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.obj_detection_model = YOLO('yolov8n.pt')
        # Set half precision mode
        if torch.cuda.is_available():

            # self.obj_detection_model = self.obj_detection_model
            # Fuse the model first (if applicable, and fusion is a method provided by your specific model class)
            self.obj_detection_model.fuse()  # Fuse conv and bn layers before converting to half precision

            self.obj_detection_model = self.obj_detection_model.half()  # Set to half precision only if CUDA is available
            self.obj_detection_model = self.obj_detection_model.to('cuda')  # Move model to GPU
        else:
            self.obj_detection_model = self.obj_detection_model
        # to capture more objects, the conf threshold was reduced
        # self.obj_detection_model.iou = 0.45
        self.obj_detection_model.conf = 0.1
        self.siamese_network_model = SiameseModel(
            base_model=efficientnet_v2_s,
            base_model_weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        state_dict = torch.load(model_file_path)
        print(state_dict.keys())
        print(self.siamese_network_model.state_dict().keys())
        self.siamese_network_model.load_state_dict(torch.load(model_file_path))
        self.siamese_network_model.eval()
        # Define your transformations
        # Define your transformations
        self.transform = transforms.Compose([
            transforms.Resize((416, 416)),  # Resize the image, works with PIL Images
            transforms.ToTensor(),  # Converts the PIL Image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizes the tensor
        ])

        path = r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\training_data'
        transform = transforms.Compose([
            transforms.Resize((228, 228)),  # Resize the image
            # transforms.ConvertImageDtype(torch.float32),  # Convert images to float tensors
            transforms.ToTensor(),  # Convert images to tensor here, not in the inference method
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
        ])
        self.reference_images = load_images_from_folder(path, transform=transform)

    def get_detected_objects(self, img):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Predict objects in the image using the object detection model
        objects_in_room = self.obj_detection_model.predict(image_rgb)

        # Iterate through each detected object and its bounding box
        for result in objects_in_room:
            for index, bbox in enumerate(result.boxes):
                # Extract bounding box coordinates and convert them to integers
                bbox_tensor = result.boxes[index].xyxy.cpu().numpy().tolist()[0]
                xmin, ymin, xmax, ymax = map(int, bbox_tensor)
                # xmin, ymin, xmax, ymax = map(int, bbox.xyxy)
                #
                # Draw rectangle on the image
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Put the class label text on the image
                label = result.names[int(bbox.cls)]
                cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the image
        cv2.imshow("Detected Objects", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def inference(self, img):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        objects_in_room = self.obj_detection_model.predict(image_rgb)
        all_xy_coords = []
        all_conf_scores = []

        print("********************************************")

        for result in objects_in_room:
            for index, cls_tensor in enumerate(result.boxes.cls):
                print("testing for object : ", result.names[int(cls_tensor)])

                bbox_tensor = result.boxes[index].xyxy.cpu().numpy().tolist()[0]
                xmin, ymin, xmax, ymax = map(int, bbox_tensor)
                cropped_image = img[ymin:ymax, xmin:xmax]
                cropped_image_pil = Image.fromarray(cropped_image)
                transformed_image = self.transform(cropped_image_pil)
                # show cropped image for testing
                cv2.imshow("cropped_" + str(index), cropped_image)
                cv2.waitKey(0)

                if len(transformed_image.shape) == 3:
                    transformed_image = transformed_image.unsqueeze(0)

                # Compare with each reference image
                for ref_label, ref_image in self.reference_images.items():
                    print("testing for reference object : ", ref_label)
                    ref_image_tensor = ref_image  # Add batch dimension
                    similarity = self.siamese_network_model(ref_image_tensor, transformed_image)

                    # Assuming similarity is a 1D tensor with shape [8]
                    threshold = 0.3  # Define your threshold

                    # Apply a sigmoid since your values seem to be logits and you want them in the [0, 1] range
                    similarity = torch.sigmoid(similarity)
                    # Perform an element-wise comparison to the threshold
                    greater_than_threshold = similarity > threshold
                    # Now you have a tensor of booleans where each element is the result of the comparison

                    if greater_than_threshold.any():
                        print("Eureka for ", ref_label)
                        print(similarity)
                    # At least one value exceeds the threshold, handle this case

                    count_above_threshold = torch.sum(similarity > threshold)
                    final_score = count_above_threshold / similarity.shape[0]
                    median_score = torch.median(similarity)
                    # Low Standard Deviation: If the standard deviation of the similarity scores is low,
                    # it means that the scores are closely clustered around the mean (average) score.
                    # This suggests that the model is consistently returning similar values for all the comparisons between the input image and the training images.
                    # In practical terms, a low standard deviation could indicate that the input image
                    # either matches closely or does not match at all with the reference images,
                    # with little variance in confidence across different comparisons.

                    # High Standard Deviation: A high standard deviation indicates that there is a wide spread in the values of the similarity scores.
                    # This can mean that while some of the training images might be very similar to the input image (resulting in high similarity scores),
                    # others might be quite different (resulting in low scores).
                    # High variance can suggest that the input image has some features strongly resembling only certain training images,
                    # which might be useful for identifying specific characteristics or outliers.

                    std_deviation = torch.std(similarity)

                    print("Final Score : ", final_score.item())
                    print("Median Score : ", median_score.item())
                    print("Standard Deviation : ", std_deviation.item())
                    print("--------------------------------------")
                print("==================================")
            print("##################################")
        print("********************************************")
        cv2.destroyAllWindows()

        return all_xy_coords, all_conf_scores
