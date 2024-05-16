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
    def __init__(self, obj_tensor=None):
        self.obj_tensor = obj_tensor
        self.obj_detection_model = YOLO('yolov8n.pt')
        # Set half precision mode
        if torch.cuda.is_available():

            # Fuse the model first (if applicable, and fusion is a method provided by your specific model class)
            self.obj_detection_model.fuse()  # Fuse conv and bn layers before converting to half precision

            self.obj_detection_model = self.obj_detection_model.half()  # Set to half precision only if CUDA is available
            self.obj_detection_model = self.obj_detection_model.to('cuda')  # Move model to GPU
        else:
            self.obj_detection_model = self.obj_detection_model
        # to capture more objects, the conf threshold was reduced

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

    def get_embeddings(self, img):
        image_1_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t_i_1 = self.transform(Image.fromarray(image_1_rgb))

        squeezed = t_i_1.unsqueeze(0)
        return self.siamese_network_model.get_embedding(squeezed)

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

    def inference_1(self, img):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        objects_in_room = self.obj_detection_model.predict(image_rgb)
        # Initialize dictionaries to store cosine distances and similarities
        all_cosine_distances = {}
        all_similarities = {}

        print("********************************************")

        for img_idx, result in enumerate(objects_in_room):  # Use enumerate to iterate over objects_in_room
            image_cosine_distances = {}
            image_similarities = {}

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
                    class_cosine_distances = []
                    class_similarities = []

                    for ref_idx, image in enumerate(ref_image):
                        image_unsqueezed = image.unsqueeze(0)
                        similarity, cosine_distance = self.siamese_network_model(image_unsqueezed, transformed_image)
                        # Assuming similarity is a 1D tensor with shape [8]
                        threshold = 0.3  # Define your threshold

                        # Apply a sigmoid since your values seem to be logits and you want them in the [0, 1] range
                        similarity_torch = torch.sigmoid(similarity)
                        print("Similarity torch : ", similarity_torch)

                        # Perform an element-wise comparison to the threshold
                        greater_than_threshold = similarity_torch > threshold
                        # Now you have a tensor of booleans where each element is the result of the comparison
                        # Store class cosine distances and similarities in image dictionaries
                        image_cosine_distances[ref_label] = class_cosine_distances
                        image_similarities[ref_label] = class_similarities

                        if greater_than_threshold.any():
                            print("Eureka for ", ref_label)
                            print(similarity)

                        print("--------------------------------------")
                        # Store image dictionaries in the main dictionaries
                    all_cosine_distances[f"Image_{img_idx + 1}"] = image_cosine_distances
                    all_similarities[f"Image_{img_idx + 1}"] = image_similarities

                print("==================================")
            all_cosine_distances.append(image_cosine_distances)
            all_similarities.append(image_similarities)
            print("##################################")
        print("********************************************")
        cv2.destroyAllWindows()

        return all_cosine_distances, all_similarities

    def inference_2(self, img):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        objects_in_room = self.obj_detection_model.predict(image_rgb)
        # Initialize dictionaries to store cosine distances and similarities
        all_cosine_distances = {}
        all_similarities = {}

        print("********************************************")
        my_index = 0
        for img_idx, result in enumerate(objects_in_room):

            image_cosine_distances = {}
            image_similarities = {}

            for index, cls_tensor in enumerate(result.boxes.cls):
                print("testing for object : ", result.names[int(cls_tensor)])

                bbox_tensor = result.boxes[index].xyxy.cpu().numpy().tolist()[0]
                xmin, ymin, xmax, ymax = map(int, bbox_tensor)
                cropped_image = img[ymin:ymax, xmin:xmax]
                cropped_image_pil = Image.fromarray(cropped_image)
                transformed_image = self.transform(cropped_image_pil)
                # show cropped image for testing
                # save cropped image for testing
                cv2.imwrite("cropped_" + result.names[int(cls_tensor)] + "_" + str(index) + ".jpg", cropped_image)

                # cv2.imwrite(f"cropped_{str(index)}.jpg", cropped_image)

                # cv2.imshow("cropped_" + str(index), cropped_image)
                # cv2.waitKey(0)

                if len(transformed_image.shape) == 3:
                    transformed_image = transformed_image.unsqueeze(0)

                # Compare with each reference image
                for ref_label, ref_image in self.reference_images.items():
                    print("testing for reference object : ", ref_label)
                    class_cosine_distances = []
                    class_similarities = []

                    for ref_idx, image in enumerate(ref_image):
                        image_unsqueezed = image.unsqueeze(0)
                        similarity, cosine_distance = self.siamese_network_model(image_unsqueezed, transformed_image)
                        # Assuming similarity is a 1D tensor with shape [8]
                        threshold = 0.3  # Define your threshold

                        # Apply a sigmoid since your values seem to be logits and you want them in the [0, 1] range
                        similarity_torch = torch.sigmoid(similarity)
                        print("Similarity torch : ", similarity_torch)

                        # Perform an element-wise comparison to the threshold
                        greater_than_threshold = similarity_torch > threshold
                        # Now you have a tensor of booleans where each element is the result of the comparison
                        # Store class cosine distances and similarities in image dictionaries
                        class_cosine_distances.append(cosine_distance.item())
                        class_similarities.append(similarity_torch.item())

                        if greater_than_threshold.any():
                            print("Eureka for ", ref_label)
                            print(similarity)

                        print("--------------------------------------")
                    image_cosine_distances[ref_label] = class_cosine_distances
                    image_similarities[ref_label] = class_similarities
                    # Store image dictionaries in the main dictionaries
                all_cosine_distances[f"Image_{my_index}"] = image_cosine_distances
                all_similarities[f"Image_{my_index}"] = image_similarities
                my_index += 1

                print("==================================")
            print("##################################")

        print("********************************************")
        cv2.destroyAllWindows()

        return all_cosine_distances, all_similarities

    def two_image_inference(self, img_1, img_2):
        image_1_rgb = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        image_2_rgb = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        t_i_1 = self.transform(Image.fromarray(image_1_rgb))
        t_i_2 = self.transform(Image.fromarray(image_2_rgb))
        t_i_2 = t_i_2.unsqueeze(0)
        t_i_1 = t_i_1.unsqueeze(0)
        output = self.siamese_network_model(t_i_1, t_i_2)
        print("output : ", output)
        return output

    def two_image_inference_difference(self, img_1, img_2):
        image_1_rgb = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        image_2_rgb = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        t_i_1 = self.transform(Image.fromarray(image_1_rgb))
        t_i_2 = self.transform(Image.fromarray(image_2_rgb))
        t_i_2 = t_i_2.unsqueeze(0)
        t_i_1 = t_i_1.unsqueeze(0)
        output = self.siamese_network_model.forward_difference(t_i_1, t_i_2)
        print("output : ", output)
        return output

    def inference(self, img):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        objects_in_room = self.obj_detection_model.predict(image_rgb)

        # Initialize dictionaries to store cosine distances and similarities
        all_cosine_distances = {}
        all_similarities = {}

        print("********************************************")
        my_index = 0

        for img_idx, result in enumerate(objects_in_room):
            image_cosine_distances = {}
            image_similarities = {}

            for index, cls_tensor in enumerate(result.boxes.cls):
                print("testing for object : ", result.names[int(cls_tensor)])
                object_class = result.names[int(cls_tensor)]

                bbox_tensor = result.boxes[index].xyxy.cpu().numpy().tolist()[0]
                xmin, ymin, xmax, ymax = map(int, bbox_tensor)
                cropped_image = img[ymin:ymax, xmin:xmax]
                cropped_image_pil = Image.fromarray(cropped_image)
                transformed_image = self.transform(cropped_image_pil)
                # show cropped image for testing
                # save cropped image for testing
                cv2.imwrite("cropped_" + result.names[int(cls_tensor)] + "_" + str(index) + ".jpg", cropped_image)

                if len(transformed_image.shape) == 3:
                    transformed_image = transformed_image.unsqueeze(0)

                image_cosine_distances = {}
                image_similarities = {}
                # Compare with each reference image
                for ref_label, ref_image in self.reference_images.items():
                    print("testing for reference object : ", ref_label)
                    class_cosine_distances = []
                    class_similarities = []

                    for ref_idx, image in enumerate(ref_image):
                        image_unsqueezed = image.unsqueeze(0)
                        similarity, cosine_distance = self.siamese_network_model(image_unsqueezed, transformed_image)
                        # Apply a sigmoid since your values seem to be logits and you want them in the [0, 1] range
                        similarity_torch = torch.sigmoid(similarity)

                        # Store class cosine distances and similarities in image dictionaries
                        class_cosine_distances.append(cosine_distance.item())
                        class_similarities.append(similarity_torch.item())

                    # Store class dictionaries in image dictionaries
                    image_cosine_distances[ref_label] = class_cosine_distances
                    image_similarities[ref_label] = class_similarities

                # Store image dictionaries in the main dictionaries
                all_cosine_distances[f"Image_{object_class}_{my_index}"] = image_cosine_distances
                all_similarities[f"Image_{object_class}_{my_index}"] = image_similarities
                my_index += 1

                print("==================================")
            print("##################################")

        print("********************************************")
        cv2.destroyAllWindows()

        return all_cosine_distances, all_similarities
