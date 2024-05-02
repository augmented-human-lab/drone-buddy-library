import pkg_resources
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.optim as optim

from dronebuddylib.atoms.placerecognition.resources.common.googlenet.googlenet_places365 import GoogLeNetPlaces365

model = GoogLeNetPlaces365(True, num_classes=6)  # Assume new dataset has 10 classes

model_path = pkg_resources.resource_filename(__name__, 'resources/common/googlenet/googlenet_places365.pth')
# Load pre-trained weights
# model.load_state_dict(torch.load(model_path), strict=False)

# Load the entire pre-trained model weights into a state_dict
state_dict = torch.load(model_path)

# Remove weights for the final classifier layers
state_dict.pop('loss2_classifier_1.weight', None)
state_dict.pop('loss2_classifier_1.bias', None)

# Load the modified state dict into your model
model.load_state_dict(state_dict, strict=False)

model.loss2_classifier_1 = nn.Linear(in_features=1024, out_features=6)

# Ensure the classifier layer weights are reset
torch.nn.init.xavier_uniform_(model.loss2_classifier_1.weight)
torch.nn.init.zeros_(model.loss2_classifier_1.bias)

# Freeze all layers except the last classifier
for param in model.parameters():
    param.requires_grad = False
model.loss2_classifier_1.weight.requires_grad = True
model.loss2_classifier_1.bias.requires_grad = True

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data_path = pkg_resources.resource_filename(__name__, 'resources/test_data/training_data')
# Load your dataset
train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Check label format in the DataLoader
for inputs, labels in train_loader:
    print(labels)  # This should print out a 1D tensor of integer labels
    break
# Define optimizer and loss function
import torch.optim as optim

optimizer = optim.Adam(model.loss2_classifier_1.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()

num_epochs = 10
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    scheduler.step()

    # Validation step
    model.eval()
    # with torch.no_grad():
    #     validation_loss = 0
    #     for inputs, labels in validation_loader:
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         validation_loss += loss.item()
    #     print(f'Validation Loss: {validation_loss / len(validation_loader)}')

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

model_save_path = pkg_resources.resource_filename(__name__, 'resources/models/tf/trained_place_tf_model_1.pth')
# Save the model
torch.save(model.state_dict(), model_save_path)
