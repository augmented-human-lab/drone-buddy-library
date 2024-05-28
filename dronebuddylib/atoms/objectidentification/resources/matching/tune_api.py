import argparse
import time

import pkg_resources
from matplotlib import pyplot as plt
from torchsummary import summary
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from .dataset import *
from dronebuddylib.atoms.objectidentification.resources.matching.model import *

transforms = T.Compose([
    T.Resize((228, 228)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomAffine(degrees=(0, 45), translate=(0., 0.2)),
    T.ConvertImageDtype(torch.float)
])


def dataloader(full_dataset, args, output_folder_path):
    train_size = int(args.train_val_split * args.num_samples)
    val_size = args.num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # note: when using a GPU it’s better to set pin_memory=True, this instructs DataLoader to use pinned memory
    # and enables faster and asynchronous memory copy from the host to the GPU.
    trainloader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=True
    )
    valloader = DataLoader(
        val_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=True
    )

    return trainloader, valloader


def train(model, criterion, optimizer, trainloader, valloader, args, device, output_folder_path, lr_scheduler=None):
    best_vloss = 1_000_000.

    for epoch in range(args.epochs):
        # training
        model.train(True)

        tloss_history = []
        tcorrect = 0
        ttotal = 0

        for _, data in enumerate(trainloader):
            img0, img1, _, _, label = data
            img0, img1, label = img0.to(device, dtype=torch.float), img1.to(device, dtype=torch.float), label.to(device,
                                                                                                                 dtype=torch.float)

            # starting from PyTorch 1.7, call model or optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)

            out1 = model(img0, img1)
            output = out1.squeeze(1)
            loss = criterion(output, label)

            tcorrect += torch.count_nonzero(label == (torch.sigmoid(output) > 0.5)).item()
            ttotal += len(label)

            loss.backward()
            optimizer.step()
            tloss_history.append(loss.item())

        if lr_scheduler != None:
            lr_scheduler.step()

        # validation
        model.eval()

        vloss_history = []
        vcorrect = 0
        vtotal = 0

        reference_embeddings = {class_name: [] for class_name in valloader.dataset.dataset.class_names}

        # disable gradient calculation for validation or inference
        with torch.no_grad():
            for _, vdata in enumerate(valloader):
                vimg0, vimg1, class_idx0, class_idx1, vlabel = vdata
                vimg0, vimg1, vlabel = vimg0.to(device, dtype=torch.float), vimg1.to(device,
                                                                                     dtype=torch.float), vlabel.to(
                    device, dtype=torch.float)
                yout1 = model(vimg0, vimg1)
                voutput = yout1.squeeze(1)
                vloss = criterion(voutput, vlabel)

                # compute embeddings for each image
                # embedding_0 = model.get_embedding(vimg0.cpu().numpy())
                # embedding_1 = model.get_embedding(vimg1.cpu().numpy())

                embedding_0 = model.get_embedding(vimg0.to(device))
                embedding_1 = model.get_embedding(vimg1.to(device))

                # Store embeddings and their corresponding class names
                # reference_embeddings[valloader.dataset.dataset.class_names[class_idx0]].append(embedding_0)

                reference_embeddings[valloader.dataset.dataset.class_names[class_idx0.to(device)[0]]].append(
                    embedding_0)
                reference_embeddings[valloader.dataset.dataset.class_names[class_idx1.to(device)[0]]].append(
                    embedding_1)
                # reference_embeddings[valloader.dataset.dataset.class_names[class_idx1.item()]].append(embedding_1)

                # reference_embeddings[valloader.dataset.dataset.class_names[class_idx1]].append(embedding_1)

                vcorrect += torch.count_nonzero(vlabel == (torch.sigmoid(voutput) > 0.5)).item()
                vtotal += len(vlabel)

                vloss_history.append(vloss.item())

        # Save reference images
        # Now, select the most representative image for each class based on the embeddings
        # for class_name, embeddings in reference_embeddings.items():
        #     # You might choose the mean, median, or another method to find the reference
        #     # Here, we'll just use the first image's embedding as an example
        #     ref_embedding = embeddings[0]  # Replace this with your actual selection method
        #     ref_embedding_cpu = ref_embedding.cpu().numpy()
        #
        #     # Save the reference embedding
        #     ref_embedding_path = os.path.join(output_folder_path, f'ref_embedding_{class_name}.npy')
        #     np.save(ref_embedding_path, ref_embedding_cpu)

        # Calculate train and val loss
        avg_tloss = np.mean(tloss_history)
        avg_vloss = np.mean(vloss_history)
        avg_tacc = tcorrect / ttotal
        avg_vacc = vcorrect / vtotal

        print(
            f'[{time.ctime()}] [Epoch {epoch}] train loss: {avg_tloss:.5f} val loss: {avg_vloss:.5f} train acc: {avg_tacc:.5f} val acc: {avg_vacc:.5f}\n')

        # Track best performance, and save the model's state
        if epoch > 0 and avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'{output_folder_path}newmodel_6_5.pth'
            torch.save(model.state_dict(), model_path)

        if (epoch + 1) % 5 == 0:
            print(
                f'[{time.ctime()}] [Epoch {epoch}] train loss: {avg_tloss:.5f} val loss: {avg_vloss:.5f} train acc: {avg_tacc:.5f} val acc: {avg_vacc:.5f}\n')

            model_path = f'{output_folder_path}newmodel.pth'
            torch.save(model.state_dict(), model_path)


def predict_with_reference_images(model, image_path, reference_embeddings, device):
    model.eval()
    new_image = read_image(image_path).to(device)
    new_image = new_image.unsqueeze(0)  # Add batch dimension
    new_embedding = model.get_embedding(new_image)

    closest_class = None
    closest_dist = float('inf')

    # Compare with each reference embedding
    for class_name, ref_embedding in reference_embeddings.items():
        # dist = np.linalg.norm(new_embedding.cpu().numpy() - ref_embedding)
        dist = np.linalg.norm(new_embedding.cpu().numpy() - ref_embedding)
        if dist < closest_dist:
            closest_dist = dist
            closest_class = class_name

    return closest_class


def tune(feature_extractor_model="efficientnetv2", num_samples=100, emb_size=20, epochs=10, lr=1e-5, batch_size=4,
         train_val_split=0.8, num_workers=1, seed=0, output_folder_name=None, lr_scheduler=False,
         pretrained_weights=None):
    args = argparse.Namespace(
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        num_samples=num_samples,
        emb_size=emb_size,
        train_val_split=train_val_split,
        num_workers=num_workers,
        seed=seed,
        output_folder_name=output_folder_name,
        lr_scheduler=lr_scheduler,
        model=feature_extractor_model,
        pretrained_weights=pretrained_weights
    )
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    output_folder_path = str(Path(__file__).resolve().parent) + '\\latest_model\\'
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ensure reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    print(f'[{time.ctime()}] getting the train and val dataloaders\n')

    # getting the train and val dataloaders
    # image_path = pkg_resources.resource_filename(__name__, "resources/model/data/training_data/")
    # C:\Users\Public\projects\drone - buddy - library\dronebuddylib\atoms\objectidentification\resources\model\data\benchmark
    # images_folder_path = str(Path(__file__).resolve().parent.parent) + '\\model\\data\\training_data\\',

    full_dataset = SiameseDataset(
        images_folder_path=str(Path(__file__).resolve().parent.parent) + '\\model\\data\\training_data\\',
        transform=transform,
        num_samples=args.num_samples
    )
    trainloader, valloader = dataloader(full_dataset, args, output_folder_path)

    print(f'[{time.ctime()}] setting up training parameters\n')

    # setting up training parameters
    if args.model == 'resnet18':
        base_model = resnet18
        base_model_weights = ResNet18_Weights.IMAGENET1K_V1
    elif args.model == 'resnet50':
        base_model = resnet50
        base_model_weights = ResNet50_Weights.IMAGENET1K_V1
    elif args.model == 'resnet50v2':
        base_model = resnet50
        base_model_weights = ResNet50_Weights.IMAGENET1K_V2
    elif args.model == 'resnet101':
        base_model = resnet101
        base_model_weights = ResNet101_Weights.IMAGENET1K_V1
    elif args.model == 'efficientnetv2':
        base_model = efficientnet_v2_s
        base_model_weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    else:  # args.model == 'mobilenetv2'
        base_model = mobilenet_v2
        base_model_weights = MobileNet_V2_Weights.IMAGENET1K_V1

    print(f'[{time.ctime()}] siamese network will be based off {args.model}\n')

    # model = SiameseModel(emb_size=args.emb_size, base_model=base_model, base_model_weights=base_model_weights)
    model = SiameseModel(base_model=base_model, base_model_weights=base_model_weights)

    # model_path = pkg_resources.resource_filename(__name__, "resources/model/data/training_data/")
    trained_model_path = str(Path(__file__).resolve().parent) + "\\latest_model\\newmodel.pth"
    model.load_state_dict(torch.load(trained_model_path))
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    else:
        lr_scheduler = None

    print(f'[{time.ctime()}] training the model\n')

    # model training
    train(model, criterion, optimizer, trainloader, valloader, args, device, output_folder_path, lr_scheduler)

    print(f'[{time.ctime()}] done training the model\n')


def loadModel():
    args = argparse.Namespace(
        batch_size=4,
        epochs=10,
        lr=1e-5,
        num_samples=100,
        emb_size=20,
        train_val_split=0.8,
        num_workers=1,
        seed=0,
        output_folder_name=None,
        lr_scheduler=False,
        model='efficientnetv2',
        pretrained_weights=None
    )
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    output_folder_path = str(Path(__file__).resolve().parent) + '\\latest_model\\'
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ensure reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    print(f'[{time.ctime()}] getting the train and val dataloaders\n')

    # getting the train and val dataloaders
    full_dataset = SiameseDataset(
        images_folder_path=str(Path(__file__).resolve().parent.parent) + '\\model\\data\\training_data\\',
        transform=transform,
        num_samples=args.num_samples
    )
    trainloader, valloader = dataloader(full_dataset, args, output_folder_path)

    print(f'[{time.ctime()}] setting up training parameters\n')

    # setting up training parameters
    if args.model == 'resnet18':
        base_model = resnet18
        base_model_weights = ResNet18_Weights.IMAGENET1K_V1
    elif args.model == 'resnet50':
        base_model = resnet50
        base_model_weights = ResNet50_Weights.IMAGENET1K_V1
    elif args.model == 'resnet50v2':
        base_model = resnet50
        base_model_weights = ResNet50_Weights.IMAGENET1K_V2
    elif args.model == 'resnet101':
        base_model = resnet101
        base_model_weights = ResNet101_Weights.IMAGENET1K_V1
    elif args.model == 'efficientnetv2':
        base_model = efficientnet_v2_s
        base_model_weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    else:  # args.model == 'mobilenetv2'
        base_model = mobilenet_v2
        base_model_weights = MobileNet_V2_Weights.IMAGENET1K_V1

    print(f'[{time.ctime()}] siamese network will be based off {args.model}\n')

    # model = SiameseModel(emb_size=args.emb_size, base_model=base_model, base_model_weights=base_model_weights)
    model = SiameseModel(base_model=base_model, base_model_weights=base_model_weights)

    # model_path = pkg_resources.resource_filename(__name__, "resources/model/data/training_data/")
    trained_model_path = str(Path(__file__).resolve().parent) + "\\latest_model\\newmodel.pth"
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    summary(model, [(3, 228, 228), (3, 228, 228)])
    # model.forward()


import torch


def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()
