import argparse, random, copy
import os, time
from datetime import datetime
import pickle
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Use ':16:8' if you want to conserve GPU memory

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, densenet121, DenseNet121_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from dataset import *
from model import *


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

    with open(f'{output_folder_path}train_dataset_b{args.batch_size}_n{args.num_samples}.pickle', 'wb+') as handle:
        pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{output_folder_path}val_dataset_b{args.batch_size}_n{args.num_samples}.pickle', 'wb+') as handle:
        pickle.dump(val_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # note: when using a GPU it’s better to set pin_memory=True, this instructs DataLoader to use pinned memory
    # and enables faster and asynchronous memory copy from the host to the GPU.
    trainloader = DataLoader(
        train_dataset,
        num_workers = args.num_workers,
        batch_size = args.batch_size,
        pin_memory=True
    )
    valloader = DataLoader(
        val_dataset,
        num_workers = args.num_workers,
        batch_size = args.batch_size,
        pin_memory=True
    )

    return trainloader, valloader

def train(model, criterion, optimizer, trainloader, valloader, args, device, output_folder_path, lr_scheduler=None):

    best_vloss = 1_000_000.
    train_loss_history = []
    val_loss_history = []

    for epoch in range(args.epochs):
        # training
        model.train(True)

        tloss_history = []
        tcorrect = 0
        ttotal = 0

        for _, data in enumerate(trainloader):
            img0, img1, _, _, label = data
            img0, img1, label = img0.to(device, dtype=torch.float), img1.to(device, dtype=torch.float), label.to(device, dtype=torch.float)

            # starting from PyTorch 1.7, call model or optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)

            output = model(img0, img1).squeeze(1)
            # criterion accepts two torch tensors - one being the raw nn outputs,
            # the other being the true class labels,
            # then wraps the first using sigmoid -
            # for each element in the tensor and then calculates
            # Cross Entropy loss (-(target*log(sigmoid(pred))) for each pair and reduces it to mean.
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

        # disable gradient calculation for validation or inference
        with torch.no_grad():
            for _, vdata in enumerate(valloader):
                vimg0, vimg1, _, _, vlabel = vdata
                vimg0, vimg1, vlabel = vimg0.to(device, dtype=torch.float), vimg1.to(device, dtype=torch.float), vlabel.to(device, dtype=torch.float)

                voutput = model(vimg0, vimg1).squeeze(1)
                vloss = criterion(voutput, vlabel)

                vcorrect += torch.count_nonzero(vlabel == (torch.sigmoid(voutput) > 0.5)).item()
                vtotal += len(vlabel)

                vloss_history.append(vloss.item())

        # Calculate train and val loss
        avg_tloss = np.mean(tloss_history)
        avg_vloss = np.mean(vloss_history)
        avg_tacc = tcorrect / ttotal
        avg_vacc = vcorrect / vtotal

        with open(f'{output_folder_path}progress_e{args.epochs}_b{args.batch_size}_lr{args.lr}_n{args.num_samples}_emb{args.emb_size}.txt', 'a+') as f:
            f.write(f'[{time.ctime()}] [Epoch {epoch}] train loss: {avg_tloss:.5f} val loss: {avg_vloss:.5f} train acc: {avg_tacc:.5f} val acc: {avg_vacc:.5f}\n')

        # Track best performance, and save the model's state
        if epoch > 0 and avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'{output_folder_path}siamese_model_e{epoch}_b{args.batch_size}_lr{args.lr}_num{args.num_samples}_emb{args.emb_size}.pth'
            torch.save(model.state_dict(), model_path)

        train_loss_history.append(tloss_history)
        val_loss_history.append(vloss_history)

        if (epoch + 1) % 5 == 0:
            with open(f'{output_folder_path}training_loss_e{args.epochs}_b{args.batch_size}_lr{args.lr}_n{args.num_samples}_emb{args.emb_size}.txt', 'a+') as f:
                f.write(f'[{time.ctime()}] [Epoch {epoch}] train loss: {avg_tloss:.5f} val loss: {avg_vloss:.5f} train acc: {avg_tacc:.5f} val acc: {avg_vacc:.5f}\n')

            model_path = f'{output_folder_path}siamese_model_e{epoch}_b{args.batch_size}_lr{args.lr}_num{args.num_samples}_emb{args.emb_size}.pth'
            torch.save(model.state_dict(), model_path)

            with open(f'{output_folder_path}train_loss_history_e{epoch}_b{args.batch_size}_lr{args.lr}_n{args.num_samples}_emb{args.emb_size}.pickle', 'wb') as handle:
                pickle.dump(train_loss_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'{output_folder_path}val_loss_history_e{epoch}_b{args.batch_size}_lr{args.lr}_n{args.num_samples}_emb{args.emb_size}.pickle', 'wb') as handle:
                pickle.dump(val_loss_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test():
    pass

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--num-samples', type=int, default=1024, metavar='N',
                        help='size of training samples (default: 1024)')
    parser.add_argument('--emb-size', type=int, default=20, metavar='N',
                        help='size of feature embedding (default: 20)')
    parser.add_argument('--train-val-split', type=float, default=0.8, metavar='LR',
                        help='percentage of train to val in data (default: 0.8)')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N',
                        help='number of workers (default: 1)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--output-folder-name', type=str, default=None, metavar='S',
                        help='output folder name (default: None)')
    parser.add_argument('--lr-scheduler', default=False, type=bool,
                        help='activate lr scheduler (default: False)')
    parser.add_argument('--model', type=str, default='efficientnetv2', metavar='S',
                        help='model (default: resnet18)')


    # perform the configurations
    args = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.output_folder_name != None:
        output_folder_path = os.getcwd() + f'\\output\\{datetime.today().strftime("%d%m%Y")}_{args.output_folder_name}\\'
    else:
        output_folder_path = os.getcwd() + f'\\output\\{datetime.today().strftime("%d%m%Y")}\\'
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    # ensure reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    with open(f'{output_folder_path}progress_e{args.epochs}_b{args.batch_size}_lr{args.lr}_n{args.num_samples}_emb{args.emb_size}.txt', 'a+') as f:
        f.write(f'[{time.ctime()}] getting the train and val dataloaders\n')

    # getting the train and val dataloaders
    data_set_path = str(Path(__file__).resolve().parent.parent) + "\\model\\data\\training_data\\"
    full_dataset = SiameseDataset(
        images_folder_path=data_set_path,
        transform=transforms,
        num_samples=args.num_samples
    )
    trainloader, valloader = dataloader(full_dataset, args, output_folder_path)


    # write the number of classes, their names and the number of samples to a file
    with open(f'{output_folder_path}progress_e{args.epochs}_b{args.batch_size}_lr{args.lr}_n{args.num_samples}_emb{args.emb_size}.txt', 'a+') as f:
        f.write(f'[{time.ctime()}] number of classes: {len(full_dataset.class_names)}\n')
        f.write(f'[{time.ctime()}] class names: {full_dataset.class_names}\n')
        for i in range(len(full_dataset.class_names)):
            f.write(f'[{time.ctime()}] number of samples in class {full_dataset.class_names[i]}: {len(full_dataset.all_images[i])}\n')

        f.write(f'[{time.ctime()}] number of samples: {args.num_samples}\n')

    with open(f'{output_folder_path}progress_e{args.epochs}_b{args.batch_size}_lr{args.lr}_n{args.num_samples}_emb{args.emb_size}.txt', 'a+') as f:
        f.write(f'[{time.ctime()}] setting up training parameters\n')

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
    elif args.model == 'densenet121':
        base_model = densenet121
        base_model_weights = DenseNet121_Weights.IMAGENET1K_V1
    else: # args.model == 'mobilenetv2'
        base_model = mobilenet_v2
        base_model_weights = MobileNet_V2_Weights.IMAGENET1K_V1


    with open(f'{output_folder_path}progress_e{args.epochs}_b{args.batch_size}_lr{args.lr}_n{args.num_samples}_emb{args.emb_size}.txt', 'a+') as f:
        f.write(f'[{time.ctime()}] siamese network will be based off {args.model}\n')

    # model = SiameseModel(emb_size=args.emb_size, base_model=base_model, base_model_weights=base_model_weights)
    model = SiameseModel(base_model=base_model, base_model_weights=base_model_weights)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    if args.lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    else:
        lr_scheduler = None

    with open(f'{output_folder_path}progress_e{args.epochs}_b{args.batch_size}_lr{args.lr}_n{args.num_samples}_emb{args.emb_size}.txt', 'a+') as f:
        f.write(f'[{time.ctime()}] training the model\n')

    # model training
    train(model, criterion, optimizer, trainloader, valloader, args, device, output_folder_path, lr_scheduler)

    with open(f'{output_folder_path}progress_e{args.epochs}_b{args.batch_size}_lr{args.lr}_n{args.num_samples}_emb{args.emb_size}.txt', 'a+') as f:
        f.write(f'[{time.ctime()}] done training the model\n')

if __name__ == '__main__':
    main()