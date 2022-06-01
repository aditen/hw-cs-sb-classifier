import copy
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

# Data augmentation and normalization for training
# Just normalization for validation
# transforms.RandomHorizontalFlip(),

data_transforms = {
    'train': transforms.Compose([
        # transforms.Resize(224),
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        # transforms.RandomRotation(45),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# This is just copied from the finetuning classification example of PyTorch
# Obviously it does not make too much sense because e. g. using train as test set as well etc
# but it is just a source to know how to load data, visualize examples etc.
# Will be removed later on
if __name__ == "__main__":
    print("hello world", torch.cuda.is_available(), torch.version.cuda)
    df = pd.read_csv("C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/dataset.csv",
                     delimiter=";", index_col="id")
    labels = df['label'].unique().tolist()
    print("found labels:", labels)

    for label in labels:
        if not os.path.isdir("C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/" + label):
            os.mkdir("C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/" + label)
            print("made directory for label", label)
    for idx, row in df.iterrows():
        shutil.copy("C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/" + str(idx) + ".jpeg",
                    "C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/" + row['label'] + "/" + str(
                        idx) + ".jpeg")
        print("Copied row", idx, "to folder", row['label'])
    image_folder_train = ImageFolder("C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data",
                                     data_transforms['train'])
    print(image_folder_train)

    dataloader_train = torch.utils.data.DataLoader(image_folder_train, batch_size=2,
                                                   shuffle=True, num_workers=2)
    class_names = image_folder_train.classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


    # Get a batch of training data
    inputs, classes = next(iter(dataloader_train))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])


    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            #for phase in ['train','val']:
            for phase in ['train']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloader_train:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(image_folder_train)
                epoch_acc = running_corrects.double() / len(image_folder_train)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                #if phase == 'val' and epoch_acc > best_acc:
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader_train):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {class_names[preds[j]]}')
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)


    model_ft = models.resnet18(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False

    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=10)

    visualize_model(model_ft)
