import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from grayscale.dataset_handling import create_grayscale_datasets, imshow
from grayscale.grayscale_model import CNN

train = True
predict = True
visualize_test_images = False

if __name__ == "__main__":
    # create datasets (no validation set atm)
    train_loader, test_loader, n_train, n_test, class_names = create_grayscale_datasets(visualize_samples=train)
    print("created sets")

    if train:
        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # initialize model (atm 4 classes, more tasks to come) as well as optimizer, scheduler
        model = CNN(n_classes=4)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # enable training mode (allow batch norm to be adjusted etc.)
        model.train()

        # training loop
        since = time.time()
        n_epochs = 20

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(n_epochs):
            print(f'Epoch {epoch + 1}/{n_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            # for phase in ['train','val']:

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            scheduler.step()

            epoch_loss = running_loss / n_train
            epoch_acc = running_corrects.double() / n_train

            print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model, "best_model.pt")

    if predict:
        print("Predicting")
        model = torch.load("best_model.pt").cpu()
        model.eval()
        images_so_far = 0
        n_correct = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                n_correct += torch.sum(preds == labels).item()

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    if visualize_test_images:
                        ax = plt.subplot(n_test // 2, 2, images_so_far)
                        ax.axis('off')
                        ax.set_title(f'predicted: {class_names[preds[j]]}')
                        imshow(inputs.cpu().data[j])

        print(
            f'Correctly guessed: {n_correct} out of {images_so_far} which equals {n_correct * 100 / images_so_far:.2f}%')
