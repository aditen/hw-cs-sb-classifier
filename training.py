import copy

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from data_handling.data_loader import DataloaderKinderlabor
from grayscale.grayscale_model import CNN


class TrainerKinderlabor:
    def __init__(self, loader: DataloaderKinderlabor):
        self.__loader = loader

    def train_model(self, n_epochs=20):
        train_loader, valid_loader, __ = self.__loader.get_data_loaders()
        n_train, n_valid, __ = self.__loader.get_num_samples()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # initialize model as well as optimizer, scheduler
        model = CNN(n_classes=len(self.__loader.get_classes()))
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # enable training mode (allow batch norm to be adjusted etc.)
        model.train()

        # initialize best model
        best_model = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in tqdm(range(n_epochs), unit="epoch"):
            running_loss = 0.0
            running_corrects = torch.tensor(0).to(device)

            model.train()

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

            model.eval()
            eval_loss, eval_corr = 0., torch.tensor(0).to(device)
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    eval_loss += loss.item() * inputs.size(0)
                    eval_corr += torch.sum(preds == labels.data)

            eval_acc = eval_corr.double() / n_valid

            if eval_acc > best_acc:
                best_acc = eval_acc
                best_model = copy.deepcopy(model.state_dict())

        print("Best model acc", (best_acc * 100).double())
        torch.save(best_model, "best_model.pt")
