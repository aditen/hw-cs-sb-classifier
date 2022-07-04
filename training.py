import copy
import math

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import os

from data_loading import DataloaderKinderlabor
from grayscale_model import CNN


class TrainerKinderlabor:
    def __init__(self, loader: DataloaderKinderlabor, load_model_from_disk=True):
        self.__loader = loader
        self.__load_model_from_disk = load_model_from_disk
        self.__epochs, self.__train_loss, self.__valid_loss, self.__train_acc, self.__valid_acc = [], [], [], [], []
        self.__test_actual, self.__test_predicted = [], []

    def train_model(self, n_epochs=20):
        if self.__load_model_from_disk and os.path.isfile(
                f"best_model_{'all' if self.__loader.get_task_type() is None else self.__loader.get_task_type()}.pt"):
            print("Found model already on disk. Set load_model_from_disk=False on function call to force training!")
            return

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
        best_loss = math.inf
        best_acc = 0.

        epochs = [i + 1 for i in range(n_epochs)]
        losses_train, acc_train, losses_valid, acc_valid = [], [], [], []

        for _ in tqdm(range(n_epochs), unit="epoch"):
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
            losses_train.append(epoch_loss)
            epoch_acc = running_corrects.double() / n_train
            acc_train.append(epoch_acc.item())

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
            acc_valid.append(eval_acc.item())
            losses_valid.append(eval_loss / n_valid)

            if eval_loss < best_loss:
                best_loss = eval_loss
                best_acc = eval_acc.item()
                best_model = copy.deepcopy(model.state_dict())

        print("Best model accuracy", best_acc * 100)
        torch.save(best_model,
                   f"best_model_{'all' if self.__loader.get_task_type() is None else self.__loader.get_task_type()}.pt")

        self.__epochs = epochs
        self.__train_loss = losses_train
        self.__valid_loss = losses_valid
        self.__train_acc = acc_train
        self.__valid_acc = acc_valid

    def get_training_progress(self):
        return self.__epochs, self.__train_loss, self.__valid_loss, self.__train_acc, self.__valid_acc

    def predict_on_test_samples(self):
        _, __, test_loader = self.__loader.get_data_loaders()
        _, __, n_test = self.__loader.get_num_samples()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CNN(n_classes=len(self.__loader.get_classes())).to(device)
        model.load_state_dict(torch.load(
            f"best_model_{'all' if self.__loader.get_task_type() is None else self.__loader.get_task_type()}.pt"))

        test_loss, test_corr = 0., torch.tensor(0).to(device)
        criterion = nn.CrossEntropyLoss()

        actual, predicted = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, unit="test batch"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                test_corr += torch.sum(preds == labels.data)
                actual += labels.cpu().numpy().flatten().tolist()
                predicted += preds.flatten().cpu().numpy().tolist()

        self.__test_actual = actual
        self.__test_predicted = predicted
        test_acc = test_corr.double() / n_test
        test_loss = test_loss / n_test
        print(f'Test Accuracy: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}')

    def get_predictions(self):
        return self.__test_actual, self.__test_predicted, self.__loader
