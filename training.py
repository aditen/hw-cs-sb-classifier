import copy
import math
import os
from enum import Enum

import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

import torch.nn.functional as F

from data_augmentation import DataAugmentationUtils
from data_loading import DataloaderKinderlabor
from grayscale_model import ModelVersion, get_model
from open_set_loss import EntropicOpenSetLoss, ObjectosphereLoss


class Optimizer(Enum):
    ENTROPIC = "ENTROPIC"
    OBJECTOSPHERE = "OBJECTOSPHERE"
    SOFTMAX = "SOFTMAX"


class TrainerKinderlabor:
    def __init__(self, loader: DataloaderKinderlabor, load_model_from_disk=True, run_id=None,
                 model_version: ModelVersion = ModelVersion.LG, optimizer: Optimizer = None):
        self.__model_dir = f'output_visualizations/{run_id if run_id is not None else loader.get_folder_name()}'
        if not os.path.isdir(self.__model_dir):
            os.mkdir(self.__model_dir)
        self.__model_version = model_version
        self.__loader = loader
        self.__load_model_from_disk = load_model_from_disk
        self.__epochs, self.__train_loss, self.__valid_loss, self.__train_acc, self.__valid_acc = [], [], [], [], []
        self.__test_actual, self.__test_predicted, self.__2d, self.__best_probs, self.__err_samples, self.__f1 = \
            [], [], [], [], [], math.nan
        self.__optimizer = optimizer
        self.__model_path = f"{self.__model_dir}/model.pt"

    def train_model(self, n_epochs=30, lr=0.01, sched=(10, 0.5), n_epochs_wait_early_stop=5):
        if self.__load_model_from_disk and os.path.isfile(self.__model_path):
            print("Found model already on disk. Set load_model_from_disk=False on function call to force training!")
            return

        train_loader, valid_loader, _ = self.__loader.get_data_loaders()
        n_train, n_valid, __ = self.__loader.get_num_samples()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # initialize model as well as optimizer, scheduler
        model = get_model(num_classes=len(self.__loader.get_classes()), model_version=self.__model_version)
        model = model.to(device)
        if self.__optimizer == Optimizer.SOFTMAX or self.__optimizer is None:
            criterion = nn.CrossEntropyLoss(reduction='mean')
        elif self.__optimizer == Optimizer.ENTROPIC:
            criterion = EntropicOpenSetLoss(num_of_classes=len(self.__loader.get_classes()))
        elif self.__optimizer == Optimizer.OBJECTOSPHERE:
            criterion = ObjectosphereLoss(num_of_classes=len(self.__loader.get_classes()))
        else:
            raise ValueError(f"Unsupported optimizer option: {self.__optimizer}")
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=sched[0], gamma=sched[1])

        # enable training mode (allow batch norm to be adjusted etc.)
        model.train()

        # initialize best model
        best_model = copy.deepcopy(model.state_dict())
        best_loss = math.inf
        best_acc = 0.

        epochs = []
        losses_train, acc_train, losses_valid, acc_valid = [], [], [], []
        n_epochs_no_improvement = 0

        for epoch_i in tqdm(range(n_epochs), unit="epoch"):
            if n_epochs_no_improvement > n_epochs_wait_early_stop:
                break
            epochs.append(epoch_i + 1)
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
                outputs, outputs_2d = model(inputs)
                _, preds = torch.max(outputs, 1)
                if isinstance(criterion, ObjectosphereLoss):
                    loss = criterion(outputs, labels, outputs_2d, reduction='mean')
                elif isinstance(criterion, EntropicOpenSetLoss):
                    loss = criterion(outputs, labels, reduction='mean')
                else:
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
                    outputs, outputs_2d = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    if isinstance(criterion, ObjectosphereLoss):
                        loss = criterion(outputs, labels, outputs_2d, reduction='mean')
                    elif isinstance(criterion, EntropicOpenSetLoss):
                        loss = criterion(outputs, labels, reduction='mean')
                    else:
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
                n_epochs_no_improvement = 0
            else:
                n_epochs_no_improvement += 1

        if len(epochs) < n_epochs:
            print(f'Early stopping criterion reached in epoch {len(epochs)}')

        print(f"Best model accuracy: {(best_acc * 100):.2f}%")
        torch.save(best_model, self.__model_path)

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
        model = get_model(num_classes=len(self.__loader.get_classes()), model_version=self.__model_version).to(device)
        model.load_state_dict(torch.load(self.__model_path))
        model.eval()

        test_loss, test_corr = 0., torch.tensor(0).to(device)

        if self.__optimizer == Optimizer.SOFTMAX or self.__optimizer is None:
            criterion = nn.CrossEntropyLoss(reduction='mean')
        elif self.__optimizer == Optimizer.ENTROPIC:
            criterion = EntropicOpenSetLoss(num_of_classes=len(self.__loader.get_classes()))
        elif self.__optimizer == Optimizer.OBJECTOSPHERE:
            criterion = ObjectosphereLoss(num_of_classes=len(self.__loader.get_classes()))
        else:
            raise ValueError(f"Unsupported optimizer option: {self.__optimizer}")

        actual, predicted = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, unit="test batch"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs, outputs_2d = model(inputs)
                probs, _ = torch.max(F.softmax(outputs, dim=1), dim=1)
                probs = probs.tolist()
                _, preds = torch.max(outputs, 1)
                if isinstance(criterion, ObjectosphereLoss):
                    loss = criterion(outputs, labels, outputs_2d, reduction='mean')
                elif isinstance(criterion, EntropicOpenSetLoss):
                    loss = criterion(outputs, labels, reduction='mean')
                else:
                    loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                test_corr += torch.sum(preds == labels.data)
                actual_batch = labels.cpu().numpy().flatten().tolist()
                actual += actual_batch
                predicted_batch = preds.flatten().cpu().numpy().tolist()
                predicted += predicted_batch
                for i in range(len(actual_batch)):
                    self.__2d.append(outputs_2d[i, :].cpu().numpy().flatten().tolist())
                    self.__best_probs.append(probs[i])
                    if actual_batch[i] != predicted_batch[i] and actual_batch[i] != -1:
                        self.__err_samples.append((inputs[i, :, :].cpu().numpy(), actual_batch[i], predicted_batch[i]))

        self.__test_actual = actual
        self.__test_predicted = predicted
        test_acc = test_corr.double() / n_test
        test_loss = test_loss / n_test

        actual_without_uu, predicted_without_uu = zip(*((ac, pr) for ac, pr in zip(actual, predicted) if ac != -1))

        f1 = f1_score(actual_without_uu, predicted_without_uu, average='macro')
        self.__f1 = f1
        print(
            f'Test Accuracy: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}, Macro-average F1 Score: {f1 * 100:.2f}%')

    def get_predictions(self):
        return self.__test_actual, self.__test_predicted, self.__best_probs, self.__err_samples, self.__2d, self.__loader

    def script_model(self):
        model = get_model(num_classes=len(self.__loader.get_classes()), model_version=self.__model_version)
        model.load_state_dict(torch.load(self.__model_path))
        model.eval()
        model_including_transforms = nn.Sequential(
            DataAugmentationUtils.get_scriptable_augmentation(self.__loader.get_augmentation_options()),
            model)
        scripted = torch.jit.script(model_including_transforms)
        scripted.save(f"{self.__model_path}.scripted")
        synset_path = f'{self.__model_dir}/synset.txt'
        if os.path.isfile(synset_path):
            os.remove(synset_path)
        with open(f'{self.__model_dir}/synset.txt', 'w', encoding="utf-8") as f:
            f.writelines("\n".join(self.__loader.get_classes()))
        print(f'Scripted model and written synset')
