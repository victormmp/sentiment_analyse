import torch.nn as nn
from typing import List, Union
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from sklearn.model_selection import KFold, train_test_split
import torch
import numpy as np

from sklearn.metrics import f1_score, balanced_accuracy_score


class Pipeline:

    def __init__(self, epochs: int = 100, splits: int = 5, validation_size: Union[int, float] = None):
        self.train_losses: List = []
        self.validation_losses: List = []
        self.f1_score_validation: List = []
        self.accuracy_validation: List = []
        self.val_size = validation_size
        self.epochs = epochs
        self.splits = splits

    def train(self, model: Module, optimizer: Optimizer, loss_function, x_train, y_train):

        if (self.val_size <= 1.0) and (self.val_size >= 0.0):
            num_val_samples = int(self.val_size * x_train.shape[0])
        elif self.val_size > 1:
            if self.val_size >= x_train.shape[0]:
                raise ValueError(f"Number of validation samples to be used ({self.val_size}) is equal or superior of training set sample count ({x_train.shape[0]}).")
            num_val_samples = int(self.val_size)
        else:
            num_val_samples = 0

        train_index, val_index = np.split(np.arange(x_train.shape[0]), [-num_val_samples])

        class_ratio = 0.5

        if num_val_samples > 0:
            print(f"Using {len(train_index)} samples to train and {len(val_index)} samples as validation set.")
            class_ratio = y_train[y_train == 1].shape[0] / y_train.shape[0]
            print(f"Class ratio: {class_ratio:0.2f} % of the samples within validation set belong to class 1.")

        for epoch in range(self.epochs):
            model.train()

            train_losses = []
            test_losses = []
            f1_scores = []
            accs = []

            val_log = ""

            optimizer.zero_grad()

            pred = model(x_train[train_index, :])
            loss = loss_function(pred.squeeze(), y_train[train_index])
            loss.backward()
            train_loss = loss.item()
            train_losses.append(train_loss)
            optimizer.step()

            if num_val_samples:
                model.eval()

                with torch.no_grad():
                    pred = model(x_train[val_index, :])
                    loss = loss_function(pred.squeeze(), y_train[val_index])
                    test_loss = loss.item()
                    test_losses.append(test_loss)
                    y_pred_bin = [1 if d > class_ratio else 0 for d in pred.squeeze().cpu().numpy()]
                    f1 = f1_score(y_train[val_index].cpu().numpy(), y_pred_bin, average='weighted')
                    acc = balanced_accuracy_score(y_train[val_index].cpu().numpy(), y_pred_bin)
                    f1_scores.append(f1)
                    accs.append(acc)

                avg_test_loss = np.mean(test_losses)
                avg_f1 = np.mean(f1_scores)
                avg_acc = np.mean(accs)
                self.validation_losses.append(avg_test_loss)
                self.f1_score_validation.append(avg_f1)
                self.accuracy_validation.append(avg_acc)

                val_log = f" - val loss: {avg_test_loss:.4f} - val f1 score: {avg_f1:0.4f} - val acc: {avg_acc:.4f}"

            avg_train_loss = np.mean(train_losses)
            self.train_losses.append(avg_train_loss)

            print(f"Epoch {epoch} of {self.epochs} - train loss: {avg_train_loss:.4f}{val_log}.")

        model.eval()
        print(f"Training finished.")

