import torch.nn as nn
from typing import List, Union
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from sklearn.model_selection import KFold
import torch
import numpy as np

from sklearn.metrics import f1_score, balanced_accuracy_score


class Pipeline:

    def __init__(self, epochs: int = 100, splits: int = 5):
        self.train_losses: List = []
        self.validation_losses: List = []
        self.f1_score_validation: List = []
        self.accuracy_validation: List = []
        self.epochs = epochs
        self.splits = splits

    def train(self, model: Module, optimizer: Optimizer, loss_function, X_train, y_train):

        for epoch in range(self.epochs):
            model.train()
            kf = KFold(n_splits=self.splits)

            train_losses = []
            test_losses = []
            f1_scores = []
            accs = []
            for train_index, test_index in kf.split(X_train):
                optimizer.zero_grad()

                pred = model(X_train[train_index, :])
                loss = loss_function(pred.squeeze(), y_train[train_index])
                loss.backward()
                train_loss = loss.item()
                train_losses.append(train_loss)
                optimizer.step()

                model.eval()

                with torch.no_grad():
                    pred = model(X_train[test_index, :])
                    loss = loss_function(pred.squeeze(), y_train[test_index])
                    test_loss = loss.item()
                    test_losses.append(test_loss)
                    y_pred_bin = [1 if d > 0.5 else 0 for d in pred.squeeze().cpu().numpy()]
                    f1 = f1_score(y_train[test_index].cpu().numpy(), y_pred_bin, average='weighted')
                    acc = balanced_accuracy_score(y_train[test_index].cpu().numpy(), y_pred_bin)
                    f1_scores.append(f1)
                    accs.append(acc)

            avg_train_loss = np.mean(train_losses)
            avg_test_loss = np.mean(test_losses)
            avg_f1 = np.mean(f1_scores)
            avg_acc = np.mean(accs)
            self.train_losses.append(avg_train_loss)
            self.validation_losses.append(avg_test_loss)
            self.f1_score_validation.append(avg_f1)
            self.accuracy_validation.append(avg_acc)

            print(f"Epoch {epoch} of {self.epochs} - train loss: {avg_train_loss:.4f} - test loss: {avg_test_loss:.4f} - test f1 score: {avg_f1:0.4f} - acc: {avg_acc:.4f}")

        model.eval()
        print(f"Training finished.")

