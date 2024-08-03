import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

# Define the neural network
class SCNet(nn.Module):
        """
        A simple neural network that trains on single-cell gene expression data to classify cell types.
        """

        def __init__(self, input_size, num_classes, embed_size=128, activation=nn.SiLU, num_hidden_layers=2, dropout=0.2, momentum=0.9, lr=3e-4):
                """
                input_size: the number of observed features in the input data
                embed_size: the dimensionality of the latent space used in the feed forward layers
                num_classes: the number of cell types to classify across
                num_hidden_layers: the number of hidden layers in the neural network
                dropout: the dropout rate, we use the same dropout rate for all layers for now
                """
                super(SCNet, self).__init__()

                assert num_hidden_layers >= 1, "we need at least one hidden layer"

                self.module1 = nn.Sequential(
                        nn.Linear(input_size, embed_size),
                        nn.Dropout(dropout),
                        activation(),
                )

                self.module_list = nn.ModuleList([
                        nn.Sequential(
                                nn.BatchNorm1d(embed_size),
                                nn.Linear(embed_size, embed_size),
                                nn.Dropout(dropout),
                                activation(),
                        ) for _ in range(num_hidden_layers - 1)
                ])

                self.clf_head = nn.Linear(embed_size, num_classes)
                self.criterion = nn.CrossEntropyLoss()

                # Interestingly, using Adam here resulted in worse model performance and overfitting.
                self.optim = optim.SGD(self.parameters(), momentum=momentum, lr=lr)

        
        def forward(self, x):
            """
            Computes a forward pass of the model.

            x: shape should be (B, E), where B is batch dimension and E is embed_size
            """

            x = self.module1(x)
            for module in self.module_list:
                    x = x + module(x)
            x = self.clf_head(x)

            return x

        def train_step(self, x_train, y_train):
            """
            Computes one training step of the model, given batch x_train. This function returns the training loss value on the batch x_train.

            x_train: shape should be (B, E)
            """

            self.train()
            self.optim.zero_grad()
            outputs = self.forward(x_train)
            loss = self.criterion(outputs, y_train)

            # We are keeping this simple for now; but we can add schedulers and such to warmup or adapt the lr in the future.
            loss.backward()
            self.optim.step()
            return loss.item()

        @torch.no_grad()
        def eval_step(self, x_test, y_test):
            self.eval()
            outputs = self.forward(x_test)
            test_loss = self.criterion(outputs, y_test)
            return test_loss.item()

        @torch.no_grad()
        def classify(self, x):
            self.eval()
            outputs = self.forward(x)
            _, predictions = torch.max(outputs, 1)
            return predictions
