import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorboardX
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertTokenizer, BertModel
from model import EFR_TX
from tqdm import tqdm
import yaml

from utils import *

class Trainer(nn.Module):
    def __init__(self, model, optimizer, criterion, device=None):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device if device else CONFIG['device']

    def train(self, train_loader, max_epochs):
        self.model.train()
        running_loss = 0.0
        for epoch in range(max_epochs):  # loop over the dataset multiple times
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                breakpoint()
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if batch_idx % 100 == 99:    # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, batch_idx + 1, running_loss / 100))
                    running_loss = 0.0

    def test(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    def save(self, path=None):
        if not path:
            path = CONFIG['erc_model_path']
        torch.save(self.model.state_dict(), path)

    def load(self, path=None):
        if not path:
            path = CONFIG['erc_model_path']
        self.model.load_state_dict(torch.load(path))