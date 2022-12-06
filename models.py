import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from torchmetrics import Accuracy

class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            # nn.Conv1d(1, 50, 4),
            # nn.Conv1d(50, 100, 4),
            # nn.MaxPool1d(4), #output size = 300
            nn.Flatten(),
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Dropout(.5),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Dropout(.5),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(.5),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        return self.model(x)


class Classifier(nn.Module):
    def __init__(self, train_dataloader, test_dataloader, model, device='cpu', epochs=200):
        super().__init__()
        self.acc = Accuracy()
        self.train_acc = 0
        self.test_acc = 0
        # input_size = np.prod(next(iter(train_dataloader))[0][0].shape)
        self.model = model
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._device = device
        self._epochs = epochs

    def train(self, alpha=0.0001, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        self.model.to(self._device)
        for epoch in range(1, self._epochs):
            epoch_loss = 0
            epoch_acc = 0
            num_samples = 0
            proper_preds = 0
            for X, y in self._train_dataloader:
                optimizer.zero_grad()
                y = y.type(torch.LongTensor)
                X, y = X.to(self._device), y.to(self._device)
                y_hat = self.model(X)
                params = torch.cat([x.view(-1) for x in self.model.parameters()])
                l2_regularization = alpha * torch.norm(params, 2) ** 2
                loss = criterion(y_hat, y) + l2_regularization
                loss.backward()
                optimizer.step()

                proper_preds += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().numpy()
                num_samples += y.size()[0]
                acc = torch.mean((torch.argmax(y_hat, dim=1) == y).float())
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                # if epoch % 20 == 0:
                #     print(f'Epoch {epoch} acc is {epoch_acc/ len(self._train_dataloader)}')

    def eval(self):
        self.model.eval()
        proper_preds_train = 0
        num_samples_train = 0
        for X, y in self._train_dataloader:
            X, y = X.to(self._device), y.to(self._device)
            y_hat = self.model(X)
            num_samples_train += y.size()[0]
            proper_preds_train += torch.sum((torch.argmax(y_hat, dim=1) == y).float()).item()

        self.train_acc = proper_preds_train / num_samples_train


        proper_preds = 0
        num_samples = 0
        for X, y in self._test_dataloader:
            X, y = X.to(self._device), y.to(self._device)
            y_hat = self.model(X)
            num_samples += y.size()[0]
            proper_preds += torch.sum((torch.argmax(y_hat, dim=1) == y).float()).item()

        self.test_acc = proper_preds / num_samples
        print(f'Classifier training has finished with acc: {self.train_acc:.3f}')
        print(f'Test acc: {self.test_acc:.3f}')

    def predict(self, dataloader):
        self.model.eval()
        preds = []
        gt = []
        for X, y in dataloader:
            X, y = X.to(self._device), y.to(self._device)
            self.model.to(self._device)
            #y_hat = self.model(X)
            #preds.append(y_hat)#torch.argmax(y_hat, dim=1).numpy())
            gt.append(y.numpy())
        return preds, gt

    def train_and_eval(self, alpha=0.001):
        self.train(alpha)
        self.eval()
        return self.test_acc, self.train_acc

    def pred(self, dataloader):
        self.model.eval()
        self.model.to(self._device)
        softmax = nn.Softmax(dim=1)
        preds = []
        gt = []
        for X, y in dataloader:
            y_hat = softmax(self.model(X.to(self._device)))[:, 1]
            preds.append(y_hat)
            gt.append(y.numpy())

        return torch.concat(preds).cpu().detach().numpy(), gt


def learn_regressor(data_loader_train, device):
    class mlp_r(nn.Module):
        def __init__(self):
            super(mlp_r, self).__init__()
            self.l1 = nn.LazyLinear(128)
            self.l2 = nn.Linear(128, 64)
            self.l3 = nn.Linear(64, 32)
            self.l4 = nn.Linear(32, 1)

            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()

        def forward(self, x):
            x = self.flatten(x)
            x = self.relu(self.l1(x))
            x = self.relu(self.l2(x))
            x = self.relu(self.l3(x))
            x = self.relu(self.l4(x))
            return x

    input_size = np.prod(next(iter(data_loader_train))[0].shape)

    regressor = mlp_r()
    criterion = nn.MSELoss()
    opt = optim.Adam(regressor.parameters(), lr=0.001)

    regressor.train()
    regressor.to(device)

    epochs = 200

    for epoch in range(1, epochs):
        epoch_loss = 0
        epoch_acc = 0
        for X, y in data_loader_train:
            opt.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = regressor(X)
            loss = criterion(y_hat, y.unsqueeze(1))
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

    print(f'Regressor learning has finished with loss: {epoch_loss / len(data_loader_train):.5f}')

    return regressor


def get_preds_regressor(dataloader, model, device):
    model.eval()

    y_preds = list()
    y_true = list()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y_hat = model(X).cpu().numpy()
            y_preds.append(y_hat)

    return np.concatenate(y_preds, axis=0).squeeze()
