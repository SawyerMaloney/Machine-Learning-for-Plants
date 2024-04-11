import sklearn.datasets
import sklearn.neural_network
from sklearn.model_selection import train_test_split
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as f

import matplotlib.pyplot as plt
# h1 = 1000, h2 = 1300, h3 = 900, h4 = 800, h5 = 600, h6 = 500, h7 = 350, h8 = 300, h9 = 150,

class NNClassifier(nn.Module):

    def __init__(self, input_features = 64 , h1 = 300, h2 = 700, h3 = 150, out_features = 100):
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.out(x)

        return x

torch.manual_seed(41)
network = NNClassifier()
name = 'one-hundred-plants-texture'
dataset_texture = sklearn.datasets.fetch_openml(name, parser = 'liac-arff', as_frame = False)
X = dataset_texture.data

# 'shapes' data values look like they are generally 100 times smaller than 'texture' and 'margin'
if name == 'one-hundred-plants-shape':
    print(X[0])
    X_texture = X * 100
    print(X_texture[0])

y = dataset_texture.target

X_train, X_test, y_train, y_test = train_test_split(X_texture, y, test_size=0.2)

X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)

y_train = torch.from_numpy(np.array([ (int(num) - 1) for num in y_train ])).type(torch.LongTensor)
y_test = torch.from_numpy(np.array( [ (int(num) - 1) for num in y_test ] )).type(torch.LongTensor)

#loss function
criterion = nn.CrossEntropyLoss()

# adam optimizer, lr
optimizer = torch.optim.Adam(network.parameters(), lr = 0.01)

# 1 epoch = 1 run of train data thru network
epochs = 1000
losses = []

for i in range(epochs):
    y_pred = network.forward(X_train)

    loss = criterion(y_pred, y_train)

    losses.append(loss.detach().numpy())

    # print every 10 epochs

    if i % 10 == 0:
        print(f'epoch: {i}, loss: {loss}')
    

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

correct_pred = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        pred = network.forward(data)

        print(f'{i + 1}.) {y_test[i]} \t {pred.argmax().item()}')

        if pred.argmax().item() == y_test[i]:
            correct_pred += 1

print(f'test sample correct prediction: {correct_pred} (out of {len(y_test)} samples)')
print(f'out-sample error estimate: {1 - correct_pred / len(y_test)}')

correct_pred = 0
with torch.no_grad():
    for i, data in enumerate(X_train):
        pred = network.forward(data)

        if pred.argmax().item() == y_train[i]:
            correct_pred += 1

print(f'train sample correct predictions: {correct_pred} (out of {len(y_train)} samples)')
print(f'in sample error: {1 - correct_pred / len(y_train)}')