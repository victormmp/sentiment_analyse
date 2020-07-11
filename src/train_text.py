import json
import os
from collections import Counter
from os.path import join

from sklearn.decomposition import PCA
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam

from src.model import MLP
from src.normalizer import Normalizer
from src.pipelines import Pipeline
from src.text_tokenize import TextTokenizer

click.echo('Starting script')

LOAD_WORDMAP = True

#########################################################################
#           LOADING DATASET                                             #
#########################################################################

click.echo("Loading data frame")
try:
    df = pd.read_csv(join(".", 'data', 'processed', 'review_message.csv'))
except FileNotFoundError:
    df = pd.read_csv(join("..", 'data', 'processed', 'review_message.csv'))

click.echo(f'Data frame loaded with {df.shape[0]} rows.')

click.echo('Cleaning missing data')

df["message"] = df["message"].apply(lambda d: "none" if type(d) == float else d)
click.echo(f"Number of float values {df[df.message.apply(type) == float].shape[0]}")

click.echo(f"Drop empty messages.")
df = df[df['message'] != 'none']

click.echo('Splitting into train / test data frames')
X = df.drop(['score'], axis=1)
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

class_ratio = y_train[y_train == 1].shape[0] / y_train.shape[0]

click.echo(f"Class balance ratio : {class_ratio:0.2f}% of data is class 1.")

click.echo(f'Train size {X_train.shape[0]} - Test size {X_test.shape[0]}')


#########################################################################
#           TOKENIZING TEXT                                             #
#########################################################################

if os.path.isfile('word_map.json') and LOAD_WORDMAP:
    click.secho("Loading word map from word_map.json file", fg='green')
    with open('word_map.json') as fp:
        map = json.load(fp)
        tk = TextTokenizer().load(map)
else:
    click.echo('Generating word map')
    tk = TextTokenizer(X_train, "message", use_gpu=True, n_tokens=200)

    file_name = 'word_map.json'
    with open(file_name, 'w') as fp:
        click.echo(f'Saving a copy of the word map generated in {file_name}')
        json.dump(tk.map, fp)

click.echo(f'Tokenizing datasets')

X_train['message'] = X_train['message'].apply(lambda d: [tk.get(w) for w in d.split()])
X_test['message'] = X_test['message'].apply(lambda d: [tk.get(w) for w in d.split()])

click.echo('Generating sparse features per token')

sparse_columns = [k for k in range(tk.n_tokens)]

new_dataset = []
with click.progressbar(X_train.iterrows(), length=X_train.shape[0], label="Generating sparse train dataset") as bar:
    for index, row in bar:
        new_dataset += [dict(Counter(row['message']))]

X_train = pd.DataFrame(new_dataset, columns=sparse_columns)


new_dataset = []
with click.progressbar(X_test.iterrows(), length=X_test.shape[0], label="Generating sparse test dataset") as bar:
    for index, row in bar:
        new_dataset += [dict(Counter(row['message']))]

X_test = pd.DataFrame(new_dataset, columns=sparse_columns)

del sparse_columns, new_dataset

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

click.echo(f"Applying PCA")

pca = PCA(n_components=0.95)

pca.fit(X_train)
n_components = max(2, pca.n_components_)
pca = PCA(n_components=n_components)
X_train = pca.fit_transform(X_train)

click.echo(f'Number of components obtained through PCA for 95% variance: {n_components}')

X_test = pca.transform(X_test)

click.echo(f"Normalizing data for neural network.")

normalizer = Normalizer()

X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

click.echo(f"Building model")

device = torch.device("cuda")

model = MLP(n_components)
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
loss_function = nn.BCELoss()
# loss_function = nn.CrossEntropyLoss()

pipeline = Pipeline(epochs=500, validation_size=0.05)

click.echo(f"Initiating training pipeline")

X_train_torch = torch.from_numpy(X_train).float().to(device)
y_train_torch = torch.from_numpy(y_train.to_numpy()).float().to(device)

X_test_torch = torch.from_numpy(X_test).float().to(device)
y_test_torch = torch.from_numpy(y_test.to_numpy()).float().to(device)

pipeline.train(
    model=model,
    optimizer=optimizer,
    loss_function=loss_function,
    x_train=X_train_torch,
    y_train=y_train_torch
)

click.echo("Save model")

torch.save(model.state_dict(), join(".", "src", "models", "model_1.pt"))

#########################################################################
#       PLOT LOSSES                                                     #
#########################################################################

fig, ax1 = plt.subplots(figsize=(15, 10))
ax1.plot(pipeline.train_losses, label='train')
ax1.plot(pipeline.validation_losses, label='validation')
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='best')
ax1.set(title='Loss')
# plt.show()

plt.savefig("training_loss.png")

plt.close('all')


fig, ax1 = plt.subplots(figsize=(15, 10))
ax1.plot(pipeline.f1_score_validation, label='F1 on validation')
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='best')
ax1.set(title='F1 on Validation')
# plt.show()

plt.savefig("validation_f1.png")
plt.close('all')


fig, ax1 = plt.subplots(figsize=(15, 10))
ax1.plot(pipeline.f1_score_validation, label='Accuracy on validation')
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='best')
ax1.set(title='Accuracy on Validation')
# plt.show()

plt.savefig("validation_acc.png")
plt.close('all')


if n_components == 2:
    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax1.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label='bad scores')
    ax1.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label='good scores')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')
    ax1.set(title='Data distribution')
    # plt.show()

    x_lab = np.linspace(-5, 5, num=1000)
    y_lab = np.linspace(-5, 5, num=1000)
    x1, x2 = np.meshgrid(x_lab, y_lab)
    x_grid = np.transpose(np.vstack([x1.flatten(), x2.flatten()]))

    with torch.no_grad():
        x_grid_torch = torch.from_numpy(x_grid).float().to(device)
        pred = model(x_grid_torch)
        z = pred.squeeze().cpu().numpy()
    z = z.reshape([1000, 1000])
    ax1.contour(x1, x2, z, linewidths=(2.5,))

    plt.savefig("dist.png")
    plt.close('all')

    del pred


#########################################################################
#       CALCULATE ON TEST DATA                                          #
#########################################################################

with torch.no_grad():
    pred = model(X_test_torch)
    y_pred_bin = [1 if d > class_ratio else 0 for d in pred.squeeze().cpu().numpy()]

score = f1_score(y_test_torch.cpu().numpy(), y_pred_bin, average='weighted')
auc = roc_auc_score(y_test_torch.cpu().numpy(), y_pred_bin, average='weighted')
print(classification_report(y_test_torch.cpu().numpy(), y_pred_bin))


click.secho(f"F1 score on test data: {score} - auc: {auc}", fg='green')