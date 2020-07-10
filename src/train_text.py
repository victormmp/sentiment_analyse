import json
from os.path import join
from collections import Counter

import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from src.text_tokenize import TextTokenizer
from src.normalizer import Normalizer
from src.model import MLP
from src.pipelines import Pipeline

from torch.optim import Adam
import torch.nn as nn

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score


click.echo('Starting script')

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

click.echo('Splitting into train / test data frames')
X = df.drop(['score'], axis=1)
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

click.echo(f'Train size {X_train.shape[0]} - Test size {X_test.shape[0]}')


#########################################################################
#           TOKENIZING TEXT                                             #
#########################################################################

click.echo('Generating word map')
tk = TextTokenizer(X_train, "message", use_gpu=True)

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
        new_dataset += dict(Counter(row['message']))

X_train = pd.DataFrame(new_dataset, columns=sparse_columns)


new_dataset = []
with click.progressbar(X_test.iterrows(), length=X_test.shape[0], label="Generating sparse test dataset") as bar:
    for index, row in bar:
        new_dataset += dict(Counter(row['message']))

X_test = pd.DataFrame(new_dataset, columns=sparse_columns)

del sparse_columns, new_dataset

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

click.echo(f"Applying PCA")

pca = PCA(n_components=0.9)

X_train = pca.fit_transform(X_train)
n_components = pca.n_components_

click.echo(f'Number of components obtained through PCA for 90% variance: {n_components}')

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

pipeline = Pipeline()

click.echo(f"Initiating training pipeline")

pipeline.train(
    model=model,
    optimizer=optimizer,
    loss_function=loss_function,
    X_train=X_train,
    y_train=y_train
)

click.echo("Save model")

torch.save(model.state_dict(), join(".", "src", "models", "model_1.pt"))

#########################################################################
#       PLOT LOSSES                                                     #
#########################################################################

plt.figure()
fig, ax1 = plt.subplots(figsize=(15, 10))
ax1.plot(pipeline.train_losses, label='train')
ax1.plot(pipeline.validation_losses, label='validation')
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='best')
ax1.set(title='Loss')
plt.show()

plt.savefig("training_loss.png")

plt.close('all')


#########################################################################
#       CALCULATE ON TEST DATA                                          #
#########################################################################

pred = model(X_test)

score = f1_score(y_test, pred)


print(f"F1 score on test data: {score}")