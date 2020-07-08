import json
from os.path import join

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from src.text_tokenize import TextTokenizer


click.echo('Starting script')

click.echo("Loading dataframe")
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

click.echo('Generating word map')
tk = TextTokenizer(X_train, "message", use_gpu=True)

file_name = 'word_map.json'
with open(file_name, 'w') as fp:
    click.echo(f'Saving a copy of the word map generated in {file_name}')
    json.dump(tk.map, fp)

click.echo(f'Tokenizing datasets')

X_train['message'] = X_train['message'].apply(lambda d: [tk.get(w) for w in d.split()])
X_test['message'] = X_test['message'].apply(lambda d: [tk.get(w) for w in d.split()])


