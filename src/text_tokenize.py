import json
from collections import Counter
from os.path import join
from typing import List, AnyStr, Union, Dict

import click
import numpy as np
import pandas as pd
import spacy
import unidecode as ud
from sklearn.model_selection import train_test_split


class TextTokenizer:
    """
    Receives a document or a document dataset and creates a mapping of words to numerical tokens, based on occurrence.
    """

    def __init__(
            self,
            data: Union[pd.DataFrame, str] = None,
            column_name: str = None,
            n_tokens: int = 100,
            use_gpu: bool = False
    ):
        """
        Constructor method of the Tokenizer object.

        Parameters
        ----------
        data: Union[pd.DataFrame, str]
            The document or dataset to be analysed.
        column_name:
            If the document is a pandas data frame, this parameter  indicates the name of the column with the documents.
        n_tokens: int
            The number of valid words to create the mapping with.
        use_gpu: bool
            Allow GPU use for spaCy package.
        """

        if data:

            self.map: Dict = {}
            self.n_tokens = n_tokens

            if use_gpu:
                click.secho("Using GPU for spacy.", fg='yellow')
                spacy.prefer_gpu()

            self.nlp = spacy.load("pt_core_news_sm")
            words = []
            if isinstance(data, pd.DataFrame):
                words = self._from_df(data, column_name)
            elif isinstance(data, str):
                words = self._from_string(data)
            else:
                raise ValueError(f"Not recognized type {type(data)}.")

            self.fit(words)

    def clean_string(self, data: str) -> List[AnyStr]:
        """
        Receives a string phrase and return a list of words from it, considering Nouns, Adverbs, Adjectives and Verbs.
        This method uses spaCy package to parse and identify words. For each word listed, it returns its lower case
        lemma, with accentuation removed (process called normalization of a string).

        Parameters
        ----------
        data: str
            The phrase to be parsed.

        Returns
        -------
        List[AnyStr]:
            List of the normalized lower case lemma for each word.
        """

        doc = self.nlp(data)
        clean_words = []
        for token in doc:
            if token.pos_ not in ["NOUN", "ADV", "ADJ", "VERB"]:
                continue
            clean_words.append(ud.unidecode(token.lemma_.lower()))

        return clean_words

    def _from_df(self, df: pd.DataFrame, column_name: str) -> List[AnyStr]:
        """
        Returns a list of words from a pandas DataFrame object.

        Parameters
        ----------
        df: pd.Dataframe
            Pandas data frame with the strings to be parsed.
        column_name: str
            Name of the column with the strings.

        Returns
        -------
        List[AnyStr]:
            List of words of the data frame, with repetitions.
        """

        words = []
        with click.progressbar(df.iterrows(), length=df.shape[0], label="Reading data frame") as bar:
            for index, row in bar:
                words += self.clean_string(row[column_name])

        return words

    def _from_string(self, s: AnyStr) -> List[AnyStr]:
        """
         Returns a list of words from a string object.

        Parameters
        ----------
        s: str
            The string to be parsed.

        Returns
        -------
        List[AnyStr]:
            List of words of the string, with repetitions.
        """

        return self.clean_string(s)

    def fit(self, words: List[AnyStr]) -> None:
        """
        With a list of words, create a mapping String to Integer considering number of occurrences as importance order.
        Words with higher occurrences receive a higher token value.

        Parameters
        ----------
        words: List[AnyStr]
            List of the words occurred.

        Returns
        -------
            None.
        """
        count_words = Counter(words)
        word_order = [k for k, v in sorted(count_words.items(), key=lambda item: item[1], reverse=True)]
        click.secho(f"{len(word_order)} words mapped.", fg='green')
        word_map = zip(word_order[0:self.n_tokens], np.arange(self.n_tokens-1, 0, -1))
        self.map = {word: int(value) for word, value in word_map}

    def get(self, word: str) -> int:
        """
        Do a search on the word map.

        Parameters
        ----------
        word: str
            Word to be mapped.

        Returns
        -------
        int:
            The respective token value, according th the mapping.
        """
        return self.map.get(ud.unidecode(word), 0)

    def load(self, map: Dict):
        self.map = map
        self.n_tokens = len(map)

        return self


if __name__ == '__main__':

    try:
        df = pd.read_csv(join(".", 'data', 'processed', 'review_message.csv'))
    except FileNotFoundError:
        df = pd.read_csv(join("..", 'data', 'processed', 'review_message.csv'))

    df["message"] = df["message"].apply(lambda d: "none" if type(d) == float else d)
    print(f"Number of float values {df[df.message.apply(type) == float].shape[0]}")

    X = df.drop(['score'], axis=1)
    y = df["score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    click.echo(f'Train size {X_train.shape[0]} Test size {X_test.shape[0]}')

    tk = TextTokenizer(X_train, "message", use_gpu=True)

    with open('word_map.json', 'w') as fp:
        json.dump(tk.map, fp)

