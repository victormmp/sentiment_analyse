import numpy as np
import pandas as pd
from typing import Dict, List, Union
import click


class Normalizer:

    def __init__(self):
        self.params: List[Dict] = []

    def fit(self, data: Union[np.ndarray, pd.DataFrame]):

        is_data_frame = isinstance(data, pd.DataFrame)

        for c in range(data.shape[1]):
            if is_data_frame:
                self.params.append({'mean': np.mean(data.iloc[:, c]), 'std': np.std(data.iloc[:, c])})
            else:
                self.params.append({'mean': np.mean(data[:, c]), 'std': np.std(data[:, c])})

    def transform(self, data):
        data_trans = data.copy()

        is_data_frame = isinstance(data, pd.DataFrame)

        with click.progressbar(length=np.prod(data.shape), label="Transforming data") as bar:
            for index in range(data.shape[0]):
                for c in range(data.shape[1]):
                    if is_data_frame:
                        data_trans.iloc[index, c] = (data.iloc[index, c] - self.params[c]["mean"]) / self.params[c]['std']
                    else:
                        data_trans[index, c] = (data[index, c] - self.params[c]["mean"]) / self.params[c]['std']
                    bar.update(1)

        return data_trans

    def fit_transform(self, data):
        self.fit(data)

        return self.transform(data)
