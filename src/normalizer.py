import numpy as np
import pandas as pd
from typing import Dict, List, Union
import click


class Normalizer:

    def __init__(self):
        self.params: List[Dict] = []

    def fit(self, data: Union[np.ndarray, pd.DataFrame]):
        """
        Considering a base dataset, obtain the parameters used to calculate the normalization.

        Parameters
        ----------
        data: Union[np.ndarray, pd.DataFrame]
            Input data

        Returns
        -------
            None.
        """

        is_data_frame = isinstance(data, pd.DataFrame)

        for c in range(data.shape[1]):
            if is_data_frame:
                self.params.append({'mean': np.mean(data.iloc[:, c]), 'std': np.std(data.iloc[:, c])})
            else:
                self.params.append({'mean': np.mean(data[:, c]), 'std': np.std(data[:, c])})

    def transform(self, data: Union[np.ndarray, pd.DataFrame]):
        """
        Apply normalization calculation upon a data matrix.

        Parameters
        ----------
        data: Union[np.ndarray, pd.DataFrame]
            Input data to be transformec

        Returns
        -------
        Union[np.ndarray, pd.DataFrame]:
            Output transformed matrix.
        """
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

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]):
        """
        Fit data and transforms it.

        Parameters
        ----------
        data: Union[np.ndarray, pd.DataFrame]
            Input data matrix.

        Returns
        -------
        Union[np.ndarray, pd.DataFrame]:
            Output transformed matrix.
        """
        self.fit(data)

        return self.transform(data)
