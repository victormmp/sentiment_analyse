import os
import string
from os.path import join
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from src.utils import check_path

try:
    df = pd.read_csv(join(".", 'data', 'raw', 'olist_order_reviews_dataset.csv'))
except FileNotFoundError:
    df = pd.read_csv(join("..", 'data', 'raw', 'olist_order_reviews_dataset.csv'))

"""
First we will try to predict reward score only using the comments. For this, we will generate a new dataset.
"""

#%%
df_new = df[["review_comment_message", "review_score"]]
df_new["score"] = df_new["review_score"].apply(lambda d: 1 if d > 3 else 0)
df_new["message"] = df_new["review_comment_message"]
df_new = df_new.drop(["review_comment_message", "review_score"], axis=1)

#%% Fill empty reviews

df_new['message'] = df_new['message'].fillna("none")
df_new["message"] = df_new["message"].apply(lambda d: "none" if type(d) == float else d)
print(f'Number of nan fields: {df_new[df_new["message"] == np.nan].shape[0]}')
print(f"Number of float values {df_new[df_new.message.apply(type) == float].shape[0]}")
df_new['message'] = df_new["message"].apply(lambda d: d.translate(str.maketrans('', '', string.punctuation)))

if os.path.exists(join('.', 'data')):
    df_new.to_csv(join(check_path('.', 'data', 'processed'), 'review_message.csv'), index=False)
elif os.path.exists(join('..', 'data')):
    print("BLA")
    df_new.to_csv(join(check_path('..', 'data', 'processed'), 'review_message.csv'), index=False)
else:
    raise FileNotFoundError('No "data" folder found.')





