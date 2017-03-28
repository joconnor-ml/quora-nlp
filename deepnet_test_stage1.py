import numpy as np

import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import log_loss
from deepnet_models import model_list 
from models import cv

import pickle

from utils import get_data_and_embeddings

try:
    with open("data/test_data.pkl", "rb") as f:
        x1, x2, y, embedding_matrix, word_index = pickle.load(f)

except:
    x1, x2, y, embedding_matrix, word_index = get_data_and_embeddings("data/test.csv.gz")
    with open("data/test_data.pkl", "wb") as f:
        pickle.dump((x1, x2, y, embedding_matrix, word_index), f)


print(x1.shape, x2.shape, embedding_matrix.shape)

preds = {}
for name, model_fn in model_list:
    p = []
    for fold in range(cv.get_n_splits()):
        model = model_fn(x1.shape, len(word_index) + 1, embedding_matrix)
        model.load_weights("models/{}.fold{}.{}fold.h5".format(name, fold, cv.get_n_splits()))
        p.append(model.predict([x1, x2]))
    preds[name] = np.concatenate(p, axis=1).mean(axis=1)
    print(preds)

stage2train = pd.DataFrame(preds)
stage2train.to_csv("data/stage2_deepnet_test.{}fold.csv".format(cv.get_n_splits), index=False)
