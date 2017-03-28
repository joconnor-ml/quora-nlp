import numpy as np

import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import log_loss
from deepnet_models import model_list, Adam
from models import cv
import pickle

    
try:
    with open("data/train_data.pkl", "rb") as f:
        (x1, x2, y, embedding_matrix, word_index) = pickle.load(f)

except:
    x1, x2, y, embedding_matrix, word_index = get_data_and_embeddings()
    with open("data/train_data.pkl", "wb") as f:
        pickle.dump((x1, x2, y, embedding_matrix, word_index), f)


print(x1.shape, x2.shape, y.shape, embedding_matrix.shape)

for name, model_fn in model_list:
    print(name)
    model = model_fn(x1.shape, len(word_index) + 1, embedding_matrix)
    model.fit([x1, x2], y, epochs=6, batch_size=512, validation_split=0.1)
    model.compile(loss="binary_crossentropy", optimizer=Adam(1e-4), metrics=["accuracy"])
    model.fit([x1, x2], y, epochs=2, batch_size=512, validation_split=0.1)
