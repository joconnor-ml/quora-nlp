import numpy as np

import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import log_loss
from deepnet_models import model_list, Adam
from models import cv
import pickle

def keras_cross_val_predict(model_fn, x1, x2, y, cv):
    preds = np.ones_like(y).astype(np.float32)
    for fold, (train, test) in enumerate(cv.split(x1)):
        # new model for each fold
        model = model_fn(x1.shape, len(word_index) + 1, embedding_matrix)
        print(y.mean())
        model.fit([x1[train], x2[train]], y[train], epochs=6, batch_size=512, validation_split=0.05)
        model.compile(loss="binary_crossentropy", optimizer=Adam(1e-4), metrics=["accuracy"])
        model.fit([x1[train], x2[train]], y[train], epochs=2, batch_size=512, validation_split=0.05)
        p = model.predict([x1[test], x2[test]])
        print(p)
        preds[test] = p
        model.save("models/{}.fold{}.{}fold.h5".format(name, fold, cv.get_n_splits()))
    return preds
    
try:
    with open("data/train_data.pkl", "rb") as f:
        (x1, x2, y, embedding_matrix, word_index) = pickle.load(f)

except:
    x1, x2, y, embedding_matrix, word_index = get_data_and_embeddings()
    with open("data/train_data.pkl", "wb") as f:
        pickle.dump((x1, x2, y, embedding_matrix, word_index), f)


print(x1.shape, x2.shape, y.shape, embedding_matrix.shape)

preds = {}
for name, model_fn in model_list:
    p = keras_cross_val_predict(model_fn, x1, x2, y, cv)
    print(name, log_loss(y, p))
    preds[name] = p

stage2train = pd.DataFrame(preds)
stage2train.to_csv("data/stage2_deepnet_train.{}fold.csv".format(cv.get_n_splits()))
