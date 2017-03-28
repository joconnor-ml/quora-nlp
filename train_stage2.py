import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict, KFold

from models import stage_2_model_list_unweighted, cv

X = pd.read_csv("data/stage2_train.{}.csv".format(cv.get_n_splits()), index_col=0, usecols=[0])
X = pd.concat([X, pd.read_csv("data/stage2_deepnet_train.{}fold.csv".format(cv.get_n_splits()), index_col=0)], axis=1)
print(X.head())
y = pd.read_csv("data/quora_features.csv", usecols=["is_duplicate"]).values.ravel()

print(X.apply(lambda col: log_loss(y, col)))
print(log_loss(y, X.mean(axis=1)))
preds = {}
for name, model in stage_2_model_list_unweighted:
    p = cross_val_predict(model, X, y, method="predict_proba", cv=cv, verbose=1)[:, 1]
    print(name, log_loss(y, p))
    preds[name] = p
df = pd.DataFrame(preds)
print("Averaged:")
print(log_loss(y, df.mean(axis=1)))
