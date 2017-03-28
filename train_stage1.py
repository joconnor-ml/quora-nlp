import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict
from models import model_list, cv

df = pd.read_csv("data/quora_features.csv")
y = df["is_duplicate"]
X = df.drop(["is_duplicate", "question1", "question2"], axis=1)

preds = {}
for name, model in model_list:
    p = cross_val_predict(model, X, y, method="predict_proba", cv=cv, verbose=1)[:, 1]
    print(name, log_loss(y, p))
    preds[name] = p

stage2train = pd.DataFrame(preds)
stage2train.to_csv("data/stage2_train.csv")

