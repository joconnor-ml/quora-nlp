import numpy as np
import pandas as pd
from models import model_list

train = pd.read_csv("data/quora_features.csv")
y = train["is_duplicate"]
X = train.drop(["is_duplicate", "question1", "question2"], axis=1).values.astype(np.float32)
del train


print("Fitting:")
models = {}
for name, model in model_list:
    print(name)
    models[name] = model.fit(X, y)

del y
del X

test = pd.read_csv("data/quora_test_features.csv", index_col=0)
test_index = test.index.values
Xtest = test.drop(["question1", "question2"], axis=1).values.astype(np.float32)
del test

print("Predicting")
preds = {}
for name, model in models.items():
    print(name)
    p = model.predict_proba(Xtest)[:, 1].astype(np.float32)
    preds[name] = p

del Xtest

stage2test = pd.DataFrame(preds, index=test_index)
stage2test.to_csv("data/stage2_test.csv")
