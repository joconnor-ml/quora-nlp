import pandas as pd
from sklearn.metrics import log_loss

from models import stage_2_model_list

train = pd.read_csv("data/stage2_train.csv", index_col=0)
train = pd.concat([train, pd.read_csv("data/stage2_deepnet_train.csv", index_col=0)], axis=1)
print(train.head())
y = pd.read_csv("data/quora_features.csv", usecols=["is_duplicate"]).values.ravel()

print(log_loss(y, train.mean(axis=1)))

models = {}
for name, model in stage_2_model_list:
    print(name)
    models[name] = model.fit(train, y)

print("Fitted stage 2 models. Reading test data")
test = pd.read_csv("data/stage2_test.csv", index_col=0)
print("Reading keras test data")
test = pd.concat([test, pd.read_csv("data/stage2_deepnet_test.csv")], axis=1)
test.index.name = "test_id"
preds = []
print("Predicting")
for name, model in models.items():
    print(name)
    p = model.predict_proba(test)[:,1]
    preds.append(p)
    test["is_duplicate"] = p
    test[["is_duplicate"]].to_csv("submissions/submission.{}.csv.gz".format(name), compression="gzip", float_format="%.3e")
    test.drop("is_duplicate", axis=1, inplace=True)

test["is_duplicate"] = sum(preds) / len(preds)
test[["is_duplicate"]].to_csv("submissions/submission.mean.csv.gz", compression="gzip", float_format="%.3e")
