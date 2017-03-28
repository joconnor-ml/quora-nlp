from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

cv = KFold(4, shuffle=True, random_state=12)

model_list = [
    ("naive_bayes", make_pipeline(Imputer(), StandardScaler(),  GaussianNB())),
    ("logistic_l2", make_pipeline(Imputer(), StandardScaler(),  LogisticRegressionCV(penalty="l2", n_jobs=-1))),
    ("logistic", make_pipeline(Imputer(), StandardScaler(),  LogisticRegression(C=1e3, penalty="l2", n_jobs=-1))),
    ("random_forest", make_pipeline(Imputer(), RandomForestClassifier(256, max_depth=6, n_jobs=-1))),
    ("extra_trees", make_pipeline(Imputer(), ExtraTreesClassifier(256, max_depth=6, n_jobs=-1))),
    ("xgboost", XGBClassifier(n_estimators=1000, max_depth=12)),
    ("xgboost", XGBClassifier(n_estimators=2000, max_depth=4)),
]

stage_2_model_list = [
    ("logistic", LogisticRegressionCV(class_weight={0:1, 1:0.5}, n_jobs=-1)),
    #("random_forest", RandomForestClassifier(100, class_weight={0:1, 1:0.5}, n_jobs=-1)),
    ("xgboost", XGBClassifier(n_estimators=400, base_score=0.17, scale_pos_weight=0.5)),
]

stage_2_model_list_unweighted = [
    ("logistic", LogisticRegressionCV()),
    #("random_forest", RandomForestClassifier(100, n_jobs=-1)),
    ("calibrated", CalibratedClassifierCV(XGBClassifier())),
    ("xgboost", XGBClassifier(n_estimators=400, base_score=0.37)),
]
