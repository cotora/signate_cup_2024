from catboost import CatBoostClassifier,Pool
import pandas as pd

class CatBoostModel:

    def __init__(self):
        self.bst = None
        self.cat_features=["Occupation","Gender","ProductPitched","Passport","Designation","MarriageStatus","CarOwnership","TypeofContact"]

    def fit(self, X_train, y_train, X_valid, y_valid):

        train_pool = Pool(X_train, y_train,cat_features=self.cat_features,feature_names=list(X_train.columns))
        valid_pool = Pool(X_valid, y_valid,cat_features=self.cat_features,feature_names=list(X_valid.columns))

        catboost_default_params = {
            'iterations': 1000,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': 100,
            'random_seed': 0,
        }

        self.bst = CatBoostClassifier(**catboost_default_params)
        self.bst.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    def predict_proba(self, X):
        if self.bst is None:
            raise Exception("fit() must be called before predict()")
        
        test_pool = Pool(X,cat_features=self.cat_features,feature_names=list(X.columns))
        return self.bst.predict_proba(test_pool)[:,1]

    def save(self, path):
        if self.bst is None:
            raise Exception("fit() must be called before save()")
        self.bst.save_model(path)

    def load(self, path):
        self.bst = CatBoostClassifier.load_model(path)
