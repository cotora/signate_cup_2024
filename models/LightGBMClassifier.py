import lightgbm as lgb
import optuna_integration.lightgbm as optuna_lgb
from preprocess import target_encoder
import pandas as pd

    

class LightGBMClassifier:

    def __init__(self):
        self.bst = None

    def fit(self, X_train, y_train, X_valid, y_valid):

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        categorical_feature=["Occupation","Gender","ProductPitched","Passport","Designation","MarriageStatus","CarOwnership","TypeofContact"]

        params = {
            "objective": "binary",
            "metric": "auc",
            "random_state": 42,
            "boosting_type": "gbdt",
        }

        self.bst = optuna_lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True)],
        )

    def predict_proba(self, X):
        if self.bst is None:
            raise Exception("fit() must be called before predict()")
        return self.bst.predict(X, num_iteration=self.bst.best_iteration)

    def save(self, path):
        if self.bst is None:
            raise Exception("fit() must be called before save()")
        self.bst.save_model(path)

    def load(self, path):
        self.bst = lgb.Booster(model_file=path)
