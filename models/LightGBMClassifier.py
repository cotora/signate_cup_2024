import lightgbm as lgb


class LightGBMClassifier:

    def __init__(self, params):
        self.params = params
        self.bst = None

    def fit(self, X_train, y_train, X_valid, y_valid):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        self.bst = lgb.train(
            self.params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True)],
        )

    def predict_proba(self, X):
        if self.bst is None:
            raise Exception("fit() must be called before predict()")
        return self.bst.predict(X)

    def save(self, path):
        if self.bst is None:
            raise Exception("fit() must be called before save()")
        self.bst.save_model(path)

    def load(self, path):
        self.bst = lgb.Booster(model_file=path)
