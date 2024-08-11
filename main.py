import pandas as pd
import preprocess as pp
import datetime
from sklearn.model_selection import train_test_split
from models.LightGBMClassifier import LightGBMClassifier
from sklearn.metrics import roc_auc_score
from numpy import ndarray
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def main(train_path, test_path, output_path):

    # データの読み込み
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 前処理
    all_data = pd.concat([train, test])
    all_data = pp.preprocess(all_data)
    train = all_data.iloc[: train.shape[0], :].reset_index(drop=True)
    test = all_data.iloc[train.shape[0] :, :].reset_index(drop=True)
    test = test.drop("ProdTaken", axis=1)

    # 前処理結果の確認
    print(train.head())

    # 特徴量と目的変数の分離
    y = train["ProdTaken"]
    X = train.drop("ProdTaken", axis=1)

    # 学習データと検証データの分離
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルの学習
    params = {"objective": "binary", "metric": "binary_logloss"}

    model = LightGBMClassifier(params)
    model.fit(X_train, y_train, X_valid, y_valid)

    # モデルの評価
    y_pred = model.predict_proba(X_valid)

    if isinstance(y_pred, ndarray):
        auc = roc_auc_score(y_valid, y_pred)
        print(auc)

        fpr, tpr, thresholds = roc_curve(y_valid, y_pred)

        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(fpr, tpr, label="LogisticRegression")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.show()

    # テストデータの予測
    y_test = model.predict_proba(test)

    # 提出用ファイルの作成
    submission = pd.DataFrame({"id": test["id"], "ProdTaken": y_test})

    print(submission.head())

    submission.to_csv(output_path, index=False, header=False)


if __name__ == "__main__":
    train_path = "data/input/train.csv"
    test_path = "data/input/test.csv"

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = f"data/output/submission_{current_time}.csv"
    main(train_path, test_path, output_path)
