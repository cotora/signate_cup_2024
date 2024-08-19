import pandas as pd
import preprocess as pp
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from models.LightGBMClassifier import LightGBMClassifier
from models.CatBoostModel import CatBoostModel
from sklearn.metrics import roc_auc_score
from numpy import ndarray
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


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
    print(train.columns)

    # 特徴量と目的変数の分離
    y = train["ProdTaken"]
    X = train.drop("ProdTaken", axis=1)

    # idの削除と保存
    X = X.drop("id", axis=1)
    test_id = test["id"]
    test = test.drop("id", axis=1)

    # stratified k-foldによる交差検証
    kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_list = []

    y_pred = np.zeros(len(y))

    models = []

    for tr_idx,va_idx, in kf.split(X,y):
        X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
        X_valid, y_valid = X.iloc[va_idx], y.iloc[va_idx]

        #train_td,valid_td=pp.target_encoder(pd.concat([X_train,y_train],axis=1), pd.concat([X_valid,y_valid],axis=1), "ProdTaken", ["Occupation","Gender","ProductPitched","Passport","Designation","MarriageStatus","CarOwnership","TypeofContact"])

        #X_train = train_td.drop("ProdTaken", axis=1)
        #X_valid = valid_td.drop("ProdTaken", axis=1)

        # モデルの学習


        model = CatBoostModel()
        model.fit(X_train, y_train, X_valid, y_valid)

        # モデルの評価
        y_pred[va_idx] = model.predict_proba(X_valid)

        auc = roc_auc_score(y_valid, y_pred[va_idx])
        auc_list.append(auc)

        models.append(model)

    if isinstance(y_pred, ndarray):
        auc = roc_auc_score(y, y_pred)
        print(auc)

        fpr, tpr, thresholds = roc_curve(y, y_pred)

        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(fpr, tpr, label="LogisticRegression")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.show()

    print("AUCの平均 : ", np.mean(auc_list))

    # テストデータの予測
    y_test_pred = np.zeros(len(test))

    # テストデータのターゲットエンコーディング
    #_,test=pp.target_encoder(pd.concat([X,y],axis=1), test, "ProdTaken", ["Occupation","Gender","ProductPitched","Passport","Designation","MarriageStatus","CarOwnership","TypeofContact"])

    # バリデーション毎の予測の平均をとる場合
    for model in models:
        y_test_pred += model.predict_proba(test)
    y_test_pred /= len(models)
    

    #最もAUCが高かったモデルを使う場合
    #y_test_pred = models[np.argmax(auc_list)].predict_proba(test)
    

    # 提出用ファイルの作成
    submission = pd.DataFrame({"id": test_id, "ProdTaken": y_test_pred})

    print(submission.head())

    submission.to_csv(output_path, index=False, header=False)


if __name__ == "__main__":
    train_path = "data/input/train.csv"
    test_path = "data/input/test.csv"

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = f"data/output/submission_{current_time}.csv"
    main(train_path, test_path, output_path)
