# 概要
[SIGNATE Cup 2024](https://signate.jp/competitions/1376)のコードです。

# 実行方法
以下のコマンドでpoetryの仮想環境が作成されます。
```
poetry install
```

`data/input`内にtrainデータとtestデータを保存し、以下のコマンドでモデルの学習と推論が実行されます。推論結果は`data/output`内に出力されます。
```
poetry run python -m main
```