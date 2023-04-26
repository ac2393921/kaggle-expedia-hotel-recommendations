# Expedia Hotel Recommendations

## About

kaggleのExpedia Hotel Recommendationsを実験用リポジトリ

URL: https://www.kaggle.com/competitions/expedia-hotel-recommendations

## Useage

### 学習を実行

```bash
make up
docker compose exec mlflow bash
python3 src/main.py
```

### Mlflowを起動

http://localhost:5000/#/

###  Jupyterを起動

http://localhost:8000/lab?

### モデルの設定

モデルはhydraで管理しており、confディレクトリで設定できます。
使用したいymlファイルはsrc/main.pyで設定してください。

## Directory

```
.
├── Makefile
├── README.md
├── conf # hydraのconf置き場
├── docker # dockerの設定
├── docker-compose.yml
├── poetry.lock
├── poetry.toml
├── project
│   ├── LICENSE
│   ├── Makefile
│   ├── README.md
│   ├── data
│   ├── docs
│   ├── models
│   ├── notebooks # notebookを保存
│   ├── references
│   ├── reports
│   ├── requirements.txt
│   ├── setup.py
│   ├── src
│   ├── test_environment.py
│   └── tox.ini
├── pyproject.toml
├── src
│   ├── __init__.py
│   ├── dataset # dataloader
│   ├── jobs # 各ジョブの配置
│   ├── main.py # 訓練実行ファイル
│   └── models # モデルを配置
└── submission # submissionファイルが保存される
```