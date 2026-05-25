# workflow — Action Path Model (APM) ML Pipeline

クリックストリーム（URL 列）の続きを予測する GRU Encoder–Decoder（seq2seq）の学習・評価用リポジトリ。**このディレクトリだけで完結**します（データは `data/` に同梱）。

## ディレクトリ構成

```
workflow/
├── configs/default.yaml    # 学習・評価パラメータ
├── data/
│   ├── processed/          # 整形済み JSON（参加者 1–21 × 9 タスク）
│   └── vocabs.txt          # URL 語彙
├── src/
│   ├── dataset.py          # データ読み込み・テンソル化
│   ├── model.py            # モデル定義
│   ├── train.py            # 学習
│   ├── evaluate.py         # 評価
│   └── predict.py          # 1 タスク推論
├── scripts/
│   └── build_vocab.py      # 任意: vocabs.txt の再生成
├── artifacts/pretrained/   # 任意: レガシー weights（evaluate の fallback）
└── outputs/                # 学習の出力（gitignore）
    ├── checkpoints/        # best.weights.h5, last.weights.h5
    ├── metrics/
    └── figures/
```

## クイックスタート

```bash
cd workflow
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export KERAS_BACKEND=torch   # Python 3.14+ では PyTorch バックエンド

python -m src.train
python -m src.evaluate
python -m src.predict --user-id 1 --task-id 1
```

`data/processed/` と `data/vocabs.txt` はリポジトリに含まれているため、**外部からのコピー作業は不要**です。

## チェックポイントの優先順位

`evaluate.py` / `predict.py` で `--checkpoint` を省略したとき:

1. `outputs/checkpoints/best.weights.h5` — 学習時の `val_loss` 最小
2. `outputs/checkpoints/last.weights.h5` — 最終 epoch
3. `artifacts/pretrained/best.h5` — 同梱のレガシー weights（無い場合はスキップ）

初回は `src.train` を実行して 1 または 2 を生成してください。

## 設定

[`configs/default.yaml`](configs/default.yaml) を編集します。

| キー | 既定 | 説明 |
|------|------|------|
| `epochs` | 10 | smoke 用。本番は 1500 などに変更 |
| `split_ratio` | 0.99 | 時系列の入力/出力分割比 |
| `train_ratio` | 0.92 | train / holdout 分割 |

論文に近い予測評価:

```bash
python -m src.evaluate --split-ratio 0.97
```

## Full 学習（任意）

```bash
python -m src.train --epochs 1500
```

## 依存関係

- Python 3.11–3.14 想定
- Keras 3 + PyTorch（`requirements.txt`）。TensorFlow は不要

## データの秘匿化

公開リポジトリ用に OAuth トークン等が URL に含まれる場合は除去します。

```bash
python3 scripts/sanitize_secrets.py
python3 scripts/build_vocab.py   # vocabs.txt を再生成
```

## ライセンス

CC-BY-NC 4.0 / MIT（HCI 修士論文データセット由来）
