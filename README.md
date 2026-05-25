# workflow — Action Path Model (APM) ML Pipeline

訪問したURLの系列から次の訪問先を予測する GRU Encoder–Decoder（seq2seq）の学習・評価用リポジトリ。（データは `data/` に同梱）。

### モデルに入力するデータの型と形状

**1. ファイル上の JSON**（`data/processed/{user_id}.json`）

| 階層 | 型 | 説明 |
|------|-----|------|
| ルート | `list[dict]` | 長さ 9（タスク数） |
| 各要素 | `dict` | キー: `task_id: int`, `clickstream: list[dict]` |
| `clickstream` の1要素 | `dict` | `user_id: int`, `previous_url: str`, `current_url: str`, `stay_seconds: float`, `time: str`（ISO8601） |

`clickstream` は「遷移ログの配列」であり、配列の並び順が時間順。モデル学習では **`previous_url` の列だけ**を取り出し、他フィールドはテンソル化しない。

**2. 1サンプル（1タスク）の中間表現**（`src/dataset.py`）

| 変数 | 型 | 長さの目安 |
|------|-----|------------|
| URL列 | `list[str]` | クリック数 = `len(clickstream)` |
| トークン列（入力側） | `list[int]` | `1 + len(先頭部分) + (0 or 1)`（`<SOA>`, URL…, 任意で `<COI>`） |
| トークン列（出力側） | `list[int]` | `len(末尾部分) + 1`（URL…, `<EOA_*>`） |

先頭/末尾の切り方: `split_ratio`（既定 `0.99`）で URL列を  
`sentence[:int(len(sentence)*split_ratio)]` と `sentence[int(len(sentence)*split_ratio):]` に分割。

語彙: `data/vocabs.txt` の行番号がトークン ID。語彙数 `V` ≈ 2873（特殊トークン 8 個 + URL）。

**3. バッチ化後の NumPy 配列**（全 189 サンプル = 21 ユーザー × 9 タスク）

`configs/default.yaml` の `split_ratio: 0.99` 時の実測例:

| 名前 | dtype | shape | 内容 |
|------|-------|-------|------|
| `input_s` | `int64` | `(189, 90)` | エンコーダ第1入力（パディング値 `0` = `<PAD>`） |
| `output_s` | `int64` | `(189, 1)` | デコーダ第2入力（教師強制用トークン ID） |
| `output_s_one_shot` | `float32` | `(189, 1, 2873)` | 正解ラベル（各時刻の one-hot、語彙次元 `V`） |

**4. `model.fit` の引数**

```text
X = [input_s, output_s]   # 長さ2のリスト（多入力）
y = output_s_one_shot     # shape (N, T_out, V)
```

`N=189`。`T_in`, `T_out` はデータセット内の最大クリック長と `split_ratio` から決まる（上記例では `T_in=90`, `T_out=1`）。

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

`[configs/default.yaml](configs/default.yaml)` を編集します。


| キー            | 既定   | 説明                     |
| ------------- | ---- | ---------------------- |
| `epochs`      | 10   | smoke 用。本番は 1500 などに変更 |
| `split_ratio` | 0.99 | 時系列の入力/出力分割比           |
| `train_ratio` | 0.92 | train / holdout 分割     |


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



## 出典

[https://github.com/changkun/MasterThesisHCI/tree/master](https://github.com/changkun/MasterThesisHCI/tree/master)

