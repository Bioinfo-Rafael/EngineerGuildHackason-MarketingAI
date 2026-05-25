# workflow — Action Path Model (APM) ML Pipeline

訪問したURLの系列から次の訪問先を予測する GRU Encoder–Decoder（seq2seq）の学習・評価用リポジトリ。（データは `data/` に同梱）。

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


## データの型と形状

#### 表1 — 生データ（`data/processed/{1..21}.json`）

| 項目 | 型 | 備考 |
|------|-----|------|
| ファイル全体 | `list`（長さ 9） | 1ファイル = 参加者1人分。要素はタスク1件 |
| タスク1件 | `dict` | `task_id: int`, `clickstream: list` |
| `clickstream` の要素1件 | `dict` | `user_id: int`, `previous_url: str`, `current_url: str`, `stay_seconds: float`, `time: str` |

`clickstream` は **Webページを閲覧した時間順に並んだ `list`**。学習コードが使うのは各要素の **`previous_url` だけ**（他キーは読まない）。

#### 前処理（`src/dataset.py` の `build_tensors`）

1. **サンプル抽出** — `user_id` 1〜21 × `task_id` 1〜9 → 計 **189 サンプル**（1サンプル = タスク1件）。
2. **URL列の作成** — そのタスクの `clickstream` を先頭から走査し、`previous_url` だけを順に並べた `list[str]` を作る。
3. **時系列分割** — `split_ratio`（既定 `0.99`）で URL 列を前後に切る。  
   - 前側 → エンコーダ用  
   - 後側 → デコーダが当てる正解側
4. **ID化** — `data/vocabs.txt` で URL 文字列を整数 ID に変換（未知 URL は `7` = `<MIS>`）。先頭・末尾に特殊 ID を付与（`<SOA>`=1, `<COI>`=2, `<EOA_*>`=4〜6 など）。
5. **長さ揃え** — 189 サンプル間で最大長に合わせ、短い列は `<PAD>`（0）で埋める（`pad_sequences`）。
6. **正解の one-hot 化** — デコーダ側の各時刻ラベルを `float32` の one-hot ベクトルにする（語彙数 `V` ≈ 2873 次元）。

#### 表2 — モデル入力直前（`split_ratio: 0.99` の実測例）

| 変数名 | dtype | shape | `model.fit` での役割 |
|--------|-------|-------|----------------------|
| `input_s` | `int64` | `(189, 90)` | `X[0]` … エンコーダ入力 |
| `output_s` | `int64` | `(189, 1)` | `X[1]` … デコーダ入力（教師強制） |
| `output_s_one_shot` | `float32` | `(189, 1, 2873)` | `y` … 各時刻の正解分布 |

呼び出し: `model.fit([input_s, output_s], output_s_one_shot, ...)`。  
`189` = サンプル数、`90` / `1` = 入力・出力のタイムステップ数（データの最大クリック長と `split_ratio` で決まる）、`2873` = 語彙数 `V`。



## ライセンス

CC-BY-NC 4.0 / MIT（HCI 修士論文データセット由来）



## 出典

[https://github.com/changkun/MasterThesisHCI/tree/master](https://github.com/changkun/MasterThesisHCI/tree/master)

