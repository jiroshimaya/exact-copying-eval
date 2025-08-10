# Exact Copying Evaluation Results

## 概要

このドキュメントは、異なるLLMモデルの正確な文章コピー能力を評価した結果をまとめています。

## 評価指標

- **Exact Match Accuracy**: 回答と実際の出力が完全に一致した割合
- **Inclusion Accuracy**: 実際の出力が回答に含まれている割合
- **Average Edit Distance**: 回答と実際の出力間の正規化編集距離（0.0が完全一致、1.0が完全不一致）

## 評価結果サマリー


### モデル別・条件別性能比較

| Model        | Condition       | Exact Match | Inclusion | Avg Edit Distance |
|--------------|----------------|-------------|-----------|-------------------|
| gpt-4.1-mini | simple-random  | 0.83        | 0.83      | 0.24              |
| gpt-4.1-mini | simple-natural | 0.020       | 0.74      | 0.35              |
| gpt-4.1-mini | qa-natural     | 0.040       | 0.95      | 0.59              |
| gpt-4.1      | simple-random  | 0.98        | 0.98      | 0.019             |
| gpt-4.1      | simple-natural | 0.00        | 0.66      | 0.72              |
| gpt-4.1      | qa-natural     | 0.00        | 1.0       | 0.62              |
| o4-mini      | simple-random  | 0.99        | 0.99      | 0.00010           |
| o4-mini      | simple-natural | 0.96        | 0.97      | 0.0018            |
| o4-mini      | qa-natural     | 0.94        | 0.98      | 0.026             |
| gpt-5-nano   | simple-random  | 0.87        | 0.87      | 0.017             |
| gpt-5-nano   | simple-natural | 0.69        | 0.69      | 0.31              |
| gpt-5-nano   | qa-natural     | 0.30        | 0.95      | 0.39              |
| gpt-5-mini   | simple-random  | 0.99        | 0.99      | 0.00              |
| gpt-5-mini   | simple-natural | 1.0         | 1.0       | 0.00              |
| gpt-5-mini   | qa-natural     | 0.86        | 0.97      | 0.023             |
| gpt-5        | simple-random  | 1.0         | 1.0       | 0.00              |
| gpt-5        | simple-natural | 1.0         | 1.0       | 0.00              |
| gpt-5        | qa-natural     | 0.97        | 0.99      | 0.0027            |

## 再現用コマンド

### データセット作成
```bash
uv run python src/exact_copying_eval/core/create_dataset.py --output_dir evaluation/dataset/
```

以下のファイルが生成される。実際に使うのは100のみ。

- evaluation/dataset/evaluation_dataset_10.json
- evaluation/dataset/evaluation_dataset_100.json
- evaluation/dataset/evaluation_dataset_4442.json

```bash
uv run python src/exact_copying_eval/core/create_dataset.py --output_dir evaluation/dataset/ --random-string
```

以下のファイルが生成される。実際に使うのは100のみ。

- evaluation/dataset/evaluation_dataset_10_random.json
- evaluation/dataset/evaluation_dataset_100_random.json
- evaluation/dataset/evaluation_dataset_4442_random.json


### モデル評価
```bash
uv run python src/exact_copying_eval/core/evaluate.py --dataset DATASET --model MODEL --prompt_type PROMPT_TYPE --output_dir evaluation/result
```

DATASETとPROMPT_TYPEの組み合わせは以下のいずれか。

```bash
DATASET=evaluation/dataset/evaluation_dataset_100.json PROMPT_TYPE=qa
DATASET=evaluation/dataset/evaluation_dataset_100.json PROMPT_TYPE=simple
DATASET=evaluation/dataset/evaluation_dataset_100_random.json PROMPT_TYPE=simple
```

MODELは以下のいずれか

- gpt-4.1-mini
- gpt-4.1
- o4-mini
- gpt-5-nano
- gpt-5-mini
- gpt-5