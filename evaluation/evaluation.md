# Exact Copying Evaluation Results

## 概要

このドキュメントは、異なるLLMモデルの正確な文章コピー能力を評価した結果をまとめています。

## 評価指標

- **Exact Match Accuracy**: 回答と実際の出力が完全に一致した割合
- **Inclusion Accuracy**: 実際の出力が回答に含まれている割合
- **Average Edit Distance**: 回答と実際の出力間の正規化編集距離（0.0が完全一致、1.0が完全不一致）

## 評価結果サマリー

### Exact Match Accuracy

| Model        | qa-natural | simple-natural | simple-random |
|--------------|------------|----------------|---------------|
| gpt-4.1-mini | 0.010      | 0.030          | 0.85          |
| gpt-4.1      | 0.00       | 0.00           | 1.0           |
| o4-mini      | 0.94       | 0.98           | 0.99          |
| gpt-5-nano   | 0.41       | 0.72           | 0.88          |
| gpt-5-mini   | 0.90       | 1.0            | 0.98          |
| gpt-5        | 0.98       | 0.99           | 0.99          |

### Inclusion Accuracy

| Model        | qa-natural | simple-natural | simple-random |
|--------------|------------|----------------|---------------|
| gpt-4.1-mini | 0.94       | 0.74           | 0.85          |
| gpt-4.1      | 1.0        | 0.63           | 1.0           |
| o4-mini      | 0.97       | 0.98           | 0.99          |
| gpt-5-nano   | 0.94       | 0.72           | 0.88          |
| gpt-5-mini   | 0.93       | 1.0            | 0.98          |
| gpt-5        | 1.0        | 0.99           | 0.99          |

### Average Edit Distance

| Model        | qa-natural | simple-natural | simple-random |
|--------------|------------|----------------|---------------|
| gpt-4.1-mini | 0.61       | 0.33           | 0.18          |
| gpt-4.1      | 0.64       | 0.62           | 0.00          |
| o4-mini      | 0.014      | 0.0093         | 0.000066      |
| gpt-5-nano   | 0.34       | 0.27           | 0.026         |
| gpt-5-mini   | 0.0039     | 0.00           | 0.000074      |
| gpt-5        | 0.0018     | 0.000062       | 0.000033      |

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

全実行の場合は以下。

```bash
uv run evaluation/reproduce_scripts/run_evaluate.py
```

個別実行の場合は以下。

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
