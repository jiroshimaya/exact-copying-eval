# Exact Copying Evaluation Results

## 概要

このドキュメントは、異なるLLMモデルの正確な文章コピー能力を評価した結果をまとめています。

## 評価指標

- **Exact Match Accuracy**: 予想回答と実際の出力が完全に一致した割合
- **Inclusion Accuracy**: 予想回答が実際の出力に含まれている割合
- **Average Edit Distance**: 予想回答と実際の出力間の正規化編集距離（0.0が完全一致、1.0が完全不一致）

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