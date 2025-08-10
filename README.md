
# exact-copying-eval

商用LLMの日本語自然文に対するExact-copying性能を評価するリポジトリです。

## 概要

大規模言語モデル（LLM）等の出力に対し、Exact-copyingの判定や自動評価を行うためのツール群・データセット・評価スクリプトを提供します。

主な用途：
- 評価データセットの作成
- 厳密一致・部分一致・正規化編集距離のメトリクスによる性能評価

## 使い方

### 環境・準備

- macOS（M1) 14.7、python 3.12にて動作確認済み。
- 環境変数にOPENAI_API_KEYを登録
- uvのインストール

### 評価用データセットの作成

```bash
uv run python src/exact_copying_eval/core/create_dataset.py --nums 10 100 --output_dir evaluation/dataset/
```
- `--nums`: 生成するデータ数（複数指定可、-1で全件）
- `--output_dir`: 出力先ディレクトリ
- `--random-strings`: 正解をランダムな日本語文字列に変換（オプション）
- `--seed`: 乱数シード（デフォルト: 42）

### モデル評価の実行

```bash
uv run python src/exact_copying_eval/core/evaluate.py --dataset evaluation/dataset/evaluation_dataset_100.json --model gpt-5-nano --prompt_type qa --output_file evaluation/result/result_gpt-5-nano_evaluation_dataset_100_qa.json
```
- `--dataset`: 評価用データセット（JSON）
- `--model`: 評価対象モデル名（例: gpt-5-nano）
- `--batch_size`: バッチサイズ（デフォルト: 10）
- `--prompt_type`: プロンプト種別（qa/simple）
- `--output_file`: 結果出力ファイル
- `--output_dir`: 結果出力ディレクトリ