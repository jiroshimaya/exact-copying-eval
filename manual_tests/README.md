# Manual Tests for extract_answer_text_by_llm

このディレクトリには、`extract_answer_text_by_llm`関数の手動テストが含まれています。

## 概要

`extract_answer_text_by_llm`関数は実際のLLM API（litellm経由）を呼び出すため、以下の理由で手動テストが必要です：

- APIキーの設定が必要
- ネットワーク接続が必要
- API使用料金が発生する可能性
- レスポンス時間が不定
- 外部サービスの依存関係

## 前提条件

### 1. 環境変数の設定

LiteLLMを使用するため、適切なAPIキーを設定してください：

```bash
# OpenAI API Key（gpt-4-miniなどを使用する場合）
export OPENAI_API_KEY="your-api-key-here"

# または他のプロバイダーのAPIキー
export ANTHROPIC_API_KEY="your-anthropic-key"
export COHERE_API_KEY="your-cohere-key"
# etc.
```

### 2. 依存関係のインストール

```bash
uv sync --all-extras
```

## テスト実行方法

### 全ての手動テストを実行

```bash
# unit tests
uv run pytest manual_tests/unit/test_extract_answer_text_by_llm.py -v -m manual

# integration tests
uv run pytest manual_tests/integration/test_extract_answer_text_by_llm_integration.py -v -m manual

# 全ての手動テスト
uv run pytest manual_tests/ -v -m manual
```

### 特定のテストを実行

```bash
# 単一の関数をテスト
uv run pytest manual_tests/unit/test_extract_answer_text_by_llm.py::TestExtractAnswerTextByLlm::test_正常系_単一の質問と回答 -v -s

# 統合テストの性能確認
uv run pytest manual_tests/integration/test_extract_answer_text_by_llm_integration.py::TestExtractAnswerTextByLlmIntegration::test_バッチ処理性能確認 -v -s
```

### ログ出力付きで実行

```bash
# 詳細なログ出力と結果表示
uv run pytest manual_tests/ -v -s -m manual --log-cli-level=INFO
```

## テスト構成

### Unit Tests (`manual_tests/unit/`)

- `test_extract_answer_text_by_llm.py`: 基本的な機能テスト
  - 正常系: 単一/複数の質問と回答
  - エッジケース: 空リスト、回答が見つからない場合
  - パラメータ化テスト: 様々な質問パターン

### Integration Tests (`manual_tests/integration/`)

- `test_extract_answer_text_by_llm_integration.py`: 統合テスト
  - 実際のデータセット形式での動作確認
  - バッチ処理性能確認
  - エラー耐性確認
  - 日本語特有の文字での動作確認

## 注意事項

### APIコストについて

- これらのテストは実際のLLM APIを呼び出すため、使用料金が発生します
- テスト実行前に現在のAPI使用状況を確認することをお勧めします
- 開発環境では低コストのモデル（gpt-4-mini等）の使用を検討してください

### レート制限

- API プロバイダーのレート制限に注意してください
- 大量のテストを一度に実行する場合は、適切な間隔を設けることを検討してください

### テスト結果の解釈

- LLMの応答は非決定的であるため、完全に一致するアサーションは避けています
- キーワードの存在確認や基本的な構造チェックを中心としています
- テスト結果は目視でも確認することをお勧めします

## トラブルシューティング

### よくある問題

1. **APIキーエラー**
   ```
   Error: API key not found
   ```
   → 環境変数が正しく設定されているか確認

2. **ネットワークエラー**
   ```
   Error: Connection timeout
   ```
   → インターネット接続を確認

3. **レート制限エラー**
   ```
   Error: Rate limit exceeded
   ```
   → しばらく待ってから再実行

4. **モデルエラー**
   ```
   Error: Model not found
   ```
   → `evaluate.py`で指定されているモデル名が利用可能か確認

### デバッグ方法

```bash
# より詳細なログ出力
export LOG_LEVEL=DEBUG
uv run pytest manual_tests/ -v -s -m manual --log-cli-level=DEBUG

# 特定のテストのみ実行してデバッグ
uv run pytest manual_tests/unit/test_extract_answer_text_by_llm.py::TestExtractAnswerTextByLlm::test_正常系_単一の質問と回答 -v -s --pdb
```

## 継続的な改善

テスト結果や新しい要件に基づいて、定期的にテストケースを見直し、追加してください：

1. 新しいエッジケースの発見時
2. パフォーマンス要件の変更時
3. サポートする言語や文字セットの拡張時
4. LLMモデルの変更時
