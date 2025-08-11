#!/usr/bin/env python
"""evaluation.mdの結果を再現するためのシンプルなサブプロセス評価スクリプト.

使用例:
    # 全ての組み合わせを実行
    uv run python evaluation/reproduce_scripts/run_evaluate.py

    # dry-runで実行予定のコマンドを確認
    uv run python evaluation/reproduce_scripts/run_evaluate.py --dry-run
"""

import argparse
import subprocess  # nosec B404 - subprocess usage is intentional and safe
import sys
from pathlib import Path

# プロジェクトルートを特定
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# evaluation.mdで定義されたモデルとデータセット/プロンプトタイプの組み合わせ
MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "o4-mini",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
]

DATASET_AND_PROMPT_TYPES = [
    ("evaluation/dataset/evaluation_dataset_100.json", "qa"),
    ("evaluation/dataset/evaluation_dataset_100.json", "simple"),
    ("evaluation/dataset/evaluation_dataset_100_random.json", "simple"),
]


def run_evaluation(
    model: str,
    dataset: str,
    prompt_type: str,
    output_dir: str,
    batch_size: int = 100,
    dry_run: bool = False,
) -> bool:
    """単一の評価を実行."""
    command = [
        "uv",
        "run",
        "python",
        str(PROJECT_ROOT / "src" / "exact_copying_eval" / "core" / "evaluate.py"),
        "--dataset",
        dataset,
        "--model",
        model,
        "--prompt_type",
        prompt_type,
        "--output_dir",
        output_dir,
        "--batch_size",
        str(batch_size),
    ]

    description = f"{model} with {Path(dataset).name} ({prompt_type})"

    if dry_run:
        print(f"[DRY RUN] {description}")
        print(f"  Command: {' '.join(command)}")
        return True

    print(f"Running: {description}")

    try:
        result = subprocess.run(  # nosec B603 - command is constructed safely
            command,
            cwd=PROJECT_ROOT,
            timeout=1800,
            check=False,  # 30分のタイムアウト
        )

        if result.returncode == 0:
            print(f"✓ Completed: {description}")
            return True
        else:
            print(f"✗ Failed: {description} (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout: {description}")
        return False
    except Exception as e:
        print(f"✗ Error: {description} - {e}")
        return False


def main() -> int:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="Run evaluation for all model/dataset combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/result",
        help="Output directory for results (default: evaluation/result)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for evaluation (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands that would be executed without running them",
    )

    args = parser.parse_args()

    # データセットディレクトリの存在確認
    dataset_dir = PROJECT_ROOT / "evaluation" / "dataset"
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return 1

    # 全ての組み合わせを実行
    total_tasks = 0
    successful = 0

    for model in MODELS:
        for dataset_path, prompt_type in DATASET_AND_PROMPT_TYPES:
            total_tasks += 1
            dataset_full_path = str(PROJECT_ROOT / dataset_path)

            success = run_evaluation(
                model=model,
                dataset=dataset_full_path,
                prompt_type=prompt_type,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )

            if success:
                successful += 1

    # 結果サマリーを表示
    print("\n=== Evaluation Summary ===")
    print(f"Total tasks: {total_tasks}")
    print(f"Successful: {successful}")
    print(f"Failed: {total_tasks - successful}")

    return 0 if successful == total_tasks else 1


if __name__ == "__main__":
    sys.exit(main())
