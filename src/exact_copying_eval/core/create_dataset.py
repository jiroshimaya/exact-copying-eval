"""Dataset creation module for exact copying evaluation."""

import json
import logging
from pathlib import Path
from typing import Any
import random

from datasets import Dataset, load_dataset
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EvaluationItem(BaseModel):
    """Single evaluation item."""

    question: str
    context: str
    expected_answer: str


class EvaluationDataset(BaseModel):
    """Complete evaluation dataset."""

    items: list[EvaluationItem]
    metadata: dict[str, Any]


def load_jsquad() -> Dataset:
    """
    Load the JSQuAD dataset.

    Returns:
        Dataset: The loaded JSQuAD validation dataset.

    Raises:
        ValueError: If the loaded dataset is not of type Dataset.
    """
    logger.info("Loading JSQuAD dataset")
    dataset = load_dataset("sbintuitions/JSQuAD", split="validation")
    if not isinstance(dataset, Dataset):
        msg = "Loaded dataset is not of type Dataset."
        logger.error(msg)
        raise ValueError(msg)
    logger.info("JSQuAD dataset loaded successfully (length: %d)", len(dataset))
    return dataset


def remove_linebreaks(text: str) -> str:
    """
    Remove line breaks from the text.

    Args:
        text (str): The input text with potential line breaks.

    Returns:
        str: The text without line breaks.
    """
    return "".join(text.splitlines())


def create_context(dataset: Dataset, indices: list[int]) -> list[str]:
    """
    Create a context string from the dataset based on the provided indices.

    Args:
        dataset (Dataset): The dataset containing the context.
        indices (list[int]): List of indices to select from the dataset.

    Returns:
        str: The concatenated context string.
    """
    contexts = [dataset[i]["context"] for i in indices]
    contexts = [remove_linebreaks(context) for context in contexts]
    return contexts


def get_dummy_indices(answer_index: int, dataset_length: int) -> list[int]:
    """
    Get dummy indices for the given answer index.

    Args:
        answer_index (int): The index of the answer in the dataset.
        dataset_length (int): The length of the dataset.

    Returns:
        list[int]: List of dummy indices.
    """
    if answer_index <= dataset_length // 2:
        return [dataset_length // 4 * 3, dataset_length - 1]
    else:
        return [0, dataset_length // 4]

def convert_to_random_string(text: str) -> str:
    """
    Convert the text to a random japanese string with same length.
    Args:
        text (str): The input text.

    Returns:
        str: The converted random string.
    """
    # Use a separate random state for string conversion to avoid affecting dataset sampling
    string_random = random.Random(hash(text) % (2**32))  # Deterministic based on input text
    
    # Generate a random Japanese string (ひらがな・カタカナ) of the same length as the input text
    japanese_chars = (
      [chr(i) for i in range(0x3041, 0x3097)] +  # ひらがな
      [chr(i) for i in range(0x30A1, 0x30FB)]    # カタカナ
    )
    random_string = ''.join(string_random.choice(japanese_chars) for _ in range(len(text)))
    return random_string

def create_dataset_item(idx: int, dataset: Dataset) -> EvaluationItem:
    """
    Create an EvaluationItem from the dataset at the given index.

    Args:
        idx (int): The index of the item in the dataset.
        dataset (Dataset): The dataset containing the item.

    Returns:
        EvaluationItem: The created evaluation item.
    """
    question = dataset[idx]["question"]
    dummy_indices = get_dummy_indices(idx, len(dataset))
    context_indices = [dummy_indices[0], idx, dummy_indices[1]]
    contexts = create_context(dataset, context_indices)
    
    # Convert expected answer to random string if needed
    expected_answer = contexts[1]
    
    return EvaluationItem(
        question=question,
        context="\n".join(contexts),
        expected_answer=expected_answer
    )

def create_dataset_random_item(idx: int, dataset: Dataset) -> EvaluationItem:
    """
    Create an EvaluationItem with random strings from the dataset at the given index.

    Args:
        idx (int): The index of the item in the dataset.
        dataset (Dataset): The dataset containing the item.

    Returns:
        EvaluationItem: The created evaluation item with random strings.
    """
    question = dataset[idx]["question"]
    dummy_indices = get_dummy_indices(idx, len(dataset))
    context_indices = [dummy_indices[0], idx, dummy_indices[1]]
    contexts = create_context(dataset, context_indices)
    contexts = [convert_to_random_string(context) for context in contexts]
    
    # Convert expected answer to random string
    expected_answer = contexts[1]
    
    return EvaluationItem(
        question=question,
        context="\n".join(contexts),
        expected_answer=expected_answer
    )

def create_evaluation_dataset(
    *, 
    nums: list[int] = [10, 100, -1], 
    output_dir: str | None = None,
    use_random_strings: bool = False,
    seed: int | None = None
) -> list[str]:
    """
    Create evaluation datasets and save to JSON files.

    Args:
        nums (list[int]): List of numbers of examples to include in each dataset.
                         -1 means all examples.
        output_dir (str | None): Output directory for the datasets. If None, uses current directory.
        use_random_strings (bool): Whether to convert expected answers to random strings.
        seed (int | None): Random seed for reproducible results. If None, uses default random behavior.

    Returns:
        list[str]: Paths to the created JSON files.
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        logger.info("Random seed set to: %d", seed)
    logger.info("Starting dataset creation (nums: %s, use_random_strings: %s)", nums, use_random_strings)

    # Load original dataset
    ds = load_jsquad()
    full_length = len(ds)
    logger.info("Loaded JSQuAD dataset (length: %d)", full_length)

    # Create full evaluation items first
    logger.info("Creating full dataset with all %d items", full_length)
    full_items = []

    for i in range(full_length):
        # Get question and expected answer
        if use_random_strings:
            item = create_dataset_random_item(i, ds)
        else:
            item = create_dataset_item(i, ds)
        full_items.append(item)
        if i % 100 == 0:
            logger.debug("Processed items: %d/%d", i, full_length)

    logger.info("Full dataset created with %d items", len(full_items))

    # Create and save datasets for each specified number
    output_files = []
    from datetime import datetime
    timestamp = datetime.now().isoformat()

    for num in nums:
        # Determine actual number of examples
        actual_num = min(num, full_length) if num > 0 else full_length
        logger.info("Creating dataset with %d examples (requested: %d)", actual_num, num)

        # Select items randomly
        if actual_num == full_length:
            selected_items = full_items[:]  # Use all items if requesting all
        else:
            selected_items = random.sample(full_items, actual_num)

        # Create dataset with metadata
        metadata = {
            "num_examples": actual_num,
            "source_dataset": "sbintuitions/JSQuAD",
            "split": "validation",
            "created_at": timestamp,
            "seed": seed,
        }

        dataset = EvaluationDataset(items=selected_items, metadata=metadata)

        # Determine output path
        suffix = "_random" if use_random_strings else ""
        if output_dir is None:
            current_output_path = f"evaluation_dataset_{actual_num}{suffix}.json"
        else:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            current_output_path = str(output_dir_path / f"evaluation_dataset_{actual_num}{suffix}.json")

        output_file = Path(current_output_path)

        # Save to JSON
        logger.info("Saving dataset to file: %s", str(output_file))
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(dataset.model_dump(), f, ensure_ascii=False, indent=2)

        output_files.append(str(output_file))
        logger.info("Dataset saved (output_path: %s, num_items: %d)", str(output_file), len(selected_items))

    logger.info("All datasets created successfully: %s", output_files)
    return output_files


def load_evaluation_dataset(file_path: str) -> EvaluationDataset:
    """
    Load evaluation dataset from JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        EvaluationDataset: The loaded evaluation dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    logger.info("Loading evaluation dataset: %s", file_path)

    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        msg = f"Evaluation dataset file not found: {file_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        with file_path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = EvaluationDataset.model_validate(data)
        logger.info("Evaluation dataset loaded successfully (num_items: %d)", len(dataset.items))
        return dataset

    except Exception as e:
        msg = f"Failed to load evaluation dataset: {e}"
        logger.error("Failed to load evaluation dataset: %s", str(e))
        raise ValueError(msg) from e


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Create evaluation dataset from JSQuAD."
    )
    parser.add_argument(
        "--nums",
        type=int,
        nargs="+",
        default=[10, 100, -1],
        help="Numbers of examples to include in the datasets. Use -1 for all examples.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Output directory for the datasets."
    )
    parser.add_argument(
        "--random-strings",
        action="store_true",
        help="Convert expected answers to random Japanese strings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results.",
    )

    args = parser.parse_args()

    output_paths = create_evaluation_dataset(
        nums=args.nums, 
        output_dir=args.output_dir,
        use_random_strings=args.random_strings,
        seed=args.seed
    )
    print(f"Datasets created: {output_paths}")
