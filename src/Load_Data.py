"""Loads the nessary data for finetuning and testing."""

import os
from datasets import DatasetDict
import datasets  # type: ignore
import typing
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_data(data_dir: str) -> datasets.DatasetDict:
    """Loads data from test json files.

    Args:
        data_dir: The path to test data.

    Returns:
        The test data. Returns None if data does not exist.
    """
    train_file = os.path.join(data_dir, "train.json")
    test_file = os.path.join(data_dir, "test.json")

    if os.path.exists(train_file) and os.path.exists(test_file):
        raw_dataset_train = datasets.load_dataset("json", data_files=train_file)
        raw_dataset_test = datasets.load_dataset("json", data_files=test_file)
        raw_dataset = raw_dataset_train
        raw_dataset["test"] = raw_dataset_test["train"]
        return raw_dataset
    else:
        logger.error(
            FileNotFoundError(
                f"The directory {data_dir}\
                    does not contain 'train.json' and 'test.json'"
            ),
            exc_info=True,
        )


def load_datasets(dataset_paths: typing.List[str]) -> datasets.DatasetDict:
    """Loads a list of datasets.

    Args:
        dataset_paths: List of dataset paths.

    Returns:
        The training and test data. Returns None if data does not exist.
    """
    merged_train_datasets = []
    merged_test_datasets = []
    for data in dataset_paths:
        loaded_dataset = load_data(data)
        # if "train" in loaded_dataset:
        merged_train_datasets.append(loaded_dataset["train"])
        # if "test" in loaded_dataset:
        merged_test_datasets.append(loaded_dataset["test"])
    merged_train_dataset = datasets.concatenate_datasets(merged_train_datasets)
    merged_test_dataset = datasets.concatenate_datasets(merged_test_datasets)
    print("Merged Train Dataset Size:", len(merged_train_dataset))
    print("Merged Test Dataset Size:", len(merged_test_dataset))

    merged_dataset = DatasetDict(
        {"train": merged_train_dataset, "test": merged_test_dataset}
    )
    return merged_dataset
