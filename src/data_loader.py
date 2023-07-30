"""Loads data from one text file into a more easily accessible format."""

import json
import random
import numpy as np
import torch
import random
import csv
import logging
import os

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def rename_columns(file_name: str, renamed_columns: dict) -> None:
    """Opens a JSON file and renames the columns of the file.

    Args:
        file_name: The path of the file to get data from
        renamed_columns: The original name to final name
            key value pairing in a dict
    """
    renamed_column_data = []

    with open(file_name, "r") as file:
        data = json.load(file)

    for i in range(len(data)):
        d = data[i]
        new_d = {}
        for k, v in renamed_columns.items():
            new_d[v] = d[k]
        renamed_column_data.append(new_d)

    with open(file_name, "w") as file:
        json.dump(renamed_column_data, file)


def prepare_data(data_dir: str) -> None:
    """Prepares data.

       Converts the column names and introduces the prefix column.

    Args:
        data_dir: the directory with the train and test files
    """
    files = os.listdir(data_dir)
    for file_name in files:
        if file_name[-5:] != ".json":
            logger.error(
                f"file '{file_name}' found in '{files}'. \
                         All files must end with the `.json` file extension"
            )
            raise FileExistsError("all files must be of type json")

    for f in files:
        print(f)
        with open(os.path.join(data_dir, f), "r") as file:
            text = json.load(file)  # claim, query
            new_data = []
            prefixes = {
                1: "generate question",
                2: "rewrite the claim",
                0: "generate a query",
            }
            for i in range(len(text)):
                new_data.append(
                    {
                        "input_text": text[i]["claim"],
                        "target_text": text[i]["query"],
                        "prefix": prefixes[i % 3],
                    }
                )
        with open(os.path.join(data_dir, f), "w") as file:
            json.dump(new_data, file, indent=4)


def train_test_split(data_file_name: str, new_data_dir: str) -> None:
    """Gets data and splits it into train and test json files.

    The split is defaulted to an 80/20 split.

    Args:
        data_file_name: The path of the file to get data from
        new_data_dir: The path where `train.json` and `test.json` will be stored
    """
    with open(data_file_name, "r") as file:
        text = json.load(file)  # claim, query
    new_data = []
    prefixes = {
        1: "generate question",
        2: "rewrite the claim",
        0: "generate a query",
    }
    for i in range(len(text)):
        new_data.append(
            {
                "input_text": text[i]["claim"],
                "target_text": text[i]["query"],
                "prefix": prefixes[i % 3],
            }
        )
    random.shuffle(new_data)
    len_data = len(new_data)
    train_data_len = int(0.8 * len_data)
    val_data_len = int(0.2 * len_data)
    # test_data_len = 0.2*len_data # assume that it will be the rest
    train_data = new_data[:train_data_len]
    val_data = new_data[train_data_len : train_data_len + val_data_len]
    # test_data = new_data[train_data_len+val_data_len:]

    with open(new_data_dir + "train.json", "w") as file:
        json.dump(train_data, file, indent=4)
    with open(new_data_dir + "test.json", "w") as file:
        json.dump(val_data, file, indent=4)
    # with open(new_data_file_name+'val.json', 'w') as file:
    #     json.dump(test_data,file, indent=4)


def csv_to_json(csv_file: str, columns: list) -> None:
    """Converts a CSV file to a JSON file.

    Args:
        csv_file: the path to teh csv file
        columns: the colmns to convert into json
    """
    json_data = []
    with open(csv_file, "r") as file:
        csv_data = csv.DictReader(file)
        for d in csv_data:
            json_dict = {}
            for c in columns:
                json_dict[c] = d[c]
            json_data.append(json_dict)
    logger.info(f"opened '{csv_file}' and read data")

    with open(csv_file[:-3] + "json", "w") as file:
        json.dump(json_data, file, indent=4)

    logger.info(f"wrote data to '{csv_file[:-3]}json'")


def set_seed(seed: int = 42) -> None:
    """Sets the seed of different random generating libraries.

    Args:
        seed (optional): The seed to set. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    set_seed()
    prepare_data("data/online-claims")
