"""Utility functions needed for both finetuning and testing."""

from transformers import (  # type: ignore
    Seq2SeqTrainingArguments,
)
import yaml  # type: ignore
from pathlib import Path
from typing import Dict, Any, List, Tuple

def read_config_file(file_name: str) -> Dict[str, Any]:
    """Reads YAML config from a config file.

    Args:
        file_name: the location where the config file is stored

    Returns:
        the contents of the YAML file
    """
    with open(file_name, "r") as file:
        config = yaml.safe_load(file)
    return config


def init_args(
    hyper_parameters: Dict[str, Any],
    output_dir: str,
    save_steps: int = 500,
) -> Seq2SeqTrainingArguments:
    """Initalize the hyperparameters for the model to be trained on.

    Args:
        hyper_parameters: Hyperparameters from config.
        output_dir: Where the model will be stored after training.
        save_steps: The number of steps before saving the model.

    Returns:
        The hyperparameters of the model.
    """
    print(output_dir)
    hyper_parameters["output_dir"] = Path(output_dir)
    hyper_parameters["learning_rate"] = float(hyper_parameters["learning_rate"])
    # hyper_parameters["save_steps"] = save_steps
    hyper_parameters["save_strategy"] = "epoch"
    hyper_parameters["evaluation_strategy"] = "epoch"
    hyper_parameters["metric_for_best_model"] = "eval_rouge1"
    hyper_parameters["greater_is_better"] = True
    hyper_parameters["load_best_model_at_end"] = True
    args = Seq2SeqTrainingArguments(**hyper_parameters)
    return args


def get_wandb_tags_finetune(config: Dict[str, Any]) -> Tuple[List[str], str]:
    """Gets the tags for wandb.

    Args:
        config: Model config.

    Returns:
        Tuple of wandb tags and dataset name.
    """
    if isinstance(config["train_data"], list):
        wandb_dataset_tags = [
            datataset.split("/")[-1] for datataset in config["train_data"]
        ]
        dataset_name = "_".join(wandb_dataset_tags)
    else:
        dataset_name = config["train_data"].split("/")[-1]
        wandb_dataset_tags = [dataset_name]
    return wandb_dataset_tags, dataset_name


def get_wandb_tags_test(config: Dict[str, Any]) -> Tuple[List[str], str]:
    """Gets the tags for wandb.

    Args:
        config: Model config.

    Returns:
        Tuple of wandb tags and dataset name.
    """
    if isinstance(config["test_data"], list):
        wandb_dataset_tags = [
            datataset.split("/")[-1] for datataset in config["test_data"]
        ]
        dataset_name = "_".join(wandb_dataset_tags)
    else:
        dataset_name = config["test_data"].split("/")[-1]
        wandb_dataset_tags = [dataset_name]
    return wandb_dataset_tags, dataset_name
