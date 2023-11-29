"""Utility functions needed for both finetuning and testing"""

from transformers import (  # type: ignore
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import yaml  # type: ignore
import argparse
import os
import logging
from pathlib import Path
import typing
import numpy as np
import wandb
import nltk  # type: ignore
import datasets  # type: ignore
import evaluate  # type: ignore
import torch
import json


def read_config_file(file_name: str) -> typing.Dict[str, typing.Any]:
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
    hyper_parameters: typing.Dict[str, typing.Any],
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
    wandb.config.update(args.to_dict())
    return args
