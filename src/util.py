"""Utility functions needed for both finetuning and testing."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import evaluate  # type: ignore
import nltk  # type: ignore
import numpy as np
import yaml  # type: ignore
from transformers import Seq2SeqTrainingArguments  # type: ignore


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


def get_wandb_tags(config: Dict[str, Any]) -> Tuple[List[str], str]:
    """Gets the tags for wandb.

    Args:
        config: Model config.

    Returns:
        Tuple of wandb tags and dataset name.
    """
    if isinstance(config["data"], list):
        wandb_dataset_tags = [
            datataset.split("/")[-1] for datataset in config["data"]
        ]
        dataset_name = "_".join(wandb_dataset_tags)
    else:
        dataset_name = config["data"].split("/")[-1]
        wandb_dataset_tags = [dataset_name]
    return wandb_dataset_tags, dataset_name


def compute_blue(
    decoded_preds, decoded_labels, prediction_lens, score_dict: Dict[str, float]
) -> None:
    """Computes the BLEU score.

    Args:
        decoded_preds: Decoded predictions.
        decoded_labels: Decoded labels.
        prediction_lens: Prediction lengths.
    """
    metric_bleu = evaluate.load("bleu")
    result_bleu = metric_bleu.compute(
        predictions=decoded_preds, references=decoded_labels
    )
    result_bleu["gen_len"] = np.mean(prediction_lens)

    score_dict["bleu"] = result_bleu["bleu"]


def compute_rouge(
    decoded_preds, decoded_labels, prediction_lens, bleu_rouge_score
) -> None:
    """Computes the ROUGE score.

    Args:
        decoded_preds: Decoded predictions.
        decoded_labels: Decoded labels.
        prediction_lens: Prediction lengths.
        bleu_rouge_score: Dictionary to store the ROUGE score.
    """
    metric_rouge = evaluate.load("rouge")
    result_rouge = metric_rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True,
    )

    result_rouge["gen_len"] = np.mean(prediction_lens)

    bleu_rouge_score["rouge1"] = result_rouge["rouge1"]
    bleu_rouge_score["rouge2"] = result_rouge["rouge2"]
    bleu_rouge_score["rougeL"] = result_rouge["rougeL"]
    bleu_rouge_score["rougeLsum"] = result_rouge["rougeLsum"]


def compute_meteor(decoded_preds, decoded_labels, bleu_rouge_score) -> None:
    """Computes the METEOR score.

    Args:
        decoded_preds: Decoded predictions.
        decoded_labels: Decoded labels.
        bleu_rouge_score: Dictionary to store the METEOR score.
    """
    bleu_rouge_score["meteor"] = evaluate.load("meteor").compute(
        predictions=decoded_preds, references=decoded_labels
    )["meteor"]


def compute_bert_score(decoded_preds, decoded_labels, bleu_rouge_score) -> None:
    """Computes the BERT score.

    Args:
        decoded_preds: Decoded predictions.
        decoded_labels: Decoded labels.
        bleu_rouge_score: Dictionary to store the BERT score.
    """
    bert_score = evaluate.load("bertscore").compute(
        predictions=decoded_preds, references=decoded_labels, lang="en"
    )

    bleu_rouge_score["bert_score_f1"] = np.mean(bert_score["f1"])
    bleu_rouge_score["bert_score_precision"] = np.mean(bert_score["precision"])
    bleu_rouge_score["bert_score_recall"] = np.mean(bert_score["recall"])
