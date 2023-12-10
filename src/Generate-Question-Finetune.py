"""Finetune a Seq 2 Seq Language Model."""

import argparse
import glob
import logging
import os
import shutil
import typing
from pathlib import Path

import datasets  # type: ignore
import evaluate  # type: ignore
import nltk  # type: ignore
import numpy as np
import wandb
import yaml  # type: ignore
from transformers import AutoModelForSeq2SeqLM  # type: ignore
from transformers import (
    BloomForCausalLM,
    BloomTokenizerFast,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5TokenizerFast,
)

from Load_Data import load_data, load_datasets
from util import (
    compute_bert_score,
    compute_blue,
    compute_meteor,
    compute_rouge,
    get_wandb_tags,
    init_args,
    read_config_file,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
metrics: typing.Dict[str, bool] = {}


def preprocess_data(
    data: typing.Dict[str, list],
    tokenizer: T5TokenizerFast,
    max_input_length=1024,
    max_target_length=1024,
    use_prefix=False,
) -> typing.Dict[str, list]:
    """Converts data to tokenized data.

    Args:
        data: Data values which contain the keys input_text, prefix
          and target_text.
        max_input_length: Max length of the input string.
            Maximum length of (`int(1e30)`). Defaults to 512.
        max_target_length: Max length of the output string.
          Maximum length of (`int(1e30)`). Defaults to 512.

    Returns:
        Tokenizes the data (parameter).
    """
    if use_prefix:
        inputs = [
            pre + ": " + inp
            for inp, pre in zip(data["input_text"], data["prefix"])
        ]
    else:
        inputs = [inp for inp in data["input_text"]]
    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
    )
    labels = tokenizer(
        text_target=data["target_text"],
        padding="max_length",
        truncation=True,
        max_length=max_target_length,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred: EvalPrediction) -> typing.Dict[str, float]:
    """Computes a custom metric combining BLEU and ROUGE.

    Using both bleu and rouge scores, compute_metrics evaluates the
    model's predictions comparing it to the target. 4 kinds
    of rouge scores and 1 bleu score is returned.

    Args:
        eval_pred: Evaluation results from the test dataset.
        rouge: If rouge should be used as a metric.
        bleu: If bleu should be used as a metric.

    Returns:
        Rouge and Bleu scores if requested.
    """

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True
    )
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]

    bleu_rouge_score = {}

    compute_rouge(
        decoded_preds, decoded_labels, prediction_lens, bleu_rouge_score
    )

    compute_blue(
        decoded_preds, decoded_labels, prediction_lens, bleu_rouge_score
    )

    compute_bert_score(decoded_preds, decoded_labels, bleu_rouge_score)

    compute_meteor(decoded_preds, decoded_labels, bleu_rouge_score)

    # Add mean generated length
    return bleu_rouge_score


def init_trainer(
    train_args: Seq2SeqTrainingArguments,
    data_collator: DataCollatorForSeq2Seq,
    tokenized_datasets: datasets.DatasetDict,
    tokenizer: T5TokenizerFast,
    model: AutoModelForSeq2SeqLM,
) -> Seq2SeqTrainer:
    """Initalizes a Sequence to Sequence Trainer.

    Args:
        train_args: Training arguments such as hyper-parameters for the trainer.
        data_collator: Dynamically padded inputs and labels.
        tokenized_datasets: Data that the model will train/test on.
        tokenizer: Used to preprocess data.
        model: The model to train.

    Returns:
        Training loop for model.
    """
    return Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="location of a YAML config file",
        default="src/model_configs/config.yaml",
    )
    args = parser.parse_args()
    options = vars(args)

    config = read_config_file(options["config"])
    metrics = config["metrics"]
    # model_checkpoint = config["model_checkpoint"]
    wandb_dataset_tags, dataset_name = get_wandb_tags(config)
    wandb_tags = [
        "query generation",
        "question generation",
        config["model_checkpoint"].split("/")[-1],
    ]
    wandb_tags.extend(wandb_dataset_tags)
    wandb.init(
        tags=wandb_tags,
        project="question generation",
    )
    if "bloom" in config["model_checkpoint"]:
        tokenizer = BloomTokenizerFast.from_pretrained(
            config["model_checkpoint"], use_auth_token=True
        )
        model = BloomForCausalLM.from_pretrained(config["model_checkpoint"])
    else:
        tokenizer = T5TokenizerFast.from_pretrained(
            config["model_checkpoint"], use_auth_token=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config["model_checkpoint"]
        )
    if isinstance(config["data"], list):
        raw_dataset = load_datasets(config["data"])
    else:
        raw_dataset = load_data(config["data"])

    # tokenized_datasets = raw_dataset.map(preprocess_data, batched=True)
    tokenized_datasets = raw_dataset.map(
        lambda examples: preprocess_data(
            examples,
            tokenizer,
            max_input_length=512,
            max_target_length=512,
            use_prefix=False,
        ),
        batched=True,
    )
    model_name = config["model_checkpoint"].split("/")[-1] + "_" + dataset_name
    index = 0
    for dir in os.listdir(config["output_dir"]):
        if dir.startswith(model_name):
            index += 1
    model_out_dir = Path(config["output_dir"]) / (model_name + "_" + str(index))
    print(tokenized_datasets)
    train_dataset_size = len(tokenized_datasets["train"])
    batch_size = config["hyper parameters"]["per_device_train_batch_size"]
    steps_per_epoch = (
        train_dataset_size
        * config["hyper parameters"]["num_train_epochs"]
        // batch_size
    )
    args = init_args(
        config["hyper parameters"],
        model_out_dir,
        save_steps=steps_per_epoch,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = init_trainer(
        args, data_collator, tokenized_datasets, tokenizer, model
    )

    trainer.train()
    print(model_out_dir)
    checkpoints_dir = str(model_out_dir / "checkpoint-*")
    print(checkpoints_dir)
    for checkpoint_dir in glob.glob(checkpoints_dir):
        if checkpoint_dir != trainer.state.best_model_checkpoint:
            shutil.rmtree(checkpoint_dir)
    with open(os.path.join(model_out_dir, "data.yaml"), "w") as file:
        yaml.dump(config, file)
