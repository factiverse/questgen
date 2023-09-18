"""Finetune a Seq 2 Seq Language Model."""

import datasets  # type: ignore
import evaluate  # type: ignore
from transformers import (  # type: ignore
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    T5TokenizerFast,
    EvalPrediction,
)
import nltk  # type: ignore
import wandb
from pathlib import Path
import yaml  # type: ignore
import argparse
import os
import logging
import numpy as np
import typing

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
metrics: typing.Dict[str, bool] = {}


def read_config_file(file_name: str) -> typing.Dict[str, typing.Any]:
    """Reads YAML config from a config file.

    Args:
        file_name: The location where the config file is stored.

    Returns:
        The contents of the YAML file.
    """
    with open(file_name, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_data(train_test_dir: str) -> datasets.DatasetDict:
    """Loads data from train and test json files.

    Args:
        train_test_dir: The path to train and test data.

    Returns:
        The training and test data. Returns None if data does not exist.
    """
    train_file = os.path.join(train_test_dir, "train.json")
    test_file = os.path.join(train_test_dir, "test.json")

    if os.path.exists(train_file) and os.path.exists(test_file):
        raw_dataset_train = datasets.load_dataset("json", data_files=train_file)
        raw_dataset_test = datasets.load_dataset("json", data_files=test_file)
        raw_dataset = raw_dataset_train
        raw_dataset["test"] = raw_dataset_test["train"]
        return raw_dataset
    else:
        logger.error(
            FileNotFoundError(
                f"The directory {train_test_dir}\
                    does not contain 'train.json' and 'test.json'"
            ),
            exc_info=True,
        )


def preprocess_data(
    data: typing.Dict[str, list], max_input_length=512, max_target_length=512
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
    inputs = [
        pre + ": " + inp for inp, pre in zip(data["input_text"], data["prefix"])
    ]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True
    )
    labels = tokenizer(
        text_target=data["target_text"],
        max_length=max_target_length,
        truncation=True,
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
    global metrics
    rouge = metrics["rouge"]
    bleu = metrics["bleu"]

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

    if rouge:
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

    if bleu:
        metric_bleu = evaluate.load("bleu")
        result_bleu = metric_bleu.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        result_bleu["gen_len"] = np.mean(prediction_lens)

        bleu_rouge_score["bleu"] = result_bleu["bleu"]

    # Add mean generated length
    return bleu_rouge_score


def init_args(
    hyper_parameters: typing.Dict[str, typing.Any],
    output_dir: str,
    model_checkpoint: str,
) -> Seq2SeqTrainingArguments:
    """Initalize the hyperparameters for the model to be trained on.

    Args:
        hyper_parameters: Hyperparameters from config.
        output_dir: Where the model will be stored after training.
        model_checkpoint: Name of the model we will finetune.

    Returns:
        The hyperparameters of the model.
    """
    print(output_dir)
    hyper_parameters["output_dir"] = Path(output_dir)
    hyper_parameters["learning_rate"] = float(hyper_parameters["learning_rate"])
    args = Seq2SeqTrainingArguments(**hyper_parameters)
    wandb.config.update(args.to_dict())
    return args


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
    model_checkpoint = config["model_checkpoint"]
    wandb_tags = [
        "query generation",
        "question generation",
        model_checkpoint.split("/")[-1],
        config["data"].split("/")[-1],
    ]
    wandb.init(
        tags=wandb_tags,
        project="question generation " + model_checkpoint.split("/")[-1],
    )
    # print("*************************************************",model_checkpoint)
    tokenizer = T5TokenizerFast.from_pretrained(
        model_checkpoint, use_auth_token=True
    )
    raw_dataset = load_data(config["data"])
    tokenized_datasets = raw_dataset.map(preprocess_data, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    index = 0
    for dir in os.listdir(config["output_dir"]):
        if dir.startswith(model_checkpoint):
            index += 1
    model_out_dir = Path(config["output_dir"]) / (
        model_checkpoint.split("/")[-1] + "_" + str(index)
    )

    args = init_args(
        config["hyper parameters"],
        model_out_dir,
        model_checkpoint.split("/")[-1],
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = init_trainer(
        args, data_collator, tokenized_datasets, tokenizer, model
    )

    trainer.train()

    with open(os.path.join(model_out_dir, "data.yaml"), "w") as file:
        yaml.dump(config, file)
