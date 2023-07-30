"""Train a Seq 2 Seq Language Model."""

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

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
metrics: dict[str, bool] = {}


def read_config_file(file_name: str) -> dict:
    """Reads YAML config from a config file.

    Args:
        file_name: the location where the config file is stored

    Returns:
        the contents of the YAML file
    """
    with open(file_name, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_data(train_test_dir) -> datasets.DatasetDict:
    """Loads data from train and test json files.

    Args:
        train_test_dir: path to train and test data

    Returns:
        train and test data
    """
    train_file = os.path.join(train_test_dir, "train.json")
    test_file = os.path.join(train_test_dir, "test.json")

    if os.path.exists(train_file) and os.path.exists(train_file):
        raw_dataset_train = datasets.load_dataset("json", data_files=train_file)
        raw_dataset_test = datasets.load_dataset("json", data_files=test_file)
        raw_dataset = raw_dataset_train
        raw_dataset["train"] = raw_dataset["train"]
        raw_dataset["test"] = raw_dataset_test["train"]
        return raw_dataset
    else:
        logger.error(
            FileNotFoundError(
                f"The directory {train_test_dir}\
                               does not contain 'train.json' and \
                               'test.json'"
            ),
            exc_info=True,
        )


def preprocess_function(
    examples: dict, max_input_length=512, max_target_length=512
) -> dict:
    """Convert example to tokenized.

    Args:
        examples: data values which contain the keys input_text, prefix
          and target_text.
        max_input_length: max length of the input string.
          Defaults to 512.
        max_target_length: max length of the output string.
          Defaults to 512.

    Returns:
        tokenized version of example
    """
    inputs = [
        pre + ": " + inp
        for inp, pre in zip(examples["input_text"], examples["prefix"])
    ]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True
    )
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """Does a custom metric computation.

    Using both bleu and rouge scores, compute_metrics evaluates the
    model's predictions comparing it to the target. 4 kinds
    of rouge scores and 1 bleu score is returned.

    Args:
        eval_pred: evaluation results from the test dataset.
        rouge: if rouge should be used as a metric
        bleu: if bleu should be used as a metric

    Returns:
        rouge and bleu scores if requested
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

        bleu_rouge_score["rouge1"] = result_rouge["rouge1"]
        bleu_rouge_score["rouge2"] = result_rouge["rouge2"]
        bleu_rouge_score["rougeL"] = result_rouge["rougeL"]
        bleu_rouge_score["rougeLsum"] = result_rouge["rougeLsum"]

    if bleu:
        metric_bleu = evaluate.load("bleu")
        result_bleu = metric_bleu.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        result_rouge["gen_len"] = np.mean(prediction_lens)
        result_bleu["gen_len"] = np.mean(prediction_lens)
        bleu_rouge_score["bleu"] = result_bleu["bleu"]

    # Add mean generated length
    return {k: round(v, 4) for k, v in bleu_rouge_score.items()}


def init_args(
    hyper_parameters: dict, output_dir: str, model_checkpoint: str
) -> Seq2SeqTrainingArguments:
    """Initalize the hyperparameters for the model to be trained on.

    Args:
        hyper_parameters: hyperparameters from config
        output_dir: where the model will be stored after training
        model_checkpoint: name of the model we will finetune

    Returns:
        the hyperparameters of the model.
    """
    hyper_parameters["output_dir"] = Path(output_dir) / model_checkpoint
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
        train_args: training arguments such as hyper-parameters for the trainer
        data_collator: dynamically padded inputs and labels
        tokenized_datasets: data that the model will train/test on
        tokenizer: used to preprocess data
        model: the model to train

    Returns:
        training loop for model
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
        default="src/QuestionGeneration/T5/config.yaml",
    )
    args = parser.parse_args()
    options = vars(args)

    config = read_config_file(options["config"])
    metrics = config["metrics"]
    model_checkpoint = config["model_checkpoint"]
    wandb_tags = ["query generation", "question generation"]
    wandb.init(
        tags=wandb_tags,
        project="question generation " + model_checkpoint.split("/")[-1],
    )
    tokenizer = T5TokenizerFast.from_pretrained(
        model_checkpoint, use_auth_token=True
    )
    raw_dataset = load_data(config["data"])
    tokenized_datasets = raw_dataset.map(preprocess_function, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    args = init_args(
        config["hyper parameters"],
        config["output_dir"],
        model_checkpoint.split("/")[-1],
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = init_trainer(
        args, data_collator, tokenized_datasets, tokenizer, model
    )
    trainer.train()

# # config.yaml file
