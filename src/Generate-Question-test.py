"""Test and Evaluate Seq2Seq Language Model."""

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
import nltk  # type: ignore
import datasets  # type: ignore
import evaluate  # type: ignore
import torch

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
        file_name: the location where the config file is stored

    Returns:
        the contents of the YAML file
    """
    with open(file_name, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_model_checkpoint_path(config: dict) -> str:
    """Get last checkpoint to test on most recently trained model.

    Args:
        config: config file for training model
    """
    model_zoo = list(os.listdir(config["output_dir"]))
    for m in model_zoo:
        if not m.startswith(config["model_checkpoint"].split("/")[-1]):
            model_zoo.remove(m)
    model_zoo.sort()
    model = "models/" + model_zoo[-1]
    checkpoints = os.listdir(model)
    checkpoints.sort()
    model_checkpoint = os.path.join(model, checkpoints[-1])
    logger.info(f"Using directory {model_checkpoint} for testing")
    return model_checkpoint


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
    model_out_dir = model_checkpoint + "_" + str(len(os.listdir(output_dir)))
    hyper_parameters["output_dir"] = Path(output_dir) / model_out_dir
    hyper_parameters["learning_rate"] = float(hyper_parameters["learning_rate"])
    args = Seq2SeqTrainingArguments(**hyper_parameters)
    return args


def predict(
    to_predict: str, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer
) -> str:
    """Generates a response from the loaded model.

    Args:
        to_predict: Fed into the model as input.
        model: The Seq2Seq model.
        tokenizer: Used to tokenize input for the Seq2Seq model.

    Returns:
        the response generated by the requested Seq2Seq model.
    """
    input_ids = tokenizer(to_predict, return_tensors="pt")
    print(input_ids["input_ids"])
    outputs = model.generate(input_ids["input_ids"])
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

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


def load_data(test_dir: str) -> datasets.DatasetDict:
    """Loads data from test json files.

    Args:
        test_dir: The path to test data.

    Returns:
        The test data. Returns None if data does not exist.
    """
    test_file = os.path.join(test_dir, "test.json")

    if os.path.exists(test_file):
        raw_dataset_test = datasets.load_dataset("json", data_files=test_file)
        raw_dataset_test["test"] = raw_dataset_test["train"]
        del raw_dataset_test["train"]
        # raw_dataset_test["test"].to(device)
        return raw_dataset_test
    else:
        logger.error(
            FileNotFoundError(
                f"The directory {test_dir} \
                    does not contain 'test.json'"
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

    # model_inputs = tokenizer(
    #     inputs, max_length=max_input_length, truncation=True
    # )

    model_inputs = tokenizer.batch_encode_plus(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # labels = tokenizer(
    #     text_target=data["target_text"],
    #     max_length=max_target_length,
    #     truncation=True,
    # )

    labels = tokenizer.batch_encode_plus(
        data["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    model_inputs["labels"] = labels["input_ids"]
    model_inputs.to(device)
    return model_inputs


def eval(
    val_data_path: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    args: Seq2SeqTrainingArguments,
) -> typing.Dict[str, float]:
    """Evaluate the Seq2Seq model.

    Arg:
        val_data_path: The path to the evaluation dataset
        model: The model to evaluate
        tokenizer: The string tokenizer.
            Converts strings to tokenized strings.
        args: Testing Arguments

    Returns:
        The Rouge and Bleu scores of the model.
    """
    raw_dataset = load_data(val_data_path)
    tokenized_datasets = raw_dataset.map(preprocess_data, batched=True)

    trainer = Seq2SeqTrainer(
        model,
        args,
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    results = trainer.evaluate(tokenized_datasets["test"])

    print(results)
    # outputs = model.generate(
    #     torch.tensor(tokenized_datasets["test"]["input_ids"],device=device)
    # )
    # model_gen = tokenizer.decode(outputs)
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model_checkpoint = get_model_checkpoint_path(config)

    args = init_args(
        config["hyper parameters"],
        config["output_dir"],
        model_checkpoint.split("/")[-1],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, use_auth_token=True, device=device
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model.to(device)

    logger.info(
        "type a claim to see how the model responds \
                 (q+enter to quit)"
    )

    eval(config["data"], model, tokenizer, args)
    # while True:
    #     inp = input()
    #     if inp == "q":
    #         break
    #     input_ids = tokenizer(inp, return_tensors="pt")
    #     print(input_ids["input_ids"])
    #     outputs = model.generate(input_ids["input_ids"])
    #     print(tokenizer.decode(outputs[0], skip_special_tokens=True))
