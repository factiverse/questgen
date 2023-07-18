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
import numpy as np
import wandb
from pathlib import Path


def load_data(train_file: str, test_file: str) -> datasets.DatasetDict:
    """Loads data from train and test json files.

    Args:
        train_file (str): path to train data
        test_file (str): path to test data

    Returns:
        DatasetDict: train and test data
    """
    raw_dataset_train = datasets.load_dataset("json", data_files=train_file)
    raw_dataset_test = datasets.load_dataset("json", data_files=test_file)
    raw_dataset = raw_dataset_train
    raw_dataset["train"] = raw_dataset["train"].select(range(0, 32))
    raw_dataset["test"] = raw_dataset_test["train"].select(range(0, 16))
    return raw_dataset


def preprocess_function(
    examples: dict, max_input_length=512, max_target_length=512
) -> dict:
    """Convert example to tokenized.

    Args:
        examples (dict): data values which contain the keys input_text, prefix
          and target_text.
        max_input_length (int, optional): max length of the input string.
          Defaults to 512.
        max_target_length (int, optional): max length of the output string.
          Defaults to 512.

    Returns:
        dict: tokenized version of example
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
        eval_pred (EvalPrediction):
            evaluation results from the test dataset.

    Returns:
        dict: rouge and bleu scores
    """
    metric_rouge = evaluate.load("rouge")
    metric_bleu = evaluate.load("bleu")
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
    result_rouge = metric_rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True,
    )
    result_bleu = metric_bleu.compute(
        predictions=decoded_preds, references=decoded_labels
    )

    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]

    result_rouge["gen_len"] = np.mean(prediction_lens)
    result_bleu["gen_len"] = np.mean(prediction_lens)

    # print(result_bleu, "\n",result_rouge)
    bleu_rouge_score = {
        "rouge1": result_rouge["rouge1"],
        "rouge2": result_rouge["rouge2"],
        "rougeL": result_rouge["rougeL"],
        "rougeLsum": result_rouge["rougeLsum"],
        "bleu": result_bleu["bleu"],
    }
    return {k: round(v, 4) for k, v in bleu_rouge_score.items()}


def init_args(batch_size=8) -> Seq2SeqTrainingArguments:
    """Initalize the hyperparameters for the model to be trained on.

    Args:
        batch_size (int, optional): the batch size of the model. Defaults to 8.

    Returns:
        Seq2SeqTrainingArguments: the hyperparameters of the model.
    """
    args = Seq2SeqTrainingArguments(
        output_dir=Path(wandb.run.dir) / "model",  # type: ignore
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
        overwrite_output_dir=True,
    )
    wandb.config.update(args.to_dict())
    return args


def init_trainer(
    args: Seq2SeqTrainingArguments,
    data_collator: DataCollatorForSeq2Seq,
    tokenized_datasets: datasets.DatasetDict,
) -> Seq2SeqTrainer:
    """Initalizes a Sequence to Sequence Trainer.

    Args:
        args (Seq2SeqTrainingArguments):
            training arguments such as hyper parameters for the trainer
        data_collator (DataCollatorForSeq2Seq):
            dynamically padded inputs and labels
        tokenized_datasets (datasets.DatasetDict):
            data that the model will train/test on

    Returns:
        Seq2SeqTrainer: training loop for model
    """
    return Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


if __name__ == "__main__":
    model_checkpoint = "T5-small"
    wandb_tags = ["query generation", "question generation"]
    # Hi
    # Hello
    wandb.init(
        tags=wandb_tags,
        project="question generation " + model_checkpoint.split("/")[-1],
    )
    #     {
    #         "tags": wandb_tags,
    #         "project": "question generation " +
    #  model_checkpoint.split("/")[-1],
    #     }
    # )
    tokenizer = T5TokenizerFast.from_pretrained(
        model_checkpoint, use_auth_token=True
    )
    raw_dataset = load_data("data/final/train.json", "data/final/test.json")
    tokenized_datasets = raw_dataset.map(preprocess_function, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    args = init_args(batch_size=8)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = init_trainer(args, data_collator, tokenized_datasets)
    trainer.train()

# config.yaml file
