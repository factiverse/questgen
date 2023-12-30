"""Quantizes a model and pushes it to the Hugging Face Hub."""
import argparse
import math
import os
import time
from typing import Any, Dict
import pandas as pd  # type: ignore
from optimum.onnxruntime import (  # type: ignore
    ORTModelForSequenceClassification,
    ORTQuantizer,
)
from optimum.onnxruntime.configuration import (  # type: ignore
    QuantFormat,
    QuantizationConfig,
    QuantizationMode,
)
from sklearn.metrics import classification_report  # type: ignore
from transformers import AutoTokenizer, pipeline  # type: ignore


def quantize_model(model_path: str, output_path: str) -> None:
    """Quantizes a model and saves it to a given output path.

    Args:
        model_path: The path to the model to quantize.
        output_path: The path to save the quantized model.
    """
    model = ORTModelForSequenceClassification.from_pretrained(
        model_path, export=True
    )

    quantizer = ORTQuantizer.from_pretrained(model)
    qconfig = QuantizationConfig(
        is_static=False,
        format=QuantFormat.QOperator,
        mode=QuantizationMode.IntegerOps,
        per_channel=True,
        reduce_range=True,
        operators_to_quantize=[
            "MatMul",
            "Add",
        ],
    )
    quantizer.quantize(
        save_dir=output_path,
        quantization_config=qconfig,
    )


def load_quantized_model(model_path: str) -> pipeline:
    """Loads a quantized model and returns a pipeline object.

    Args:
        model_path: The path to the quantized model.

    Returns:
        A pipeline object for the quantized model.
    """
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return pipeline("text-classification", model=ort_model, tokenizer=tokenizer)


def push_to_hub(model_path: str, hf_repo_id: str, model_id: str) -> None:
    """Pushes the quantized model to the Hugging Face Hub.

    Args:
        model_path: Path or id of the model to push to the hub.
        hf_repo_id: Repo id of the repo to push to.
        model_id: Model id of the model to push to.
    """
    # Assumes that model_quantized.onnx exists in model_path.
    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        model_path,
        file_name="model_quantized.onnx",
    )
    onnx_model.push_to_hub(
        hf_repo_id,
        model_id,
        use_auth_token=True,
        private=True,
    )


def parse_args() -> argparse.Namespace:
    """Parses command line arguments.

    Returns:
        Returns the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Quantizes a model and pushes it to the Hugging Face Hub."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to the model to quantize.",
        default=os.path.join(
            "Factiverse", "factiverse_stance_detection_multilingual"
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="The path to save the quantized model.",
        default=os.path.join(
            "onnx", "factiverse_stance_detection_multilingual_quantized_dynamic"
        ),
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to evaluate the quantized model.",
        default=False,
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the quantized model to the huggingface hub.",
        default=False,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="The model ID on the Hugging Face Hub.",
        default="Factiverse/factiverse_stance_detection_multilingual_quantized",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="The model ID on the Hugging Face Hub.",
        default="factiverse_stance_detection_multilingual_quantized",
    )
    parser.add_argument(
        "--datasets_config",
        type=str,
        help="The dataset config file to evaluate the quantized model.",
        default=os.path.join(
            "src/stance_detection/bert/simpletransformers/experiments",
            "xlmroberta_chatgpt_selected_multilingual",
            "datasets_config.json",
        ),
    )
    return parser.parse_args()


def evaluate(
    classifier_pipeline: pipeline, dataset: pd.DataFrame
) -> Dict[str, Any]:
    """Evaluates the quantized model on the given dataset.

    Args:
        classifier_pipeline: Pipline object for the quantized model.
        dataset: Dataset to evaluate the quantized model on.

    Returns:
        Classification report from the evaluation.
    """
    data = dataset.to_dict("records")
    statrt_time = time.time()
    predictions = []
    ground_truth = []
    for _, row in enumerate(data):
        model_predictions = classifier_pipeline(
            row["text_a"] + "|" + row["text_b"]
        )
        if not math.isnan(row["labels"]):
            if model_predictions[0]["label"] == "SUPPORTS":
                predictions.append(1)
            else:
                predictions.append(0)
            ground_truth.append(row["labels"])
    print("total time taken by unquantized model", time.time() - statrt_time)
    report_dict = classification_report(
        ground_truth, predictions, output_dict=True
    )

    return report_dict


# def get_dataset(datasets_config_file: str) -> pd.DataFrame:
#     """Get the dataset for the given config.

#     Args:
#         datasets_config: File name of the dataset config.

#     Returns:
#         Dataframe containing the dataset.
#     """
#     datasets_config_path = Path(datasets_config_file)

#     datasets_config = DatasetsConfig.load(datasets_config_path)
#     loaded_datasets = [
#         config.create_dataset() for config in datasets_config.test_datasets
#     ]
#     dataset = AggregatedDataset(loaded_datasets).as_dataframe(
#         simple_transformers=True,
#         labels=StanceDataset.VALID_LABELS,
#         label_mapping={"REFUTES": 0, "SUPPORTS": 1},
#     )
#     if datasets_config.agument_test_data:
#         for augmentation_config in datasets_config.augmentations:
#             augmentation: Augmentor = Augmentor.from_name(
#                 augmentation_config.name, **augmentation_config.params
#             )
#             dataset = augmentation(dataset)
#     return dataset


def main():
    """The main function."""
    pass
    # args = parse_args()

    # if args.push_to_hub:
    #     print("Pushing to hub")
    #     push_to_hub(args.output_path, args.hf_repo_id, args.model_id)

    # if args.evaluate:
    # dataset = get_dataset(args.datasets_config)
    # quantize_model(args.model_path, args.output_path)
    # onnx_classifier = load_quantized_model(args.model_id)
    # report_dict = evaluate(onnx_classifier, dataset)
    # print(report_dict)


if __name__ == "__main__":
    main()
