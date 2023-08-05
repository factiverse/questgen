"""Test Seq2Seq Language Model."""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
import yaml  # type: ignore
import argparse
import os
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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


def get_model_checkpoint(config: dict):
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

    model_checkpoint = get_model_checkpoint(config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, use_auth_token=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    logger.info(
        "type a claim to see how the model responds \
                 (q+enter to quit)"
    )
    while True:
        inp = input()
        if inp == "q":
            break
        input_ids = tokenizer(inp, return_tensors="pt")
        print(input_ids["input_ids"])
        outputs = model.generate(input_ids["input_ids"])
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
