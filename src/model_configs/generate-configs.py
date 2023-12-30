"""Uses hardcoded Seq2Seq models and datasets to generate config-files."""
import yaml  # type: ignore
import os


def get_all_models() -> list:
    """Returns a list of all finetuned models.

    Returns:
        list of paths to all finetuned models
    """
    paths = [os.path.join("models", path) for path in os.listdir("models")]

    return paths


print(get_all_models())

with open("src/model_configs/config.yaml", "r") as file:  # Base Model Config
    base_file = yaml.safe_load(file)

datasets = ["literal-implied-questions"]
# datasets.remove("sample")
model_zoo = get_all_models()

config_dir = "src/model_configs/literal-implied-test"

for dataset in datasets:
    for model in model_zoo:
        new_file = base_file
        new_file["model_checkpoint"] = model
        new_file["data"] = os.path.join("data", dataset)
        name = "config-" + model.split("/")[-1] + "-" + dataset.split("/")[-1]
        with open(os.path.join(config_dir, name + ".yaml"), "w") as file:
            yaml.dump(new_file, file)
        print(
            "python3 src/Generate-Question-Test.py --config "
            + os.path.join(config_dir, name + ".yaml")
        )
