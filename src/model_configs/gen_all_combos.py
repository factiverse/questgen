"""Uses hardcoded Seq2Seq models and datasets to generate config-files."""
import subprocess
import yaml  # type: ignore
import os

with open("src/model_configs/config.yaml", "r") as file:
    base_file = yaml.safe_load(file)

# datasets = os.listdir("data")
# datasets.remove("sample")

datasets = ["AVeriTeC"]
model_zoo = ["facebook/bart-base", "t5-base", "google/flan-t5-base"]

config_dir = "src/model_configs"
cmds = []
for dataset in datasets:
    for model in model_zoo:
        new_file = base_file
        new_file["model_checkpoint"] = model
        new_file["data"] = os.path.join("data", dataset)
        name = "config-" + model.split("/")[-1] + "-" + dataset
        with open(os.path.join(config_dir, name + ".yaml"), "w") as file:
            yaml.dump(new_file, file)
        print(
            "python3 src/Generate-Question-Finetune.py \
            --config src/model_configs/"
            + name
            + ".yaml"
        )
        cmds.append(
            "python3 src/Generate-Question-Finetune.py \
            --config src/model_configs/"
            + name
            + ".yaml"
        )

for cmd in cmds:
    subprocess.run(cmd, shell=True)
