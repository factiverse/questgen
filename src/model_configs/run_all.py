import os
import yaml

all_datasets = os.listdir("./data")
all_models = os.listdir("./models")
base_yaml = []
with open("src/model_configs/config.yaml", "r") as file:
    base_yaml = yaml.safe_load(file)

# for d in all_datasets:
#     print(d)

# for m in all_models:
#     print(m)

# print(base_yaml)

for d in all_datasets:
    if str(d) != "sample":
        for m in all_models:
            if not str(d) in m:
                print("data/" + str(d), "models/" + str(m))
                base_yaml["data"] = "data/" + str(d)
                base_yaml["model_checkpoint"] = "models/" + str(m)
                with open("src/model_configs/config.yaml", "w") as file:
                    yaml.safe_dump(base_yaml, file)
                os.system(
                    "python3 src/Generate-Question-test.py --config src/model_configs/config.yaml"
                )
