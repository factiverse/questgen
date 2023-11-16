#!/bin/bash
#python3 src/Generate-Question-train.py --config src/model_configs/config-bart-base-all.yaml
python3 src/Generate-Question-train.py --config src/model_configs/config-flan-t5-all.yaml
python3 src/Generate-Question-train.py --config src/model_configs/config-t5-base-all.yaml