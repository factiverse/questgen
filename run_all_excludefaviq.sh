#!/bin/bash
ls src/model_configs/config-bart-base-exclude_faviq.yaml
ls src/model_configs/config-flan-t5-base-exclude_faviq.yaml
ls src/model_configs/config-t5-base-exclude_faviq.yaml
ls src/model_configs/config-t5-3b-exclude_faviq.yaml
# python3 src/Generate-Question-train.py --config src/model_configs/config-bart-base-exclude_faviq.yaml
# python3 src/Generate-Question-train.py --config src/model_configs/config-flan-t5-base-exclude_faviq.yaml
# python3 src/Generate-Question-train.py --config src/model_configs/config-t5-base-exclude_faviq.yaml
python3 src/Generate-Question-train.py --config src/model_configs/config-t5-3b-exclude_faviq.yaml