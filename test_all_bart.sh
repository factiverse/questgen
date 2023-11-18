#!/bin/bash
CUDA_VISIBLE_DEVICES=-1 python -m src.Generate-Question-test --config src/model_configs/config-bart-base-all_claim_decomp.yaml
CUDA_VISIBLE_DEVICES=-1 python -m src.Generate-Question-test --config src/model_configs/config-bart-base-all_fact_checking_briefs.yaml
CUDA_VISIBLE_DEVICES=-1 python -m src.Generate-Question-test --config src/model_configs/config-bart-base-all_gpt_generated.yaml
CUDA_VISIBLE_DEVICES=-1 python -m src.Generate-Question-test --config src/model_configs/config-bart-base-all_faviq_r_set.yaml
CUDA_VISIBLE_DEVICES=-1 python -m src.Generate-Question-test --config src/model_configs/config-bart-base-all_faviq_a_set.yaml