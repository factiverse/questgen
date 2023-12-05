#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2 python3 src/Generate-Question-Finetune.py --config src/model_configs/config-bloom-560m-claim_decomp.yaml
CUDA_VISIBLE_DEVICES=1,2 python3 src/Generate-Question-Finetune.py --config src/model_configs/config-bloom-560m-fact_checking_briefs.yaml
CUDA_VISIBLE_DEVICES=1,2 python3 src/Generate-Question-Finetune.py --config src/model_configs/config-bloom-560m-claim_decomp-fact_checking_briefs.yaml
CUDA_VISIBLE_DEVICES=1,2 python3 src/Generate-Question-Finetune.py --config src/model_configs/config-bloom-560m-gpt_generated.yaml
CUDA_VISIBLE_DEVICES=1,2 python3 src/Generate-Question-Finetune.py --config src/model_configs/config-bloom-560m-all.yaml
CUDA_VISIBLE_DEVICES=1,2 python3 src/Generate-Question-Finetune.py --config src/model_configs/config-bloom-560m-faviq_r_set.yaml
CUDA_VISIBLE_DEVICES=1,2 python3 src/Generate-Question-Finetune.py --config src/model_configs/config-bloom-560m-faviq_a_set.yaml