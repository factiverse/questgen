#!/bin/bash
# python3 src/Generate-Question-Finetune.py --config src/model_configs/config-bart-base-fact_checking_briefs.yaml
# python3 src/Generate-Question-Finetune.py --config src/model_configs/config-bart-base-claim_decomp.yaml
#python3 src/Generate-Question-Finetune.py --config src/model_configs/config-bart-base-gpt_generated.yaml
#python3 src/Generate-Question-Finetune.py --config src/model_configs/config-bart-base-all.yaml
#python3 src/Generate-Question-Finetune.py --config src/model_configs/config-bart-base-faviq_r_set.yaml
#python3 src/Generate-Question-Finetune.py --config src/model_configs/config-bart-base-faviq_a_set.yaml
#python3 src/Generate-Question-Finetune.py --config src/model_configs/config-t5-base-claim_decomp.yaml
#python3 src/Generate-Question-Finetune.py --config src/model_configs/config-t5-base-gpt_generated.yaml
#python3 src/Generate-Question-Finetune.py --config src/model_configs/config-t5-base-fact_checking_briefs.yaml
python3 src/Generate-Question-Finetune.py --config src/model_configs/config-t5-base-all.yaml
python3 src/Generate-Question-Finetune.py --config src/model_configs/config-t5-base-faviq_r_set.yaml
python3 src/Generate-Question-Finetune.py --config src/model_configs/config-t5-base-faviq_a_set.yaml
python3 src/Generate-Question-Finetune.py --config src/model_configs/config-flan-t5-base-claim_decomp.yaml
python3 src/Generate-Question-Finetune.py --config src/model_configs/config-flan-t5-base-gpt_generated.yaml
python3 src/Generate-Question-Finetune.py --config src/model_configs/config-flan-t5-base-all.yaml
python3 src/Generate-Question-Finetune.py --config src/model_configs/config-flan-t5-base-faviq_r_set.yaml
python3 src/Generate-Question-Finetune.py --config src/model_configs/config-flan-t5-base-fact_checking_briefs.yaml
python3 src/Generate-Question-Finetune.py --config src/model_configs/config-flan-t5-base-faviq_a_set.yaml
