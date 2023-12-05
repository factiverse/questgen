#!/bin/bash
echo src/model_configs/config-t5-base-all_claim_decomp.yaml
python -m src.Generate-Question-test --config src/model_configs/config-t5-base-all_claim_decomp.yaml
echo src/model_configs/config-t5-base-all_fact_checking_briefs.yaml
python -m src.Generate-Question-test --config src/model_configs/config-t5-base-all_fact_checking_briefs.yaml
echo src/model_configs/config-t5-base-all_gpt_generated.yaml
python -m src.Generate-Question-test --config src/model_configs/config-t5-base-all_gpt_generated.yaml
echo src/model_configs/config-t5-base-all_faviq_r_set.yaml
python -m src.Generate-Question-test --config src/model_configs/config-t5-base-all_faviq_r_set.yaml
echo src/model_configs/config-t5-base-all_faviq_a_set.yaml
python -m src.Generate-Question-test --config src/model_configs/config-t5-base-all_faviq_a_set.yaml