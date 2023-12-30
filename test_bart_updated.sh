#!/bin/bash
# python -m src.Generate-Question-Test --config src/model_configs/config-bart-base-all-eval-claim_decomp.yaml
# python -m src.Generate-Question-Test --config src/model_configs/config-bart-base-eval-claim_decomp.yaml
python -m src.Generate-Question-Test --config src/model_configs/config-bart-base-all-eval-fact_checking_briefs.yaml
python -m src.Generate-Question-Test --config src/model_configs/config-bart-base-eval-fact_checking_briefs.yaml