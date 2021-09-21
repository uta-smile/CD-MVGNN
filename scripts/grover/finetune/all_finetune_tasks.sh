#!/bin/bash
rm main.sh
rm seven.yml
cp scripts/grover/finetune/main_finetune.sh main.sh
cp scripts/grover/yml_src/seven_fintune.yml seven.yml
#automl run --conf scripts/grover/yml_src/finetune/bace_SB.yml
#automl run --conf scripts/grover/yml_src/finetune/bbbp_SB.yml
#automl run --conf scripts/grover/yml_src/finetune/tox21_SB.yml
#automl run --conf scripts/grover/yml_src/finetune/toxcast_SB.yml
#automl run --conf scripts/grover/yml_src/finetune/sider_SB.yml
#automl run --conf scripts/grover/yml_src/finetune/clintox_SB.yml
#automl run --conf scripts/grover/yml_src/finetune/freesolv_RD.yml
#automl run --conf scripts/grover/yml_src/finetune/lipo_RD.yml
#automl run --conf scripts/grover/yml_src/finetune/delaney_RD.yml
automl run --conf scripts/grover/yml_src/finetune/qm7_RD.yml
automl run --conf scripts/grover/yml_src/finetune/qm8_RD.yml
rm main.sh
rm seven.yml
