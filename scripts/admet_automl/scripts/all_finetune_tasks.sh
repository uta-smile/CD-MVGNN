#!/bin/bash
rm main.sh
rm seven.yml
cp scripts/admet_automl/scripts/main_finetune.sh main.sh
cp scripts/admet_automl/seven_fintune.yml seven.yml
automl run --conf scripts/admet_automl/tmp/bbbp_SB.yml
rm main.sh
rm seven.yml
