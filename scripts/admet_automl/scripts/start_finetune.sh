#!/bin/bash
rm main.sh
rm seven.yml
cp scripts/grover/finetune/main_finetune.sh main.sh
cp scripts/grover/yml_src/seven_fintune.yml seven.yml

automl run --conf ./scripts/grover/yml_src/finetune_template.yml

rm main.sh
rm seven.yml
