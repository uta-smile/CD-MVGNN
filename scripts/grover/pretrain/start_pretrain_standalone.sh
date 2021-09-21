#!/bin/bash
rm main.sh
rm seven.yml

cp ./scripts/grover/pretrain/main_pretrain_standalone.sh main.sh
cp ./scripts/grover/yml_src/seven_standalone.yml seven.yml

seven create -conf ./seven.yml -code . -name grover

rm main.sh
rm seven.yml
