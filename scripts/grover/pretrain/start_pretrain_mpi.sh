#!/bin/bash
rm main.sh
rm seven.yml

cp ./scripts/grover/pretrain/main_pretrain_mpi.sh main.sh
cp ./scripts/grover/yml_src/seven.yml seven.yml

seven create -conf ./seven.yml -code . -name grover -cluster bee

rm main.sh
rm seven.yml
