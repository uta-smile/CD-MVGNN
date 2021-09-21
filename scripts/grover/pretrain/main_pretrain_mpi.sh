#!/bin/bash 
unset http_proxy
unset https_proxy
pip install -i https://mirrors.tencent.com/pypi/simple/ --trusted-host mirrors.tencent.com tqdm 

mkdir -p /opt/ml/env/out
mkdir -p /opt/ml/disk/out
mkdir -p /opt/ml/env/model_dir/grover
mkdir -p /opt/ml/disk/grover/
echo START
sh ./scripts/grover/pretrain/run_seven.sh

cp -r /opt/ml/env/model_dir/grover /opt/ml/model/grover
ls -l /opt/ml/disk/grover/
#cp -r /opt/ml/env/model_dir/grover/* /opt/ml/disk/grover/

rm -r /opt/ml/env/out
echo DONE
