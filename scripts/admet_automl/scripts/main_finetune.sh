#!/bin/bash
unset http_proxy
unset https_proxy
pip install -i https://mirrors.tencent.com/pypi/simple/ --trusted-host mirrors.tencent.com tqdm 
mkdir -p /opt/ml/env/out
mkdir -p /opt/ml/disk/out

echo START
CUDA_VISIBLE_DEVICES=0 sh ./scripts/admet_automl/scripts/admet_automl.sh
mv /opt/ml/env/model_dir/ /opt/ml/model/
rm -r /opt/ml/env/outOAE
echo END
