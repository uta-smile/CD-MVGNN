#!/bin/bash

nvidia-docker run -it \
    -v `pwd`:/env -v /data2/weiyangxie/mit:/mit --shm-size 16G \
    docker.oa.com/g_tfplus/docker.oa.com/g_tfplus/horovod:py3.6-tf1.12-pytorch1.1-rdkit2019-openbabel-torch-geometric bash
    #docker.oa.com/g_tfplus/ai-drug:tf1.12-deepchem2.1.1-rdkit2018.03-pytorch1.0.1-keras2.2.4-java8-hadoop2.7.5 bash
