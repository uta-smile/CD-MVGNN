#!/bin/bash
alias seven='~/seven'

# create a minimal code directory
DIR_FULL=`pwd`
DIR_MINIMAL=${DIR_FULL}-minimal
rm -rf ${DIR_MINIMAL} && mkdir ${DIR_MINIMAL}
cp -rv ${DIR_FULL}/chemprop ${DIR_MINIMAL}
cp -rv ${DIR_FULL}/descriptastorus ${DIR_MINIMAL}
cp -v ${DIR_FULL}/scripts ${DIR_MINIMAL}
cp -v ${DIR_FULL}/main.sh ${DIR_MINIMAL}
cp -v ${DIR_FULL}/*.yml ${DIR_MINIMAL}
cp -v ${DIR_FULL}/*.py ${DIR_MINIMAL}
cd ${DIR_MINIMAL}

## select the seven cluster
#CLUSTER=bee2       # dev or bee2
#CLUSTER=dev       # dev or bee2
CLUSTER=low       # dev or bee2

## Set the seven job name
GPU_NUM=$1        # how many gpus for the training
BS=$2
echo "GPU_NUM:" $GPU_NUM "Batch Size:" ${BS}

#CFG_FILE="seven_mpi.yml"
CFG_FILE="seven_standalone.yml"

## Start the job
## Train ##
seven create -conf ${CFG_FILE} -code `pwd` -name dmpnn-${GPU_NUM}gpu-bs${BS} -cluster ${CLUSTER}
