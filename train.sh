#!/bin/bash


CUDA_VIS_DEV=$1
GPU_NUM=$2
MMDET_ROOT=$3
CONFIGS=$4


for CONFIG_PATH in "${CONFIGS}"/*.py
do
    CUDA_VISIBLE_DEVICES=${CUDA_VIS_DEV} "${MMDET_ROOT}"/tools/dist_train.sh "${CONFIG_PATH}" "${GPU_NUM}" --validate --seed 0
done
