#!/bin/bash


CUDA_VIS_DEV=$1
GPU_NUM=$2
MMDET_ROOT=$3
CONFIGS=$4
WORK_DIR=$5


for CONFIG_PATH in "${CONFIGS}"/*.py
do
    CONFIG_FILE=$(basename "$CONFIG_PATH")
    CONFIG_FILE=${CONFIG_FILE%.*}

    for EP in 30 29 28
    do
        CHECKPOINT_FILE=${WORK_DIR}/${CONFIG_FILE}/epoch_${EP}.pth
        RESULT_FILE=${WORK_DIR}/${CONFIG_FILE}/test_${EP}.pkl

        CUDA_VISIBLE_DEVICES=${CUDA_VIS_DEV} "${MMDET_ROOT}"/tools/dist_test.sh "${CONFIG_PATH}" "${CHECKPOINT_FILE}" "${GPU_NUM}" --out "${RESULT_FILE}"
    done
done
