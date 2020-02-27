#!/bin/bash


MMDET_ROOT=/home/brodt/soft/mmdetection
CUDA_DECS=0,1,2,3
GPU_NUM=4
CONFIGS=./configs
WORK_DIR=./work_dirs
DATA_PATH=./data


python -m src.prepare_data train --data-path ${DATA_PATH} --seed 314158 --n-splits 5
python -m src.prepare_data test --data-path ${DATA_PATH}

sh train.sh ${CUDA_DECS} ${GPU_NUM} ${MMDET_ROOT} ${CONFIGS}

sh predict.sh ${CUDA_DECS} ${GPU_NUM} ${MMDET_ROOT} ${CONFIGS} ${WORK_DIR}

python -m src.ensemble --data-path ${DATA_PATH} --preds-path ${WORK_DIR} --save-path ./easy_gold.json
