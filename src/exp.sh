#!/usr/bin/env sh

# you can start multiple experments with this script
# train
{
# run several exp at one time
BASE_ARGS="-a cse_resnet50 -b 120 --num-instance 8 --num-hashing 512 --cse-end 4 \
--lr-sch cos --warmup 5  --fixbase-epochs 2 --base-lr-mult 0.1 --lr 1e-4 \
--epoch-lenth 200 --eval-period 20 --epochs 40 \
--loss ce --margin 0.2 --tri-lambda 1.0 --savedir ../logs1 "

EXP_ARGS=( \
"-a cse_resnet50 --loss ce --eval-period 10 --remarks cos40 " \
"-a cse_resnet50 --loss cross --eval-period 10 --remarks cos40 " \
"-a cse_resnet50 --loss within --eval-period 10 --remarks cos40 " \
"-a cse_resnet50 --loss hybrid --eval-period 10 --remarks cos40 " \
"-a cse_resnet50 --loss mathm --eval-period 10 --remarks cos40 " \
"-a cse_resnet50 --loss all --eval-period 10 --remarks cos40 " \
)
REPEAT_ARGS=(
"--savedir ../logs/2/ "
)

for DATASET in tuberlin sketchy # sketchy2 #
do
 for REPEAT_ARG in "${REPEAT_ARGS[@]}"
 do
    for EXP_ARG in "${EXP_ARGS[@]}"
    do
      CUDA_VISIBLE_DEVICES=0 \
      python runner_train.py "$BASE_ARGS -d $DATASET $EXP_ARG $REPEAT_ARG"
    done
 done
done
}

# test
python test.py -d tuberlin --precision --recompute \
--resume-dir 'path to your exp folder'
