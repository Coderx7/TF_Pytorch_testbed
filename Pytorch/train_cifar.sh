#!/bin/bash
#In the name of God the most compassionate the most merciful 

ITERATION=1
LR=0.1
SAVE_DIR=./snapshots/simpnet
DATASET_DIR=./data/cifar.python
WORKER=2


ARCH=simpnet
DATASET=cifar10
EPOCHS=450
BATCH_SIZE=100


for (( i=1; i<=$ITERATION; i++ ))
do
python main.py  $DATASET_DIR --dataset $DATASET --arch $ARCH --save_path $SAVE_DIR --epochs $EPOCHS --learning_rate $LR --batch_size $BATCH_SIZE --workers $WORKER 
done
