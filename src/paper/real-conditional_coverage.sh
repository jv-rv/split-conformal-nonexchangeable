#!/bin/sh

export LC_NUMERIC="C"

EVENTS="highvol lowvol uptrend downtrend"
N_CAL="500 1000 5000"
MODELS="boosting"
DATASETS="eurusd bcousd spxusd"

alpha="0.1"
n_train="1000"
n_test="1"
lags="10"

for event in $EVENTS
do
  for n_cal in $N_CAL
  do
    for model in $MODELS
    do
      for dataset in $DATASETS
      do

        python src/eval/real/conditional_coverage.py \
          --dataset "$dataset" \
          --quantile_model "$model" \
          --alpha "$alpha" \
          --n_train "$n_train" \
          --n_cal "$n_cal" \
          --n_test "$n_test" \
          --lags "$lags" \
          --event "$event"

      done
    done
  done
done
