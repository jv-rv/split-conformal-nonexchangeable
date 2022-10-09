#!/bin/sh

export LC_NUMERIC="C"

MODELS="boosting random_forest neural_network"
N_CAL="500"

alpha="0.1"
n_train="1000"
lags="10"
simulations="10000"
n_test="1"

for model in $MODELS
do
  for n_cal in $N_CAL
  do

    for phi in $(seq 0 0.02 0.98; echo 0.99 0.999 0.9999 0.99999)
    do
      python src/eval/synthetic/coverage.py \
        --quantile_model "$model" \
        --alpha "$alpha" \
        --n_train "$n_train" \
        --n_cal "$n_cal" \
        --n_test "$n_test" \
        --lags "$lags" \
        --simulations "$simulations" \
        ar1 --phi "$phi"
    done

    for pq in $(echo 0.001; seq 0.01 0.01 0.5)
    do
      python src/eval/synthetic/coverage.py \
        --quantile_model "$model" \
        --alpha "$alpha" \
        --n_train "$n_train" \
        --n_cal "$n_cal" \
        --n_test "$n_test" \
        --lags "$lags" \
        --simulations "$simulations" \
        two_state_markov_chain --prob_p "$pq" --prob_q "$pq"
    done

  done
done
