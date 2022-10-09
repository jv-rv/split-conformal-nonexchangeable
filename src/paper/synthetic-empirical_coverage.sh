#!/bin/sh

export LC_NUMERIC="C"

MODELS="boosting"
N_CAL="15000"
N_TEST="15000"

alpha="0.1"
n_train="1000"
lags="10"
simulations="1000"

for model in $MODELS
do
  for n_cal in $N_CAL
  do
    for n_test in $N_TEST
    do

      # Two-state markov chain
      for pq in $(seq 0.3 0.01 0.5)
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
done
