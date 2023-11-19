#!/bin/sh

export LC_NUMERIC="C"

for pq in $(seq 0.3 0.05 0.5)
do
  python src/eval/synthetic/theoretical_guarantee.py --delta 0.01 --pq "$pq" --n_jobs 5
done
