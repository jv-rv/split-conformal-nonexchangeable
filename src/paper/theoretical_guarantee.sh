#!/bin/sh

PQ="0.50 0.45 0.40 0.35 0.30"

for pq in $PQ
do
  python src/eval/synthetic/theoretical_guarantee.py --delta 0.01 --pq "$pq" --n_jobs 5
done
