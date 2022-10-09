#!/bin/sh

DATASETS="eurusd bcousd spxusd"
YEARS="2020 2021"

for dataset in $DATASETS
do
  for year in $YEARS
  do
    python src/data/process.py --assets "$dataset" --year "$year"
  done
done
