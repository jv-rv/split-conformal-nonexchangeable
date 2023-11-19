#!/bin/sh

# Verify that raw data files contain information for every month in the year.
# This script should be run after `src/data/download_forex.py`.

expected="010203040506070809101112"

for file in data/raw/*csv
do
  result=$(cut -c5-6 "$file" | uniq | sort | tr -d '\n')
  if ! [ "$result" = "$expected" ]
  then
    printf "%s is incomplete\n" "$file"
  fi
done
