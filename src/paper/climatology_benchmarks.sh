#!/bin/sh

export LC_NUMERIC="C"


printf "\nSplit CP\n"
printf "========\n"

printf "Started experiment on %s\n" "$(date -u +%s)"
python src/eval/climatology/split_cp.py
printf "Finished on %s\n" "$(date -u +%s)"


printf "\nSplit NexCP\n"
printf "==========\n"

for decay in $(seq 0.1 0.2 0.9; echo 0.99)
do
  printf "Started experiment for decay = %s on %s\n" "$decay" "$(date -u +%s)"
  python src/eval/climatology/split_nexcp.py --decay_time "$decay" --decay_space "$decay"
done

printf "Finished on %s\n" "$(date -u +%s)"


printf "\nDtACI\n"
printf "====\n"

for I in $(seq 1 2 9)
do
  printf "Started experiment for I = %s on %s\n" "$I" "$(date -u +%s)"
  python src/eval/climatology/dtaci.py --I "$I"
done

printf "Finished on %s\n" "$(date -u +%s)"
printf "\n"
