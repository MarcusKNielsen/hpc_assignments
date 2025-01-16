#!/bin/bash

ITERATIONS="500"
TOLERANCE="0.01"
EXECUTABLE="poisson_j"

THREADS="1 2 4 8 12 16 20"
S="400"

for T in $THREADS
do
  OUTPUT=$( { time OMP_NUM_THREADS=$T ./$EXECUTABLE $S $ITERATIONS $TOLERANCE 0; } 2>&1 )

  METRICS=$(echo "$OUTPUT" | head -n 1)           # Extract the first line
  REAL_TIME=$(echo "$OUTPUT" | grep "real" | awk '{print $2}' | sed 's/,/./' | awk -F'm' '{printf "%.3f", $1 * 60 + $2}')
  USER_TIME=$(echo "$OUTPUT" | grep "user" | awk '{print $2}' | sed 's/,/./' | awk -F'm' '{printf "%.3f", $1 * 60 + $2}')

  echo "$METRICS, $T, $REAL_TIME, $USER_TIME"
done
