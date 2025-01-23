#!/bin/bash

#!/bin/bash
#BSUB -J part7
#BSUB -o test_%J.out
#BSUB -q hpcintrogpu
##BSUB -B
##BSUB -N
#BSUB -n 8
#BSUB -R "rusage[mem=10GB]"
#BSUB -W 15
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"


# Usage: ./script.sh <iterations> <tolerance>

export OMP_PLACES=numa_domains
export OMP_PROC_BIND=spread
export OMP_DISPLAY_ENV=TRUE
export OMP_DISPLAY_AFFINITY=TRUE

export CUDA_VISIBLE_DEVICES=0,1

ITERATIONS="800"
TOLERANCE="0.01"
EXECUTABLE="poisson7"

OUTPUT_FILE="${EXECUTABLE}_I${ITERATIONS}_T${TOLERANCE}.dat"


SIZES="40 80 100 150 200 300 400 750"

lscpu


for S in $SIZES
do
OUTPUT=$( { time  ./$EXECUTABLE $S $ITERATIONS $TOLERANCE 0; } 2>&1 )
echo "$OUTPUT"

METRICS=$(echo "$OUTPUT" | tail -n 5 | head -n 1)           # Extract the first line
REAL_TIME=$(echo "$OUTPUT" | grep "real" | awk '{print $2}' | sed 's/,/./' | awk -F'm' '{printf "%.3f", $1 * 60 + $2}')
USER_TIME=$(echo "$OUTPUT" | grep "user" | awk '{print $2}' | sed 's/,/./' | awk -F'm' '{printf "%.3f", $1 * 60 + $2}')
#   SYS_TIME=$(echo "$OUTPUT" | grep "sys" | awk '{print $2}' | sed 's/,/./' | awk -F'm' '{printf "%.3f", $1 * 60 + $2}')

echo "$METRICS, $REAL_TIME, $USER_TIME" >> $OUTPUT_FILE
done

