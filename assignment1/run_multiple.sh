#!/bin/bash
# 02614 - High-Performance Computing, January 2022
# 
# batch script to run matmult on a decidated server in the hpcintro
# queue
#
# Author: Bernd Dammann <bd@cc.dtu.dk>
# Modified: Asta Rustad 08.01.25
#
#BSUB -J mm_batch
# -- Output File --
#BSUB -o job_out/Output_%J.out
# -- Error File --
#BSUB -e job_out/Output_%J.err
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 60
#BSUB -R "select[model==XeonE5_2650v4]"
# uncomment the following line, if you want to assure that your job has
# a whole CPU for itself (shared L3 cache)
#BSUB -R "span[hosts=1] affinity[socket(1)]"

# define the driver name to use
# valid values: matmult_c.studio, matmult_f.studio, matmult_c.gcc or
# matmult_f.gcc
#
EXECUTABLE=matmult_f.gcc

# define the mkn values in the MKN variable
SIZES="10 20 43 50 100 120 150 200 400 800 1300 1500 2000"



# define the permutation type in PERM
#
#PERM="mkn"
PERMS="nat mkn mnk nkm nmk kmn knm lib"

# uncomment and set a reasonable BLKSIZE for the blk version
#
BLKSIZE=1

# enable(1)/disable(0) result checking
export MATMULT_COMPARE=0



lscpu
lscpu -C
for PERM in $PERMS
do
for S in $SIZES
do
    ./$EXECUTABLE $PERM $S $S $S $BLKSIZE >> results.dat
done
done