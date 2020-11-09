#!/bin/sh

#block size, iterations, particles

rm ex_3_gpu_output.dat
touch ex_3_gpu_output.dat

printf "|----------------------------------------------------------------------------------------------------|\n"
printf "|"
for i in 10000 100000 1000000 10000000
do
    for b in 16 32 64 128 256
    do
        for x in 1 2 3 4 5
        do
            ../../ex_3/bin/out $b 1000 $i gpu >> ex_3_gpu_output.dat
            printf "."
        done
    done
done
printf "|\n"
echo "Done!"