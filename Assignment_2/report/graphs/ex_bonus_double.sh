#!/bin/sh

# <block size> <num threads> <iterations per thread> bench

rm ex_bonus_double.dat
touch ex_bonus_double.dat

printf "|--------------------|\n"
printf "|"
for i in 1000 10000 100000 1000000
do
    for b in 16 32 64 128 256
    do
        ../../ex_bonus/bin/out $b 100000 $i bench >> ex_bonus_double.dat
        printf "."
    done
done
printf "|\n"
echo "Done!"