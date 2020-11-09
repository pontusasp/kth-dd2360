#!/bin/sh

#block size, iterations, particles

rm ex_3_cpu_output.dat
touch ex_3_cpu_output.dat

printf "|--------------------|\n"
printf "|"
for i in 10000 100000 1000000 10000000
do
    for x in 1 2 3 4 5
    do
        ../../ex_3/bin/out $x 1000 $i nogpu cpu >> ex_3_cpu_output.dat
        printf "."
    done
done
printf "|\n"
echo "Done!"