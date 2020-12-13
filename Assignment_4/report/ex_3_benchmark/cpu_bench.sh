#!/bin/sh

# particles iterations blocksize benchtype

rm cpu.dat
touch cpu.dat

printf "|-----|\n"
printf "|"
for i in 10 100 1000 10000 100000
do
    ../../ex_3/bin/ex_3 $i 10000 256 2 >> cpu.dat
    printf "."
done
printf "|\n"
echo "CPU Done!"