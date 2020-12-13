#!/bin/sh

# particles iterations blocksize benchtype

rm gpu_16.dat
touch gpu_16.dat

rm gpu_32.dat
touch gpu_32.dat

rm gpu_64.dat
touch gpu_64.dat

rm gpu_128.dat
touch gpu_128.dat

rm gpu_256.dat
touch gpu_256.dat

printf "|-------------------------|\n"
printf "|"
for i in 10 100 1000 10000 100000
do
    for b in 16 32 64 128 256
    do
        ../../ex_3/bin/ex_3 $i 10000 $b 3 >> gpu_$b.dat
        printf "."
    done
done
printf "|\n"
echo "GPU Done!"