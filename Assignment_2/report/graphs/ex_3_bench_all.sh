#!/bin/sh

printf "Running benchmark on CPU...\n"

./ex_3_bench_cpu.sh

printf "Done.\n\n"

printf "Running benchmark on GPU...\n"

./ex_3_bench_gpu.sh

printf "Done.\n\n"
printf "All Done!\n"

/sbin/shutdown -h 1