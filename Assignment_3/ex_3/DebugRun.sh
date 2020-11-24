#!/bin/sh

mkdir -p logs
make debug && cuda-memcheck ./bin/exercise_3.out 16 1000 100000 > logs/memcheck.log