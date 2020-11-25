#!/bin/sh

mkdir -p logs

# 64, 128, ... , to 4096
echo "###" > logs/bench.txt
bin/exercise_bonus.out -s 64 -v >> logs/bench.txt
echo "###" >> logs/bench.txt
bin/exercise_bonus.out -s 128 -v >> logs/bench.txt
echo "###" >> logs/bench.txt
bin/exercise_bonus.out -s 256 -v >> logs/bench.txt
echo "###" >> logs/bench.txt
bin/exercise_bonus.out -s 512 -v >> logs/bench.txt
echo "###" >> logs/bench.txt
bin/exercise_bonus.out -s 1024 -v >> logs/bench.txt
echo "###" >> logs/bench.txt
bin/exercise_bonus.out -s 2048 -v >> logs/bench.txt
echo "###" >> logs/bench.txt
bin/exercise_bonus.out -s 4096 -v >> logs/bench.txt