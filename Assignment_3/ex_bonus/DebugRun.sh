#!/bin/sh

make debug && cuda-memcheck ./exercise_bonus.out -s 1024 -v > memcheck.log