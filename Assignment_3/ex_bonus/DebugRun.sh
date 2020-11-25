#!/bin/sh

make debug && cuda-memcheck bin/exercise_bonus.out -s 1024 -v > memcheck.log