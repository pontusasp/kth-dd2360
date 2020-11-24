#!/bin/sh

make debug && cuda-memcheck ./bin/exercise_2a.out 16 1000 10000 gpu > memcheck.log