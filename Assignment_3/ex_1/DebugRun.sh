#!/bin/sh

make debug && cuda-memcheck ./hw3_ex1.out images/rome.bmp > memcheck.log