#!/bin/sh
torchrun --nproc_per_node 1 src/$1
