#!/bin/bash

# For IBM BlueGene P.
# Originalli written by Kama The Bullet.

if [ ! -d results ]; then
    mkdir results
fi

for x in 1000 2000; do
    for n in 1 128 256 512; do
        file="res_"$x"_"$n
        if [ ! -f results/$file.txt ] || [ -f results/$file.err ]; then
            mpisubmit.bg -w 00:10:00 -n $n -m smp --stdout results/$file.txt --stderr results/$file.err main -- $x $x
            echo $file submitted
        fi
    done
done

