#!/bin/bash
source ~/.virtualenvs/nnv/bin/activate
for i in `seq 1 3`;
do
    nice -10 python inference-fast.py data rcvpr2020 1 --seed 1 > out$i.txt
done
cowsay "NNV - I am done!"
