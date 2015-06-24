#!/usr/bin/env sh

python3 -m mnist_exp.mnist_exp
python3 -m mnist_exp.mnist_exp no_inner_layer
python3 -m mnist_exp.mnist_exp var_inner_layer
