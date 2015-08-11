#!/usr/bin/env sh

python3 mnist.py > res_default
python3 mnist.py no_inner_layer > res_no_inner_layer
python3 mnist.py var_inner_layer > res_var_inner_layer
python3 mnist.py 500 > res_500
