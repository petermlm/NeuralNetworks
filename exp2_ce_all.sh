#!/usr/bin/env sh

python3 -m exp2_ce.exp2_ce its_var
python3 -m exp2_ce.exp2_ce its_var 2
python3 -m exp2_ce.exp2_ce its_var 3
python3 -m exp2_ce.exp2_ce its_var 4

python3 -m exp2_ce.exp2_ce step_var
python3 -m exp2_ce.exp2_ce step_var 2
python3 -m exp2_ce.exp2_ce step_var 3
python3 -m exp2_ce.exp2_ce step_var 4

python3 -m exp2_ce.exp2_ce train_var
python3 -m exp2_ce.exp2_ce train_var 2
python3 -m exp2_ce.exp2_ce train_var 3
python3 -m exp2_ce.exp2_ce train_var 4
