#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# 100 dimension
python run.py --do_train --dataset fb15k237 --cuda --do_valid --do_test --evaluate_train \
  --model TransE -n 128 -b 512 -d 100 -g 30 -a 1.0 -adv \
  -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen --resample

python run.py --do_train --dataset fb15k237 --cuda --do_valid --do_test --evaluate_train \
  --model DistMult -n 128 -b 512 -d 100 -g 30 -a 1.0 -adv \
  -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen --resample

python run.py --do_train --dataset fb15k237 --cuda --do_valid --do_test --evaluate_train \
  --model RotatE -n 128 -b 512 -d 100 -g 30 -a 1.0 -adv \
  -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen -de --resample

python run.py --do_train --dataset fb15k237 --cuda --do_valid --do_test --evaluate_train \
  --model ComplEx -n 128 -b 512 -d 100 -g 30 -a 1.0 -adv \
  -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen -de -dr --resample
