#!/bin/bash


# 100 dimension
python run.py --do_train --dataset ogbl-wikikg2 --cuda --do_valid --do_test --evaluate_train \
  --model TransE -n 128 -b 512 -d 100 -g 30 -a 1.0 -adv \
  -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen

# 600 dimension
#python run.py --do_train --cuda --do_valid --do_test --evaluate_train \
#  --model TransE -n 128 -b 512 -d 600 -g 30 -a 1.0 -adv \
#  -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32