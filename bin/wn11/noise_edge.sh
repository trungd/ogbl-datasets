#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 100 dimension
python run.py --do_train --dataset wn11 --cuda --model TransE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen --new_edge_type --new_edge_frac 0.01 --save_path noise1
python run.py --do_train --dataset wn11 --cuda --model TransE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen --new_edge_type --new_edge_frac 0.05 --save_path noise5
python run.py --do_train --dataset wn11 --cuda --model TransE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen --new_edge_type --new_edge_frac 0.1 --save_path noise10
python run.py --do_train --dataset wn11 --cuda --model TransE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen --new_edge_type --new_edge_frac 0.2 --save_path noise20
python run.py --do_train --dataset wn11 --cuda --model TransE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen --new_edge_type --new_edge_frac 0.5 --save_path noise50
python run.py --do_train --dataset wn11 --cuda --model TransE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen --new_edge_type --new_edge_frac 1. --save_path noise100

python run.py --do_train --dataset wn11 --cuda --model PairRE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen -dr --new_edge_type --new_edge_frac 0.01 --save_path noise1
python run.py --do_train --dataset wn11 --cuda --model PairRE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen -dr --new_edge_type --new_edge_frac 0.05 --save_path noise5
python run.py --do_train --dataset wn11 --cuda --model PairRE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen -dr --new_edge_type --new_edge_frac 0.1 --save_path noise10
python run.py --do_train --dataset wn11 --cuda --model PairRE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen -dr --new_edge_type --new_edge_frac 0.2 --save_path noise20
python run.py --do_train --dataset wn11 --cuda --model PairRE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen -dr --new_edge_type --new_edge_frac 0.5 --save_path noise50
python run.py --do_train --dataset wn11 --cuda --model PairRE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen -dr --new_edge_type --new_edge_frac 1. --save_path noise100