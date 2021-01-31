#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# 100 dimension
python run.py --do_train --dataset fb15k --cuda --do_valid --do_test --evaluate_train --model TransE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen --freeze_relation_emb --save_path freeze
python run.py --do_train --dataset fb15k --cuda --do_valid --do_test --evaluate_train --model DistMult -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen -r 0.000002 --freeze_relation_emb --save_path freeze
python run.py --do_train --dataset fb15k --cuda --do_valid --do_test --evaluate_train --model RotatE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen -de --freeze_relation_emb --save_path freeze
python run.py --do_train --dataset fb15k --cuda --do_valid --do_test --evaluate_train --model ComplEx -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen -de -dr -r 0.000002 --freeze_relation_emb --save_path freeze
python run.py --do_train --dataset fb15k --cuda --do_valid --do_test --evaluate_train --model PairRE -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen -dr --freeze_relation_emb --save_path freeze
python run.py --do_train --dataset fb15k --cuda --do_valid --do_test --evaluate_train --model TuckER -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen --freeze_relation_emb --save_path freeze
python run.py --do_train --dataset fb15k --cuda --do_valid --do_test --evaluate_train --model Groups -n 128 -b 512 -d 1000 -g 24 -a 1.0 -adv -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 --print_on_screen -de --freeze_relation_emb --save_path freeze
