#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

from ogb.linkproppred import LinkPropPredDataset, Evaluator
from collections import defaultdict
from tqdm import tqdm
import time
from tensorboardX import SummaryWriter
import pdb


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--dataset', type=str, default='ogbl-biokg', help='dataset name, one of [ogbl-biokg, ogbl-wikikg2]')
    parser.add_argument('--dataset_root', type=str, default='/scratch/trungvd/ogb', help='folder to download datasets')
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('--rel_init_scale', default=1.0, type=float,
                        help='scale initialization of the relation vectors')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=1000, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--print_on_screen', action='store_true', help='log on screen or not')
    parser.add_argument('--ntriples_eval_train', type=int, default=200000, help='number of training triples to evaluate eventually')
    parser.add_argument('--neg_size_eval_train', type=int, default=500, help='number of negative samples when evaluating training triples')

    # Extra
    parser.add_argument('--new_edge_type', action='store_true')
    parser.add_argument('--new_edge_frac', type=float)
    parser.add_argument('--resample', action='store_true', help='resample test/valid sets')
    parser.add_argument('--freeze_relation_emb', action='store_true', help='not train relation emb')
    parser.add_argument('--no_reltype', action='store_true', help='only one relation type allowed')



    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.dataset = argparse_dict['dataset']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics, writer):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        writer.add_scalar("_".join([mode, metric]), metrics[metric], step)
        
        
def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and (not args.evaluate_train):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)

    args.save_path = 'log/%s/%s/%s-%s/%s' % (args.dataset, args.model, args.hidden_dim, args.gamma, time.time()) \
        if args.save_path is None \
        else 'log/%s/%s/%s-%s/%s' % (args.dataset, args.model, args.hidden_dim, args.gamma, args.save_path)
    writer = SummaryWriter(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)

    if args.dataset == 'ogbl-biokg':
        from datasets.biokg import BioKG
        ds = BioKG(args)
    elif args.dataset == 'ogbl-wikikg2':
        from datasets.wikikg2 import WikiKG2
        ds = WikiKG2(args)
    elif args.dataset == 'fb15k':
        from datasets.fb15k import FB15K
        ds = FB15K(args)
    elif args.dataset == 'fb15k237':
        from datasets.fb15k237 import FB15K237
        ds = FB15K237(args)
    elif args.dataset == 'fb13':
        from datasets.fb13 import FB13
        ds = FB13(args)
    elif args.dataset == 'wn11':
        from datasets.wn11 import WN11
        ds = WN11(args)
    elif args.dataset == 'wn18':
        from datasets.wn18 import WN18
        ds = WN18(args)

    evaluator = Evaluator(name='ogbl-wikikg2')
    
    logging.info('Model: %s' % args.model)
    logging.info('Dataset: %s' % args.dataset)
    logging.info('#entity: %d' % ds.nentity)
    logging.info('#relation: %d' % ds.nrelation)
    
    # train_triples = split_dict['train']
    logging.info('#train: %d' % len(ds.train_triples['head']))
    # valid_triples = split_dict['valid']
    logging.info('#valid: %d' % len(ds.valid_triples['head']))
    # test_triples = split_dict['test']
    logging.info('#test: %d' % len(ds.test_triples['head']))
    
    kge_model = KGEModel(
        args,
        model_name=args.model,
        nentity=ds.nentity,
        nrelation=ds.nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        evaluator=evaluator
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
    
    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(args, ds, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(args, ds, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        # logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        # checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        # entity_dict = checkpoint['entity_dict']
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        
        #Training Loop
        for step in range(init_step, args.max_steps):
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0 and step > 0: # ~ 41 seconds/saving
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps,
                }
                save_model(kge_model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Train', step, metrics, writer)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0 and step > 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, ds, ds.valid_triples, args)
                log_metrics('Valid', step, metrics, writer)
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, ds, ds.valid_triples, args)
        log_metrics('Valid', step, metrics, writer)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, ds, ds.test_triples, args)
        log_metrics('Test', step, metrics, writer)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        small_train_triples = {}
        indices = np.random.choice(len(ds.train_triples['head']), args.ntriples_eval_train, replace=False)
        for i in ds.train_triples:
            if 'type' in i:
                small_train_triples[i] = [ds.train_triples[i][x] for x in indices]
            else:
                small_train_triples[i] = ds.train_triples[i][indices]
        metrics = kge_model.test_step(kge_model, ds, small_train_triples, args, random_sampling=True)
        log_metrics('Train', step, metrics, writer)
        
if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main(parse_args())
