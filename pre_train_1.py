import argparse
import json
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR

import index
from model import TransformerModel
from tokenizer import CharTokenizer
from dataset import TextDataset, collate_fn
from data_loader import read_strings
from meter import Meter
from evaluation import em, gleu

import nsml
from nsml import DATASET_PATH

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def divide_dataset(pairs):
    
    train_list = index.train_index
    val_list = index.val_index

    train_data = []
    valid_data = []

    for i in train_list:
        train_data.append(pairs[i])
        
    for i in val_list:
        valid_data.append(pairs[i])

    return train_data, valid_data

def get_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data_dir", type=str, default=os.path.join(DATASET_PATH, 'train'))
    parser.add_argument("--num_val_data", type=int, default=1000)

    # model
    parser.add_argument("--vocab_size", type=int, default=1600)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=1024)

    # training
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=128)

    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--adam_betas", type=str, default="(0.9, 0.98)")
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--max_grad_norm", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--max_steps", type=int, default=10000000)
    parser.add_argument("--num_warmup_steps", type=int, default=4000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=20)

    # nsml
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default="0")

    args = parser.parse_args()
    return args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def calc_loss(model, batch):
    src, tgt, src_mask, tgt_mask, tgt_label = batch

    output = model(src=src, tgt=tgt, src_key_padding_mask=~src_mask, tgt_key_padding_mask=~tgt_mask)

    bsz = tgt.size(1)
    raw_loss = F.cross_entropy(output.view(-1, output.size(-1)), tgt_label.view(-1), reduction='none')
    raw_loss = raw_loss.view(-1, bsz)
    loss = (raw_loss * tgt_mask.float()).sum(0).mean()
    items = [loss.data.item(), bsz, tgt_mask.sum().item()]
    return loss, items

def evaluate(model, data_loader, args):
    model.eval()
    meter = Meter()
    with torch.no_grad():
        for batch in data_loader:
            batch = tuple(t.to(args.device) for t in batch)
            _, items = calc_loss(model, batch)
            meter.add(*items)
    return meter.average(), meter.print_str(False)

def correct(model, tokenizer, test_data, args):
    model.eval()
    prediction = []
    for i in range(0, len(test_data), args.eval_batch_size):
        batch = test_data[i:i + args.eval_batch_size]

        src_token_ids = [tokenizer(x) for x in batch]
        src_seq_length = [len(x) for x in src_token_ids]
        src_max_seq_length = max(src_seq_length)
        src_padded = []
        src_padding_mask = []
        for x in src_token_ids:
            x = x[:src_max_seq_length]
            src_pad_length = src_max_seq_length - len(x)
            src_padded.append(x + [1] * src_pad_length)
            src_padding_mask.append([1] * len(x) + [0] * src_pad_length)
        src_padded = torch.tensor(src_padded).t().contiguous().to(args.device)
        src_padding_mask = torch.tensor(src_padding_mask).bool().t().to(args.device)

        memory = model(src=src_padded, src_key_padding_mask=~src_padding_mask)

        tgt_token_ids = [[2] for _ in batch]
        end = [False for _ in batch]
        for l in range(src_max_seq_length + 20):
            tgt = torch.tensor(tgt_token_ids).t().contiguous().to(args.device)
            output = model(tgt=tgt, memory=memory, memory_key_padding_mask=~src_padding_mask)
            top1 = output[-1].argmax(-1).tolist()
            for i, tok in enumerate(top1):
                if tok == 3 or l >= src_seq_length[i] + 20:
                    end[i] = True
                tgt_token_ids[i].append(tok if not end[i] else 3)
            if all(end):
                break

        prediction.extend([''.join([tokenizer.i2c[tok] for tok in x if tok >= 4]) for x in tgt_token_ids])
    return prediction

def pre_train(model, tokenizer, train_data, args):
    model.train()

    train_dataset = TextDataset(train_data, tokenizer, pre_train_mode = True)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                  batch_size=args.train_batch_size, num_workers=args.num_workers,
                                  collate_fn=lambda x: collate_fn(x, tokenizer, args.max_seq_length))

    epochs = (args.max_steps - 1) // len(train_dataloader) + 1

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=eval(args.adam_betas), eps=args.eps,
                                 weight_decay=args.weight_decay)
    lr_lambda = lambda x: x / args.num_warmup_steps if x <= args.num_warmup_steps else (x / args.num_warmup_steps) ** -0.5
    scheduler = LambdaLR(optimizer, lr_lambda)

    step = 0
    best_loss = float("inf")
    meter = Meter()
    for epoch in range(1, epochs + 1):
        
        total_loss = 0
        for num_k, batch in enumerate(train_dataloader):
            step += 1
            batch = tuple(t.to(args.device) for t in batch)

            optimizer.zero_grad()
            loss, items = calc_loss(model, batch)
            meter.add(*items)

            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()
            scheduler.step()
            
            total_loss += loss.item()

            if num_k % 500 == 0:
                print(num_k, "/", len(train_dataloader))
                print("loss = ", loss.item())

        train_loss = total_loss / len(train_dataloader)

        nsml.report(step=step, train__loss=train_loss)
        print("epoch : " , epoch, "loss = ", train_loss)

        nsml.save(str(epoch))
            
def bind_nsml(model, tokenizer=None, args=None):
    def save(path, **kwargs):
        torch.save(model.state_dict(), open(os.path.join(path, 'model.pt'), 'wb'))
        if tokenizer is not None:
            tokenizer.save(os.path.join(path, 'vocab.txt'))

    def load(path, **kwargs):
        model.load_state_dict(torch.load(open(os.path.join(path, 'model.pt'), 'rb'),
                                         map_location=lambda storage, loc: storage))
        if tokenizer is not None:
            tokenizer.load(os.path.join(path, 'vocab.txt'))

    def infer(test_data, **kwargs):
        '''
        :param test_data: list of noisy sentences
        :return: list of corrected sentences
        '''
        return correct(model, tokenizer, test_data, args)

    import nsml
    nsml.bind(save, load, infer)

def main():
    args = get_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    #모델 설정
    model = TransformerModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
    ).to(args.device)
   
    tokenizer = CharTokenizer([])
    bind_nsml(model, tokenizer, args)
    
    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == "train":
        clean_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_corpus"))
        
        new_clean = []

        for i in clean_sents:
            if len(i) < 3:
                pass
            else:
                new_clean.append(i)

        pairs = [{"noisy": noisy, "clean": clean} for noisy, clean in zip(new_clean, new_clean)]

        train_data = pairs

        logger.info(f"# of train data: {len(train_data)}")

        train_sents = [x['noisy'] for x in train_data]
        tokenizer = CharTokenizer.from_strings(train_sents, args.vocab_size)
        
        k = dict(tokenizer.show_vocab())

        bind_nsml(model, tokenizer, args)
        #nsml.load(checkpoint='2', session='KR95444/airush2021-2-4/181')
        #nsml.load(checkpoint='best', session='KR95444/airush2021-2-4/182')
        
    if args.n_gpu > 1:
        print("멀티 gpu", torch.cuda.device_count())
        model = torch.nn.DataParallel(model, dim=1)

    if args.mode == "train":
        pre_train(model, tokenizer, train_data, args)


if __name__ == "__main__":
    main()
