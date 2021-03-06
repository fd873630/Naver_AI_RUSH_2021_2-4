import argparse
import json
import logging
import math
import os
import random
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR

import index
import ssl_data_1, ssl_data_2, ssl_data_3, ssl_data_4
#import ssl_data_1, ssl_data_2, ssl_data_3, ssl_data_4, ssl_data_5
from model import TransformerModel
from tokenizer import CharTokenizer
from dataset import TextDataset, collate_fn
from data_loader import read_strings
from meter import Meter
from evaluation import em, gleu

import nsml
from nsml import DATASET_PATH

from sklearn.model_selection import KFold

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_warining(pairs):
    #warning_list = index.warning_index_6

    warning_list = index.final_index
    warning_list.reverse()

    for i in warning_list:
        del pairs[i]

    
    total = []
    '''
    for i in pairs:
        a = i['clean']
        b = i['noisy']

        if len(b) < len(a):
            total.append(i)
    '''
    
    random.shuffle(pairs)

    return pairs

def ssl_dataset():
           
    data_1 = ssl_data_1.first_data
    data_2 = ssl_data_2.second_data
    data_3 = ssl_data_3.third_data
    data_4 = ssl_data_4.fourth_data
    #data_5 = ssl_data_5.fifth_data
    
    total_data = data_1 + data_2 + data_3 + data_4 # + data_5    
    #total_data = data_1
    return total_data

def divide_dataset(pairs):
           
    train_list = index.train_index
    val_list = index.val_index
        
    #add_train_list = index.new_train   
    #train_list = add_train_list + train_list
    #val_list = index.new_val

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
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=1024)

    # training
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--adam_betas", type=str, default="(0.9, 0.98)")
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--max_grad_norm", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
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

def beam_search(model, tokenizer, test_data, beam_width, args):
    #beam search
    #beam_width = 3
    model.eval()
    prediction = []

    for i in range(0, len(test_data)):
        batch = []
        for _ in range(beam_width):
        
            batch += test_data[i:i + 1] 
       
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
        # (len, batch, dim)
               
        tgt_token_ids = [[2] for _ in batch]
        end = [False for _ in batch]

        top_k_prop = [[] for i in range(beam_width)]

        for l in range(src_max_seq_length + 20):
            tgt = torch.tensor(tgt_token_ids).t().contiguous().to(args.device)        
            #[len, batch]
            
            output = model(tgt=tgt, memory=memory, memory_key_padding_mask=~src_padding_mask) # torch.Size([13, 5, 1600])
            #[len, batch, dim]
            
            p = F.log_softmax(output, dim=-1)
            #torch.Size([6, 5, 1600])

            #top1 = output[-1].argmax(-1).tolist()

            topk_value, topk_index = torch.topk(p[-1], beam_width)

            topk_index = topk_index.tolist()
            topk_value = topk_value.tolist()
            #print(tgt_token_ids) # [[2], [2], [2]]
            

            if l == 0:
                for i, (tok, tok_value) in enumerate(zip(topk_index[0], topk_value[0])):
                    
                    tgt_token_ids[i].append(tok if not end[i] else 3)
                    top_k_prop[i].append(tok_value if not end[i] else 3)
              
            else:
                k_tgt_token_ids = []
                k_top_k_prop = []
                k_end = []
                #print(tgt_token_ids) [[2, 7, 7], [2, 171, 7], [2, 0, 7]]
                #print(top_k_prop) [[-0.0027608871459960938], [-6.23060417175293], [-9.305124282836914]]

                #beam width ?????? ??????
                
                

                for num, (i, j, k) in enumerate(zip(tgt_token_ids, top_k_prop, end)):
                    if i[-1] == 3:
                        i_new = copy.deepcopy(i)
                        j_new = copy.deepcopy(j)
                        k_new = copy.deepcopy(k)

                        k_tgt_token_ids.append(i_new)
                        k_top_k_prop.append(j_new)
                        k_end.append(k_new)

                        topk_index[num] = [3]
                        topk_value[num] = [0]

                    else:
                        for _ in range(beam_width):
                            i_new = copy.deepcopy(i)
                            j_new = copy.deepcopy(j)
                            k_new = copy.deepcopy(k)

                            k_tgt_token_ids.append(i_new)
                            k_top_k_prop.append(j_new)
                            k_end.append(k_new)

                #print(k_tgt_token_ids) [[2, 7, 7], [2, 7, 7], [2, 7, 7], [2, 171, 7], [2, 171, 7], [2, 171, 7], [2, 0, 7], [2, 0, 7], [2, 0, 7]]
                #print(k_top_k_prop) [[-0.0027608871459960938], [-0.0027608871459960938], [-0.0027608871459960938], [-6.23060417175293], [-6.23060417175293], [-6.23060417175293], [-9.305124282836914], [-9.305124282836914], [-9.305124282836914]]
                
                count = 0
                
                #?????? ???????????? ????????????.
                for i, i_val in zip(topk_index, topk_value):
                    for j, j_val in zip(i, i_val):
                        
                        if j == 3 or l >= src_seq_length[0] + 20 :
  
                            k_end[count] = True
                        
                        if k_tgt_token_ids[count][-1] == 3:
                            k_tgt_token_ids[count].append(3)
                            k_top_k_prop[count] = j_val
                        
                        else:
                        
                            k_tgt_token_ids[count].append(j)
                            k_top_k_prop[count] = j_val + k_top_k_prop[count][0]

                        count += 1
                
                #top k ???????????? ???????????? ??????
                sort_index = np.argsort(k_top_k_prop)               
                sort_index = sort_index[::-1].tolist()
                
                tgt_token_ids = [] 
                end = []

                #top k ????????? ????????? ??????
                for i in sort_index[:beam_width]:
                    tgt_token_ids.append(k_tgt_token_ids[i])
                    end.append(k_end[i])
                
                
                new_k_top_k_prop = sorted(k_top_k_prop, reverse=True)
                               
                top_k_prop = []

                for i in new_k_top_k_prop[:beam_width]:
                    top_k_prop.append([i])

                if all(end):
                    break
        
        tgt_token_ids = [tgt_token_ids[0]]
      
        prediction.extend([''.join([tokenizer.i2c[tok] for tok in x if tok >= 4]) for x in tgt_token_ids])

    return prediction

def train(model, tokenizer, train_data, valid_data, args, augment):
    model.train()

    train_dataset = TextDataset(train_data, tokenizer, pre_train_mode=augment)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                  batch_size=args.train_batch_size, num_workers=args.num_workers,
                                  collate_fn=lambda x: collate_fn(x, tokenizer, args.max_seq_length))

    valid_dataset = TextDataset(valid_data, tokenizer, pre_train_mode=False)
    valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset),
                                  batch_size=args.eval_batch_size, num_workers=args.num_workers,
                                  collate_fn=lambda x: collate_fn(x, tokenizer, args.max_seq_length))

    valid_noisy = [x['noisy'] for x in valid_data]
    valid_clean = [x['clean'] for x in valid_data]

    epochs = (args.max_steps - 1) // len(train_dataloader) + 1
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=eval(args.adam_betas), eps=args.eps,
                                 weight_decay=args.weight_decay)
    lr_lambda = lambda x: x / args.num_warmup_steps if x <= args.num_warmup_steps else (x / args.num_warmup_steps) ** -0.5
    scheduler = LambdaLR(optimizer, lr_lambda)

    step = 0
    best_val_gleu = -float("inf")
    meter = Meter()
    for epoch in range(1, epochs + 1):
        for batch in train_dataloader:
            aaaa = len(train_dataloader)
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

            if step % args.log_interval == 0:
                lr = scheduler.get_lr()[0]
                loss_sent, loss_token = meter.average()

                logger.info(f' [{step:5d} / {aaaa:8d} ] lr {lr:.6f} | {meter.print_str(True)}')
                nsml.report(step=step, scope=locals(), summary=True,
                            train__lr=lr, train__loss_sent=loss_sent, train__token_ppl=math.exp(loss_token))
                meter.init()

            if step % args.eval_interval == 0:
                start_eval = time.time()

            if step % 500 == 0:
                (val_loss, val_loss_token), valid_str = evaluate(model, valid_dataloader, args)
                prediction = correct(model, tokenizer, valid_noisy, args)
                #prediction = beam_search(model, tokenizer, valid_noisy, 3, args)

                val_em = em(prediction, valid_clean)
                val_gleu = gleu(prediction, valid_clean)

                logger.info('-' * 89)
                logger.info(f' [{epoch:6d}] valid | {valid_str} | em {val_em:5.2f} | gleu {val_gleu:5.2f}')
                logger.info('-' * 89)
                nsml.report(step=step, scope=locals(), summary=True,
                            valid__loss_sent=val_loss, valid__token_ppl=math.exp(val_loss_token),
                            valid__em=val_em, valid__gleu=val_gleu)

                #ppl ???????????????
                if val_gleu > best_val_gleu:
                    best_val_gleu = val_gleu
                    nsml.save("best")

                nsml.save((str(step) + "best"))

        '''
        (val_loss, val_loss_token), valid_str = evaluate(model, valid_dataloader, args)
        prediction = correct(model, tokenizer, valid_noisy, args)
        val_em = em(prediction, valid_clean)
        val_gleu = gleu(prediction, valid_clean)

        logger.info('-' * 89)
        logger.info(f' [{epoch:6d}] valid | {valid_str} | em {val_em:5.2f} | gleu {val_gleu:5.2f}')
        logger.info('-' * 89)
        nsml.report(step=epoch, scope=locals(), summary=True,
                    valid__loss_sent=val_loss, valid__token_ppl=math.exp(val_loss_token),
                    valid__em=val_em, valid__gleu=val_gleu)

        #ppl ???????????????
        if val_gleu > best_val_gleu:
            best_val_gleu = val_gleu
            nsml.save("best")
           
        nsml.save(str(epoch))
        '''

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
    #logger.info(f"args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    #?????? ??????
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

        pre_train_mode = "second"
        augment = False
        
        # step 1
        if pre_train_mode == "first":
            clean_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_corpus"))
        
            new_clean = []

            for i in clean_sents:
                if len(i) < 3:
                    pass
                else:
                    new_clean.append(i)

            train_data = [{"noisy": noisy, "clean": clean} for noisy, clean in zip(new_clean, new_clean)]
            clean_sents = read_strings(os.path.join(args.data_dir, "train_label"))
            
            pairs = [{"noisy": noisy, "clean": clean} for noisy, clean in zip(clean_sents, clean_sents)]

            _, valid_data = divide_dataset(pairs)
            valid_data = valid_data
        
        # step 2 - ssl
        elif pre_train_mode == "second":
            
            clean_sents = ssl_dataset()
        
            noisy_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_corpus"))
            #noisy_sents = noisy_sents[:len(clean_sents)]
            #noisy_sents = noisy_sents[500000:]
            
            k = noisy_sents[:800000]
            
            new_noisy = []
            for i in k:
                if len(i) < 3:
                    pass
                else:
                    new_noisy.append(i)


            print(len(new_noisy))
            print(len(clean_sents))

            pairs = [{"noisy": noisy, "clean": clean} for noisy, clean in zip(new_noisy, clean_sents)]


            #pairs = remove_warining(pairs)

            original_noise_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_data"))
            original_clean_sents = read_strings(os.path.join(args.data_dir, "train_label"))

            original_pairs = [{"noisy": noisy, "clean": clean} for noisy, clean in zip(original_noise_sents, original_clean_sents)]

            train_data, valid_data = divide_dataset(original_pairs)

            train_data = train_data + pairs # + valid_data

            random.shuffle(train_data)

            #train_data, valid_data = pairs[:-3000], pairs[-3000:] 

        logger.info(f"# of train data: {len(train_data)}")
        logger.info(f"# of valid data: {len(valid_data)}")

        #train_sents = [x['noisy'] for x in train_data] + [x['clean'] for x in train_data]        
        
        #tokenizer = CharTokenizer.from_strings(train_sents, args.vocab_size)
        
        #bind_nsml(model, tokenizer, args)
        
        #pre_train unlabeling ????????????
        nsml.load(checkpoint='best', session='KR95444/airush2021-2-4/1408')

    if args.n_gpu > 1:
        print("?????? gpu", torch.cuda.device_count())
        model = torch.nn.DataParallel(model, dim=1)

    if args.mode == "train":
        train(model, tokenizer, train_data, valid_data, args, augment)


if __name__ == "__main__":
    main()
