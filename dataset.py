import random
import numpy as np

import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, pre_train_mode):
        self.data = data
        self.tokenizer = tokenizer
        self.pre_train_mode = pre_train_mode
        self.a, self.b = self._solve_ab_given_mean_var(0.1, 0.03)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        src_token_ids = self.tokenizer(self.data[idx]['noisy'])
        tgt_token_ids = self.tokenizer(self.data[idx]['clean'])

        a = random.random()        

        if self.pre_train_mode:
            
            funcs = [self._add_func, self._replace_func, self._delete_func, self._shuffle_func]
            np.random.shuffle(funcs)
            
            for f in funcs:
                src_token_ids = f(src_token_ids)
            
            #src_token_ids = self._replace_func(src_token_ids)
            
        output = {"noisy" : src_token_ids, "clean": tgt_token_ids}

        return output

    def _solve_ab_given_mean_var(self, mean, var):
        a = mean * mean * (1. - mean) / var - mean
        b = (1. - mean) * (mean * (1. - mean) / var - 1.)

        return a, b

    def _delete_func(self, tgt):
        delete_ratio = np.random.beta(self.a, self.b)
        ret = []
        rnd = np.random.random(len(tgt))


        for i, p in enumerate(tgt):
            if rnd[i] < delete_ratio:
                pass
            else:
                ret.append(p)

        k = len(tgt) - len(ret)

        for i in range(k):
            ret.append(4)

        return ret

    def _add_func(self, tgt):
        add_ratio = np.random.beta(self.a, self.b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, p in enumerate(tgt):
            if rnd[i] < add_ratio:
                rnd_ex = np.random.randint(4, 1599)
                ret.append(rnd_ex)    

            ret.append(p)

        return ret

    def _replace_func(self, tgt):
        replace_ratio = np.random.beta(self.a, self.b)
        ret = []
        rnd = np.random.random(len(tgt))

        for i, p in enumerate(tgt):
            if rnd[i] < replace_ratio: 
                rnd_ex = np.random.randint(4, 1599)
                ret.append(rnd_ex)
            else:
                ret.append(p)
                
        return ret

    def _shuffle_func(self, tgt):

        shuffle_key = [i + np.random.normal(loc=0, scale=0.5) for i in range(len(tgt))]
        new_idx = np.argsort(shuffle_key)
        res = [tgt[i] for i in new_idx]
            
        return res

def collate_fn(data, tokenizer, max_seq_length=None):
    #src_token_ids = [tokenizer(x['noisy']) for x in data]
    #tgt_token_ids = [[2] + tokenizer(x['clean']) + [3] for x in data]
    
    src_token_ids = [x['noisy'] for x in data]
    tgt_token_ids = [[2] + x['clean'] + [3] for x in data]

 
    src_max_seq_length = max([len(x) for x in src_token_ids])
    if max_seq_length and max_seq_length < src_max_seq_length:
        src_max_seq_length = max_seq_length
    tgt_max_seq_length = max([len(x) + 1 for x in tgt_token_ids])
    if max_seq_length and max_seq_length + 1 < tgt_max_seq_length:
        tgt_max_seq_length = max_seq_length + 1

    src_padded = []
    src_padding_mask = []
    tgt_padded = []
    tgt_padding_mask = []
    for src, tgt in zip(src_token_ids, tgt_token_ids):
        src = src[:src_max_seq_length]
        src_pad_length = src_max_seq_length - len(src)
        src_padded.append(src + [1] * src_pad_length)
        src_padding_mask.append([1] * len(src) + [0] * src_pad_length)
        tgt = tgt[:tgt_max_seq_length]
        tgt_pad_length = tgt_max_seq_length - len(tgt)
        tgt_padded.append(tgt + [1] * tgt_pad_length)
        tgt_padding_mask.append([1] * (len(tgt) - 1) + [0] * tgt_pad_length)

    src_padded = torch.tensor(src_padded).t().contiguous()
    src_padding_mask = torch.tensor(src_padding_mask).bool().t()
    tgt_padded = torch.tensor(tgt_padded).t().contiguous()
    tgt_padding_mask = torch.tensor(tgt_padding_mask).bool().t()
    return src_padded, tgt_padded[:-1], src_padding_mask, tgt_padding_mask, tgt_padded[1:]
