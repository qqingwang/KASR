import pandas as pd
import numpy as np
import argparse
import os
import time
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import utils
import KASR 
device = torch.device('cuda:0')


def _parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/')
    parser.add_argument('--emb_size', type=int, default=100)
    parser.add_argument('--emb_dropout', type=int, default=0.1) 
    parser.add_argument('--hidden_dropout', type=int, default=0) 
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0005) 
    parser.add_argument('--lr_dc', type=float, default=1e-5)
    parser.add_argument('--step', type=int, default=1)   
    parser.add_argument('--hist_min_len', type=int, default=10)
    parser.add_argument('--hist_max_len', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size',type=int, default=128)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--gradient_clip', type=int, default=0) 
    parser.add_argument('--n_neg', type=int, default=500)
    parser.add_argument('--order', type=int, default=1)
    parser.add_argument('--neibor_size', type=int, default=4)
    parser.add_argument('--attr_size', type=int, default=2)
    parser.add_argument('--attention', type=bool,default=True)
    parser.add_argument('--aggregate', type=str, default='concat')
    parser.add_argument('--valid', type=bool, default=False)
    args = parser.parse_args()
    return args 


def _update(model, optimizer, batch): 
    model.train() 
    optimizer.zero_grad()
    uids, h_items, h_attrs, t_item = [x for x in batch]
    loss = model.nce_loss(h_items.to(device), h_attrs.to(device), t_item.to(device))
    loss.backward()
    if model.gradient_clip > 0:
        clip_grad_norm_(model.parameters(), engine.gradient_clip)
    optimizer.step()
    return loss.item()


def _inference(model, batch, all_items_list):
    model.eval()
    res = np.array([0.]*6)
    with torch.no_grad():
        uids, h_items, h_attrs, t_iids = [x for x in batch]   
        scores = model(h_items.to(device), h_attrs.to(device)) 
        for u in range(len(uids)):
            target = t_iids[u].numpy()
            leave_out_one_sample = random.sample(all_items_list, args.n_neg)
            if target not in leave_out_one_sample:
                leave_out_one_sample[0] = target
            scores_temp = scores[u].cpu().detach().numpy()
            item_score = [(j, scores_temp[j]) for j in leave_out_one_sample]
            item_score = sorted(item_score, key=lambda x: x[1]) 
            item_score.reverse()
            item_sort = [x[0] for x in item_score] 
            res[0:3] += utils.metrics(item_sort[0:10], target)
            res[3:6] += utils.metrics(item_sort[0:20], target)
    res /= len(t_iids)
    return res


def main(args):
    # load data
    print('-- load data...')
    loader = utils.Loader(args)
    train_dl, valid_dl = loader.train_dl, loader.valid_dl

    def adjust_learning_rate(optimizer, epoch):
        lr = args.lr * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # construct model
    print('-- construct model...')
    model = KASR.KASR(args, loader.n_entity, loader.n_relation,
                 torch.Tensor(loader.D_node).to(device),
                 torch.Tensor(loader.adj_entity).long().to(device),
                 torch.Tensor(loader.adj_relation).long().to(device)) 
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_dc)  

    #train & eval
    for epoch in range(args.epoch):
        print('\n[Epoch {}]'.format(epoch))
        #adjust_learning_rate(optimizer, epoch)
        
        total_loss = 0.
        t1 = time.time()
        for batch in tqdm(train_dl, desc = 'train', ncols = 80):
            loss = _update(model, optimizer, batch)        
            total_loss += loss 
        print('loss:{:.2f}\tseconds:{:.2f}'.format(total_loss/len(train_dl), time.time() - t1))

        #eval
        res = np.array([0.]*6)
        for batch in tqdm(valid_dl,desc = 'eval ', ncols = 80):
            res += _inference(model, batch, loader.all_items_list)
        res = res/len(valid_dl)
        print('hr@20:{:.4f}\tndcg@20:{:.4f}\tmrr@20:{:.4f}'.format(res[3], res[4], res[5]))
    
    
if __name__ == '__main__':
    args = _parser_args()  
    main(args)
    