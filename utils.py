import math
import time
import random
import torch
import numpy as np
import pandas as pd
import networkx as nx

from kg import KGraph
from torch.utils.data import Dataset, DataLoader


########################################## Evaluation #########################################
def getHitRatio(ranklist, targetItem):
    for item in ranklist:
        if item == targetItem:
            return 1
    return 0

def getNDCG(ranklist, targetItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == targetItem:
            return math.log(2) / math.log(i+2)
    return 0

def getMRR(ranklist, targetItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == targetItem:
            return 1.0/(i+1)
    return 0

def metrics(ranklist, targetItem):
    hr = getHitRatio(ranklist, targetItem)
    ndcg = getNDCG(ranklist, targetItem)     
    mrr = getMRR(ranklist, targetItem)
    return hr, ndcg, mrr


######################################### Data Loader #########################################
class HistDataset(Dataset):
    def __init__(self, df, idx_list, attr_size, hist_max_len):
        self.data = df.values  # [user, item, timestamp]
        self.idx_list = idx_list
        self.MASK = 0
        self.attr_size = attr_size
        self.hist_max_len = hist_max_len
        
    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        l, r, t = self.idx_list[item] 
        submat = self.data[l:r]
        h_uid, h_items, h_attrs = submat.T 
        uid, t_item, _ = self.data[t] 
        assert np.all(h_uid == uid)  
    
        h_attrs[-1] = [self.MASK]*self.attr_size 
        h_attrs = np.array([i for i in h_attrs])
        n = len(h_items) 
        if n < self.hist_max_len:
            h_items = np.pad(h_items, [0, self.hist_max_len - n], 'constant', constant_values=self.MASK)
            h_attrs = np.pad(h_attrs, [(0,self.hist_max_len - n), (0,0)], 'constant', constant_values=self.MASK)
        return uid, np.array(list(h_items)).astype(np.long), h_attrs, t_item
    

class Loader():
    def __init__(self, args):
        self.dataset = args.dataset
        self.attr_size = args.attr_size
        self.neibor_size = args.neibor_size
        self.hist_min_len = args.hist_min_len
        self.hist_max_len = args.hist_max_len
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.n_workers = args.n_workers
        self.valid = args.valid
        self.n_neg = args.n_neg
        self.MASK = 0
        
        self.kg = KGraph(self.dataset, self.attr_size)
        self.node_neibors = self.kg.node_neibors
        self.n_entity, self.n_relation = self.kg.n_entity, self.kg.n_relation
        
        self.nodes_degree = self.kg.nodes_degree
        self.D_node = self.construct_D()
        self.adj_entity, self.adj_relation = self.construct_neibors_adj()
        self.all_items_list, self.train_dl, self.valid_dl = self.load_data()

    def construct_D(self):
        sorted_degree = sorted(self.nodes_degree.items(), key = lambda x:x[0])
        D_node = [i[0] for i in sorted_degree]
        return D_node
    
        
    def construct_neibors_adj(self):
        adj_entity = np.zeros([self.n_entity, self.neibor_size], dtype=np.int64)
        adj_relation = np.zeros([self.n_entity, self.neibor_size], dtype=np.int64)
        
        for node in range(self.n_entity):
            neighbors = self.node_neibors[node]
            n_neighbors = len(neighbors)
            #sample
            if n_neighbors >= self.neibor_size:
                sampled_indices = np.random.choice(neighbors, size = self.neibor_size, replace=False)
            else:
                sampled_indices = np.random.choice(neighbors, size = self.neibor_size, replace=True)
                
            adj_entity[node] = np.array([n for n in sampled_indices])
            adj_relation[node] = np.array([self.kg.G.get_edge_data(node,n)['rel'] for n in sampled_indices])
        return adj_entity, adj_relation
    
    
    def extract_subseq(self, n):
        idx_list = []
        for right in range(self.hist_min_len, n):   
            left = max(0, right - self.hist_max_len)
            target = right
            idx_list.append([left, right, target])
        return np.asarray(idx_list)
        
    def get_idx(self, df):
        offset = 0
        train_idx_list = []
        valid_idx_list = []
        for n in df.groupby('user').size():
            train_idx_list.append(self.extract_subseq(n-1) + offset)
            valid_idx_list.append(np.add([max(0, n-1 - self.hist_max_len), n-1, n-1], offset))
            offset += n 
        train_idx_list = np.concatenate(train_idx_list)
        valid_idx_list = np.stack(valid_idx_list)
        return train_idx_list, valid_idx_list
   


    def load_data(self):
        users, entities, attrs = list(), list(), list()
        df = pd.read_csv(f'{self.dataset}data.csv', header=None, names=['user', 'item', 'rating', 'timestamp'])
        del df['rating']
        df = df.sort_values(['user', 'timestamp'], ascending=[True, True])
        all_items_list = sorted(df['item'].unique().tolist())
    
        black_list = df.groupby('user').apply(lambda subdf:
            [i for i in subdf.item.values]).to_dict() 
 
        black_list_path = {u:self.kg.entity_seq_shortest_path(black_list[u]) 
                           for u in black_list}    
        for (user, entity_list) in black_list_path.items():
            users += [user]*len(entity_list)
            entities += [e_a[0] for e_a in entity_list]
            attrs += [e_a[1] for e_a in entity_list]
        data = pd.DataFrame({'user':users, 'item':entities, 'attr':attrs})
        
        train_idx_list, valid_idx_list = self.get_idx(data)
        train_ds = HistDataset(data, train_idx_list, self.attr_size, self.hist_max_len)
        valid_ds = HistDataset(data, valid_idx_list, self.attr_size, self.hist_max_len)
        
        train_dl = DataLoader(train_ds, self.train_batch_size, pin_memory=True, shuffle=True, drop_last=True, num_workers=self.n_workers)
        valid_dl = DataLoader(valid_ds, self.eval_batch_size, pin_memory=True, num_workers=self.n_workers)
        return all_items_list, train_dl, valid_dl
