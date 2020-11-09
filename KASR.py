
import time
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class KASR(nn.Module):
    def __init__(self, args, n_items, n_rels, D_node, adj_entity, adj_relation):
        super().__init__()
        self.n_items = n_items
        self.n_rels = n_rels
        
        self.emb_size = args.emb_size
        self.hidden_size = args.emb_size
        self.n_layers = args.n_layers
        self.emb_dropout = args.emb_dropout
        self.hidden_dropout = args.hidden_dropout
        self.gradient_clip = args.gradient_clip
        
        self.order = args.order
        self.neibor_size = args.neibor_size
        self.attention = args.attention
        self.aggregate = args.aggregate
        
        self.D_node = D_node
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.model_init()
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def model_init(self):
        self.rel_emb_table = nn.Embedding(self.n_rels, self.emb_size, padding_idx=0)
        self.item_emb_table = nn.Embedding(self.n_items, self.emb_size, padding_idx=0)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True) #local interest
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True) #global perference 
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False) 
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        
        #Attention
        self.linear_attention_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_attention = nn.Linear(self.hidden_size, 1, bias=True)
        
        #Attribute
        self.linear_attr = nn.Linear(self.hidden_size*2, self.hidden_size, bias=True)
        
        if self.aggregate == 'concat':
            self.linear_attention_transform = nn.Linear(self.hidden_size*2, self.hidden_size)
        else:
            self.linear_attention_transform = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.gru = nn.GRU(self.emb_size, self.hidden_size, self.n_layers,
                          dropout=self.hidden_dropout, batch_first=True)
        self.emb_dropout_layer = nn.Dropout(self.emb_dropout)
        self.loss_function = nn.CrossEntropyLoss() 
        
        #activication function
        #self.final_activation = nn.Tanh()
        self.final_activation = nn.ReLU()
        self.activation_sigmoid = nn.Sigmoid()
    
            
    def get_neighbors(self, center_entities):
        entities = [center_entities] #center_entities [bs, len] 
        relations = list() 
        
        # Search neighbors
        for i in range(self.order):
            neighbor_entities = self.adj_entity[entities[i]]
            neighbor_entities = neighbor_entities.view([neighbor_entities.shape[0] , -1])
            
            neighbor_relations = self.adj_relation[entities[i]]
            neighbor_relations = neighbor_relations.view([neighbor_relations.shape[0] , -1])

            entities.append(neighbor_entities) 
            relations.append(neighbor_relations)

        return entities, relations
    
    
    def high_order_aggregate(self, entities, relations):
        #k-hop entities [dim, len] - [dim, len*neg] - [dim, len*neg*neg]...
        entity_embeddings = [self.item_emb_table(i) for i in entities]
        relations_embedding = [self.rel_emb_table(i) for i in relations]
        
        #high-order information
        neibor_embeddings = entity_embeddings[self.order]
        for i in range(self.order):
            self_embeddings = entity_embeddings[self.order - i - 1] #[bs, len, dim]
            relation_embedding = relations_embedding[self.order - i - 1] #[bs, len*T, dim]
            stack_self_embedding = torch.stack([self_embeddings],2) #[bs, len, 1, dim]
            
            #[bs, len, neibor_size, dim]
            relation_embedding = relation_embedding.view(
                [relation_embedding.shape[0], -1, self.neibor_size, self.emb_size]) 
            
            #Attention
            #[bs, len, neibor_size, dim] --> [bs, len, dim, 1]
            atten_alpha = self.linear_attention(
                torch.mul(torch.mul(stack_self_embedding, relation_embedding), 
                          neibor_embeddings.view(relation_embedding.shape)))
            alpha = F.softmax(atten_alpha, dim = 2) #[bs, len, neibor_size, 1] 
            
            atten_self_embeddings = torch.sum(torch.mul(
                neibor_embeddings.view(relation_embedding.shape), alpha), dim = 2)#[bs, len, dim]
            
            #Aggregation
            if self.aggregate == 'dot': 
                atten_self_embeddings = torch.mul(self_embeddings, atten_self_embeddings) 
                atten_self_embeddings = self.linear_attention_transform(atten_self_embeddings) 
            elif self.aggregate == 'concat': 
                atten_self_embeddings = torch.cat([self_embeddings, atten_self_embeddings], 2) 
                atten_self_embeddings = self.linear_attention_transform(atten_self_embeddings) 
            else: 
                atten_self_embeddings = self_embeddings + atten_self_embeddings 
                atten_self_embeddings = self.linear_attention_transform(atten_self_embeddings)
            neibor_embeddings = atten_self_embeddings    
        return neibor_embeddings
        
        
    def transform_input(self, h_iids, a_iids): 
        item_embs = self.item_emb_table(h_iids) #[bs, len, dim]
        if self.attention == True:
            entities, relations = self.get_neighbors(h_iids)    
            atten_self_embeddings = self.high_order_aggregate(entities, relations)
            item_embs = self.activation_sigmoid(atten_self_embeddings)
            item_embs = self.emb_dropout_layer(atten_self_embeddings)
        else:
            item_embs = self.emb_dropout_layer(item_embs)  # [bs, len, dim]

        #attr_embedding = self.item_emb_table(a_iids) 
        #attr_degree = self.D_node[a_iids]  #[128, 20, T]
        
        # degree
        #d = 1.0 - torch.unsqueeze(F.softmax(attr_degree, dim = 2), -1)
        #d = torch.unsqueeze(F.softmax(attr_degree, dim = 2), -1)
        # sum 
        #attr_embedding = torch.sum(torch.mul(attr_embedding, d), dim = 2)
        #attr_embedding = torch.mean(attr_embedding, dim = 2) #d[128,T]
        #item_embs = torch.cat([item_embs, attr_embedding], 2)
        #item_embs = self.linear_attr(item_embs)
        
        output, state = self.gru(item_embs)
        
        last_idx = h_iids.ne(0).sum(1) - 1 #[bs] 
        local_ht = output[range(output.size(0)), last_idx] #[bs, hdden_size] 
        #return local_ht
                                 
        q1 = self.linear_one(local_ht).view(local_ht.shape[0], 1, local_ht.shape[1]) 
        q2 = self.linear_two(output) #global [bs, len, dim]
        alpha = self.linear_three(torch.sigmoid(q1 + q2))   #[bs, len, 1]
    
        global_ht = torch.sum(alpha * output, dim = 1)  #[bs, len, dim]->[bs, len, 1]
        global_ht = self.linear_transform(torch.cat([local_ht, global_ht,], 1)) 
        return global_ht
           
        
    def nce_loss(self, h_iids, a_iids, t_iids):
        hidden = self.transform_input(h_iids, a_iids)  #[bs, dim]
        item_emebdding = self.item_emb_table.weight.t()   
        logits = hidden @ item_emebdding  
        logits = self.final_activation(logits)
        loss = self.loss_function(logits, t_iids) 
        return loss
    
    
    def forward(self, h_iids, a_iids):
        hidden = self.transform_input(h_iids, a_iids)  # [bs, dim] 
        item_emebdding = self.item_emb_table.weight.t() 
        logits = hidden @ item_emebdding
        logits = self.final_activation(logits)
        return logits
    