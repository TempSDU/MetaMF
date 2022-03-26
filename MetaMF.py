#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter


# In[ ]:


class MetaRecommender(nn.Module):#in fact, it's not a hypernetwork
    def __init__(self, user_num, item_num, item_emb_size=32, item_mem_num=8, user_emb_size=32, mem_size=128, hidden_size=512):#note that we have many users and each user has many layers
        super(MetaRecommender, self).__init__()
        self.item_num = item_num
        self.item_emb_size = item_emb_size
        self.item_mem_num = item_mem_num
        #For each user
        self.user_embedding = nn.Embedding(user_num, user_emb_size)
        self.memory = Parameter(nn.init.xavier_normal_(torch.Tensor(user_emb_size, mem_size)), requires_grad=True)
        #For each layer
        self.hidden_layer_1, self.weight_layer_1, self.bias_layer_1 = self.define_one_layer(mem_size, hidden_size, item_emb_size, int(item_emb_size/4))
        self.hidden_layer_2, self.weight_layer_2, self.bias_layer_2 = self.define_one_layer(mem_size, hidden_size, int(item_emb_size/4), 1)
        self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2 = self.define_item_embedding(item_num, item_emb_size, item_mem_num, mem_size, hidden_size)
    
    def define_one_layer(self, mem_size, hidden_size, int_size, out_size):#define one layer in MetaMF
        hidden_layer = nn.Linear(mem_size, hidden_size)
        weight_layer = nn.Linear(hidden_size, int_size*out_size)
        bias_layer = nn.Linear(hidden_size, out_size)
        return hidden_layer, weight_layer, bias_layer
    
    def define_item_embedding(self, item_num, item_emb_size, item_mem_num, mem_size, hidden_size):
        hidden_layer = nn.Linear(mem_size, hidden_size)
        emb_layer_1 = nn.Linear(hidden_size, item_num*item_mem_num)
        emb_layer_2 = nn.Linear(hidden_size, item_mem_num*item_emb_size)
        return hidden_layer, emb_layer_1, emb_layer_2 
            
    def forward(self, user_id):
        #collaborative memory module
        user_emb = self.user_embedding(user_id)#input_user=[batch_size, user_emb_size]
        cf_vec = torch.matmul(user_emb, self.memory)#cf_vec=[batch_size, mem_size]
        #collaborative memory module
        #meta recommender module
        output_weight = []
        output_bias = []
        
        weight, bias = self.get_one_layer(self.hidden_layer_1, self.weight_layer_1, self.bias_layer_1, cf_vec, self.item_emb_size, int(self.item_emb_size/4))
        output_weight.append(weight)
        output_bias.append(bias) 
                
        weight, bias = self.get_one_layer(self.hidden_layer_2, self.weight_layer_2, self.bias_layer_2, cf_vec, int(self.item_emb_size/4), 1)
        output_weight.append(weight)
        output_bias.append(bias)
        
        item_embedding = self.get_item_embedding(self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2, cf_vec, self.item_num, self.item_mem_num, self.item_emb_size)
        #meta recommender module
        return output_weight, output_bias, item_embedding, cf_vec#([len(layer_list)+1, batch_size, *, *], [len(layer_list)+1, batch_size, 1, *], [batch_size, item_num, item_emb_size], [batch_size, mem_size])
    
    def get_one_layer(self, hidden_layer, weight_layer, bias_layer, cf_vec, int_size, out_size):#get one layer in MetaMF
        hid = hidden_layer(cf_vec)#hid=[batch_size, hidden_size]
        hid = F.relu(hid)
        weight = weight_layer(hid)#weight=[batch_size, self.layer_list[i-1]*self.layer_list[i]]
        bias = bias_layer(hid)#bias=[batch_size, self.layer_list[i]] 
        weight = weight.view(-1, int_size, out_size)
        bias = bias.view(-1, 1, out_size)
        return weight, bias
    
    def get_item_embedding(self, hidden_layer, emb_layer_1, emb_layer_2, cf_vec, item_num, item_mem_num, item_emb_size):
        hid = hidden_layer(cf_vec)#hid=[batch_size, hidden_size]
        hid = F.relu(hid)
        emb_left = emb_layer_1(hid)#emb_left=[batch_size, item_num*item_mem_num]
        emb_right = emb_layer_2(hid)#emb_right=[batch_size, item_mem_num*item_emb_size]
        emb_left = emb_left.view(-1, item_num, item_mem_num)#emb_left=[batch_size, item_num, item_mem_num]
        emb_right = emb_right.view(-1, item_mem_num, item_emb_size)#emb_right=[batch_size, item_mem_num, item_emb_size]
        item_embedding = torch.matmul(emb_left, emb_right)#item_embedding=[batch_size, item_num, item_emb_size]
        return item_embedding


# In[ ]:


class MetaMF(nn.Module):
    def __init__(self, user_num, item_num, item_emb_size=32, item_mem_num=8, user_emb_size=32, mem_size=128, hidden_size=512):
        super(MetaMF, self).__init__()
        self.item_num = item_num
        self.metarecommender = MetaRecommender(user_num, item_num, item_emb_size, item_mem_num, user_emb_size, mem_size, hidden_size)
        
    def forward(self, user_id, item_id):
        #prediction module
        model_weight, model_bias, item_embedding, _ = self.metarecommender(user_id)
        item_id = item_id.view(-1, 1)#item_id=[batch_size, 1]
        item_one_hot = torch.zeros(len(item_id), self.item_num, device=item_id.device)#we generate it dynamically, and default device is cpu
        item_one_hot.scatter_(1, item_id, 1)#item_one_hot=[batch_size, item_num]
        item_one_hot = torch.unsqueeze(item_one_hot, 1)#item_one_hot=[batch_size, 1, item_num]
        item_emb = torch.matmul(item_one_hot, item_embedding)#out=[batch_size, 1, item_emb_size]
        out = torch.matmul(item_emb, model_weight[0])#out=[batch_size, 1, item_emb_size/4]
        out = out+model_bias[0]#out=[batch_size, 1, item_emb_size/4]
        out = F.relu(out)#out=[batch_size, 1, item_emb_size/4]
        out = torch.matmul(out, model_weight[1])#out=[batch_size, 1, 1]
        out = out+model_bias[1]#out=[batch_size, 1, 1]
        out = torch.squeeze(out)#out=[batch_size]
        #prediction module
        return out
    
    def loss(self, prediction, rating):
        #regularizer = torch.sum(torch.matmul(self.metarecommender.memory, self.metarecommender.memory.t()))
        return torch.mean(torch.pow(prediction-rating,2))#+self._lambda*regularizer
