# -*- coding: utf-8 -*-
import sys

import math
import random
import numpy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
import torch.optim as optim

from conf import *
import opt

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
random.seed(0)
numpy.random.seed(0)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)

class Network(nn.Module):
    def __init__(self, embedding_size, embedding_dimention, embedding_matrix, hidden_dimention):
        super(Network,self).__init__()
        self.embedding_layer = nn.Embedding(embedding_size,embedding_dimention)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.mention_width_embeddings = nn.Embedding(10,nnargs["feature_dimention"])
        self.distance_embeddings = nn.Embedding(10,nnargs["feature_dimention"])

        self.lstm = nn.LSTM(embedding_dimention,hidden_dimention, batch_first=True, bidirectional=True)
        self.hidden_dimention = hidden_dimention

        self.zp_pre_representation_layer_1 = nn.Linear(self.hidden_dimention*4,self.hidden_dimention*2)
        self.zp_pre_representation_layer_2 = nn.Linear(self.hidden_dimention*2,1)

        self.mention_emb_layer = nn.Linear(hidden_dimention*2,1)
        self.np_representation_layer_1 = nn.Linear(self.hidden_dimention*6+nnargs["feature_dimention"],self.hidden_dimention*2)
        self.np_representation_layer_2 = nn.Linear(self.hidden_dimention*2,1)

        self.zp_score_layer = nn.Linear(self.hidden_dimention*4,hidden_dimention*2)
        self.np_score_layer = nn.Linear(self.hidden_dimention*6+nnargs["feature_dimention"],hidden_dimention*2)
        self.feature_score_layer = nn.Linear(nnargs["feature_dimention"],hidden_dimention*2)
        self.hidden_score_layer_1 = nn.Linear(hidden_dimention*2,hidden_dimention)
        self.hidden_score_layer_2 = nn.Linear(hidden_dimention,1)

        self.zp_x_layer = nn.Linear(self.hidden_dimention*4,1)

        #self.activate = nn.ReLU()
        self.activate = nn.Tanh()

    def encode_zp(self, zp_in, dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        this_hidden = self.zp_pre_representation_layer_1(zp_in)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        this_hidden = self.zp_pre_representation_layer_2(this_hidden)
        scores = this_hidden
        #scores = self.activate(this_hidden)
        #scores = F.sigmoid(this_hidden)
        return scores

    def encode_np(self,sentence_embedding,mention_starts,mention_ends,mention_indeces,mention_mask,dropout=0.0):
        mention_emb_list = []
        dropout_layer = nn.Dropout(dropout)
        
        mention_start_emb = sentence_embedding[mention_starts] # [num_mentions, emb(hidden_dimention*2)]
        mention_emb_list.append(mention_start_emb)
        mention_end_emb = sentence_embedding[mention_ends] # [num_mentions, emb]
        mention_emb_list.append(mention_end_emb)
    
        mention_width = 1 + mention_ends - mention_starts # [num_mentions]

        mention_width_index = mention_width - 1 # [num_mentions]

        mention_width_emb = self.mention_width_embeddings(mention_width_index)
        mention_emb_list.append(mention_width_emb) 
                
        shapes = list(mention_indeces.shape)
        num_mentions = shapes[0]
        max_mention_width = shapes[1]
        
        mention_text_emb = sentence_embedding[mention_indeces] #[num_mentions, max_mention_width, emb(hidden_dimention*2)]

        self.head_scores = self.mention_emb_layer(sentence_embedding) # [num_words, 1]
        mention_head_scores = torch.squeeze(self.head_scores[mention_indeces]) #[num_mentions, max_mention_width]

        mention_attention = F.softmax(mention_head_scores + torch.log(mention_mask),1) #[num_mentions, max_mention_width]
        mention_head_emb = torch.sum(mention_text_emb * mention_attention.view(num_mentions,max_mention_width,1),1) # [num_mentions, emb]
        mention_emb_list.append(mention_head_emb)
        
        mention_emb = torch.cat(mention_emb_list,1) #[num_mentions, emb(hidden_dimention*2 *3 + 16(feature_dimention))]

        this_hidden = self.np_representation_layer_1(mention_emb)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        this_hidden = self.np_representation_layer_2(this_hidden)
        scores = this_hidden
        #scores = self.activate(this_hidden)
        #scores = F.sigmoid(this_hidden)
        return scores,mention_emb

    def encode_sentences_dynantic(self,inpt,inpt_len,mask):
        inpt_sort_idx = numpy.argsort(-inpt_len)
        inpt_unsort_idx = torch.LongTensor(numpy.argsort(inpt_sort_idx))
        inpt_len = inpt_len[inpt_sort_idx]
        inptx = inpt[torch.LongTensor(inpt_sort_idx)]
        inpt_emb_p = torch.nn.utils.rnn.pack_padded_sequence(inptx, inpt_len, batch_first=True)
        out_pack, (ht, ct) = self.lstm(inpt_emb_p)

        out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=True)  # (sequence, lengths)
        out = out[0]
        output = out[inpt_unsort_idx]

        value_shape = list(output.shape)
        emb_dimention = int(value_shape[2])
        sentence_embedding = torch.masked_select(output.transpose(2,1).transpose(0,1), mask)
        sentence_embedding = sentence_embedding.view(emb_dimention,-1).transpose(0,1)
        return sentence_embedding

    def forward(self,doc,dropout=0.0,out=False,np_lamda=0.2,zp_lamda=0.1,train=False):
        mask = torch.tensor(doc.mask).type(torch.cuda.ByteTensor)
        sen_vec = torch.tensor(doc.vec).type(torch.cuda.LongTensor)
        inpt_len = doc.sentence_len
        word_embedding = self.embedding_layer(sen_vec) #[sentence_num,word_num,embedding_dimention]
        sentence_embedding = self.encode_sentences_dynantic(word_embedding,inpt_len,mask) #[total_word_num,hidden_dimention*2]
        sentence_embedding = torch.cat([sentence_embedding,torch.zeros(1,self.hidden_dimention*2).cuda()],0)

        zp_pre_index = torch.tensor(doc.zp_pre_index).type(torch.cuda.LongTensor)
        zp_post_index = torch.tensor(doc.zp_post_index).type(torch.cuda.LongTensor)
        zp_embedding = torch.cat([sentence_embedding[zp_pre_index],sentence_embedding[zp_post_index]],1) #[total_word_num,hidden_dimention*4]
        zp_score = self.encode_zp(zp_embedding,dropout) #[total_word_num,1]

        zp_x_score = self.zp_x_layer(zp_embedding)
        #zp_x_score = F.sigmoid(zp_x_score)
        #zp_x_score = self.activate(zp_x_score)
        
        np_index_start = torch.tensor(doc.np_index_start).type(torch.cuda.LongTensor)
        np_index_end = torch.tensor(doc.np_index_end).type(torch.cuda.LongTensor)
        np_indecs = torch.tensor(doc.np_indecs).type(torch.cuda.LongTensor)
        np_mask = torch.tensor(doc.np_mask).type(torch.cuda.FloatTensor)

        np_score,np_embedding = self.encode_np(sentence_embedding,np_index_start,np_index_end,np_indecs,np_mask,dropout) #[num_nps,1],[num_nps,hidden_dimention*6+16] 
        selected_np_indexs = opt.get_candis_by_scores(doc.train_ante,doc.np_index_start,doc.np_index_end,numpy.squeeze(np_score.data.cpu().numpy()),lamda=np_lamda,train=train)
        selected_zp_indexs = opt.get_zps_by_scores(doc.train_azp,numpy.squeeze(zp_score.data.cpu().numpy()),lamda=zp_lamda,train=train) 

        zp_reindex,np_reindex,np_reindex_real,start,end,distance_feature = opt.get_pairs(selected_zp_indexs,selected_np_indexs,doc)
        zp_reindex_torch = torch.tensor(zp_reindex).type(torch.cuda.LongTensor)
        np_reindex_torch = torch.tensor(np_reindex).type(torch.cuda.LongTensor)
        np_reindex_real_torch = torch.tensor(np_reindex_real).type(torch.cuda.LongTensor)

        zp_pair_representation = zp_embedding[zp_reindex_torch]
        np_pair_representation = np_embedding[np_reindex_real_torch]
        np_pair_score = np_score[np_reindex_real_torch]
        distance_feature = torch.tensor(distance_feature).type(torch.cuda.LongTensor)
        pair_score = self.generate_score(zp_pair_representation,np_pair_representation,distance_feature,dropout)

        #if out:
        #    print selected_np_indexs
        #    print selected_zp_indexs
        #    print zp_reindex

        return start,end,zp_reindex_torch.data.cpu().numpy(),np_reindex_torch.data.cpu().numpy(),np_pair_score,pair_score,zp_score,np_score,zp_x_score

    def generate_score(self,zp_embedding,np_embedding,distance_feature,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        mention_distance_emb = self.distance_embeddings(distance_feature)
        hidden = self.zp_score_layer(zp_embedding) + self.np_score_layer(np_embedding) + self.feature_score_layer(mention_distance_emb)
        hidden = self.activate(hidden)
        hidden = dropout_layer(hidden)
        hidden = self.hidden_score_layer_1(hidden)
        hidden = self.activate(hidden)
        hidden = dropout_layer(hidden)
        score = self.hidden_score_layer_2(hidden)
        #score = self.activate(score)
        #score = F.sigmoid(score)
        return score


