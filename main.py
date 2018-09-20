#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import numpy as np
import random
from subprocess import *

from conf import *
from buildTree import get_info_from_file
import utils
import opt
from data_generater import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
from torch.optim import lr_scheduler

from conf import *
import utils
from data_generater import *
from net import *

random.seed(0)
numpy.random.seed(0)

import cPickle
sys.setrecursionlimit(1000000)

MAX = 2

def setup():
    utils.mkdir(args.data)
    utils.mkdir(args.data+"train/")
    utils.mkdir(args.data+"train_reduced/")
    utils.mkdir(args.data+"test/")
    utils.mkdir(args.data+"test_reduced/")

def list_vectorize(wl,words):
    il = []
    for word in wl:
        if word in words:
            index = words.index(word)
        else:
            index = 0
        il.append(index) 
    return il
def mask_array(embedding_vec):
    max_length = max([len(em) for em in embedding_vec])
    out_em = []
    out_mask = []
    for em in embedding_vec:
        out_em.append(em+[0]*(max_length-len(em)))
        out_mask.append([1]*len(em)+[0]*(max_length-len(em)))
    return numpy.array(out_em),numpy.array(out_mask)

def generate_doc_data(path,files):
    paths = [w.strip() for w in open(files).readlines()]
    docs = []
    done_num = 0
    for p in paths:
        if p.strip().endswith("DS_Store"):continue
        done_num += 1
        file_name = p.strip()
        if file_name.endswith("onf"):
            if args.reduced == 1 and done_num >= 30:break
            doc = get_info_from_file(file_name,2)
            docs.append(doc) 
    return docs


def generate_ZP_index(inpt_len):
    pre_index = []
    post_index = []
    zp2sen = {}
    base = 0 
    sen_index = 0
    zp_index = 0
    zp2real = []
    for length in inpt_len:
        pre_index.append(-1) 
        zp2sen[zp_index] = sen_index
        zp_index += 1
        for i in range(length):
            pre_index.append(i+base) 
            post_index.append(i+base)
            zp2real.append((sen_index,i))
            zp2sen[zp_index] = sen_index
            zp_index += 1
        post_index.append(-1) 
        zp2real.append((sen_index,-1))
        base += length 
        sen_index += 1
    return pre_index,post_index,zp2sen,zp2real

def generate_NP_index(inpt_len,MAX=10):
    start_index = []
    end_index = []
    sen2index = {}
    mask = []
    base = 0 
    sen_index = 0
    np2real = []
    sen2index[sen_index] = []
    np_index = 0
    for length in inpt_len:
        for i in range(length):
            for j in range(i,min(length,i+MAX)):
                start_index.append(i+base) 
                end_index.append(j+base)
                mask.append([1]*(j-i+1)+[0]*(MAX-j+i-1))
                sen2index[sen_index].append(np_index)
                np2real.append((sen_index,i,j))
                np_index += 1
        base += length 
        sen_index += 1
        sen2index[sen_index] = []
    starts = np.array(start_index)
    ends = np.expand_dims(np.array(end_index),1)
    mention_indices = np.expand_dims(starts, axis=1)+ np.expand_dims(np.array(range(MAX)),axis=0)
    mention_indices = np.minimum(mention_indices,ends)
    return np.array(start_index),np.array(end_index),mention_indices,np.array(mask),sen2index,np2real

class Doc:
    def __init__(self):
        self.vec = None
        self.mask = None
        self.sentence_len = None 
        self.zp_pre_index = None
        self.zp_post_index = None

        self.np_index_start = None
        self.np_index_end = None
        self.np_indecs = None
        self.np_mask = None
        self.zp2real = None # zp_index: real_sentence_num,real_index
        self.np2real = None # np_index: real_sentence_num,real_start_index,real_end_index
        self.zp2candi_dict = None

        self.zp_candi_distance_dict = None
        self.train_ante = None
        self.train_azp = None

def generate_vec(doc,sentences):

    vectorized_sentences = []
    for i in range(len(sentences)):
        nodes = sentences[i]
        vectorize_words = list_vectorize(nodes,words) 
        vectorized_sentences.append(vectorize_words)
    vec,mask = mask_array(vectorized_sentences)
    doc.vec = vec
    doc.mask = mask
    inpt_len = numpy.sum(doc.mask,1)
    doc.sentence_len = inpt_len
    
    pre_index,post_index,zp2sen_dict,zp2real = generate_ZP_index(inpt_len)
    doc.zp_pre_index = pre_index
    doc.zp_post_index = post_index

    np_index_start,np_index_end,np_indecs,np_mask,np_sen2index_dict,np2real = generate_NP_index(inpt_len,MAX=nnargs["max_width"])
    doc.np_index_start = np_index_start
    doc.np_index_end = np_index_end
    doc.np_indecs = np_indecs
    doc.np_mask = np_mask
    doc.zp2real = zp2real # zp_index: real_sentence_num,real_index
    doc.np2real = np2real # np_index: real_sentence_num,real_start_index,real_end_index
        
    zp2candi_dict = {} #zp_index: [np_index]
    for i in range(len(pre_index)):
        zp_index = pre_index[i]
        sen_index = zp2sen_dict[i]
        zp2candi_dict[i] = []
        for sen_id in range(max(0,sen_index-2),sen_index+1):
            np_indexs = np_sen2index_dict[sen_id] 
            for np_index in np_indexs:
                if not ( (sen_id == sen_index) and (np_index_end[np_index] > zp_index) ):
                    zp2candi_dict[i].append(np_index)
    doc.zp2candi_dict = zp2candi_dict

    zp_candi_distance_dict = {}
    gold_azp = []
    for zp_index in doc.zp2candi_dict:
        gold_azp_add = 0
        if len(doc.zp2candi_dict[zp_index]) > 0:
            this_zp_real_sentence_num,this_zp_real_index = doc.zp2real[zp_index]

            np_indexes_of_zp = doc.zp2candi_dict[zp_index]
            max_index = max(np_indexes_of_zp)
            zp_candi_distance_dict[zp_index] = [] #utils.get_bin(distance)
            for ii, np_index in enumerate(np_indexes_of_zp):
                distance = max_index-np_index
                zp_candi_distance_dict[zp_index].append(utils.get_bin(distance))

    doc.zp_candi_distance_dict = zp_candi_distance_dict     


def generate_ellipsis_sentences(sentences,replace,doc):
    output = sentences[-1]
    add_item = {}
    for zp_index,np_index in replace:
        real_sentence_num,real_index = doc.zp2real[zp_index]
        if real_sentence_num == len(sentences)-1:
            real_sentence_num,real_start,real_end = doc.np2real[np_index]
            add_item[real_index] = sentences[real_sentence_num][real_start:real_end+1]

    out = []
    for i,word in enumerate(output):
        if i-1 in add_item:
            out.append("*"+"".join(add_item[i-1]))
            out.append(word)
    return out
    

if __name__ == "__main__":

    read_f = file("./data/emb","rb")
    embedding,words,_ = cPickle.load(read_f)
    read_f.close()

    doc = Doc()
    sentences = [["我","喜欢","吃","苹果","。"],["很","好吃","。"]]
    generate_vec(doc,sentences)

    model = torch.load("./model/model") 

    start,end,zp_indexs,np_indexs,np_pair_score,pair_score,zp_score,np_score,zp_x_score = model.forward(doc,dropout=nnargs["dropout"],zp_lamda=0.2,np_lamda=0.2)

    replace = []

    for s,e in zip(start,end):
        this_zp_indexs = zp_indexs[s:e][0]
        this_np_indexs = np_indexs[s:e]
        this_not_zp_score = zp_x_score[this_zp_indexs].view(1,1)
        this_pair_score = np_pair_score[s:e]+pair_score[s:e]+zp_score[this_zp_indexs]
        this_score = F.softmax(torch.squeeze(torch.cat( [this_pair_score,this_not_zp_score] )),0)
        
        this_score = this_score.data.cpu().numpy()
        max_index = numpy.argmax(this_score[:-1])

        if max_index == len(this_np_indexs): # ZP is no anaphoric
            continue

        this_np_index = this_np_indexs[max_index]
        np_indexe_selected = doc.zp2candi_dict[this_zp_indexs][this_np_index]

        replace.append((this_zp_indexs,np_indexe_selected))

    ns = generate_ellipsis_sentences(sentences,replace,doc)
    print "\t".join(ns) 
