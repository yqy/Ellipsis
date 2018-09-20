#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import random
import cPickle
sys.setrecursionlimit(1000000)

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
import evaluate

print >> sys.stderr, "PID", os.getpid()
random.seed(0)
numpy.random.seed(0)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.set_device(args.gpu)

MAX = 2

def main():
    fix=""
    if args.reduced == 1:
        fix="_reduced"
    read_f = file("./data/train_data"+fix,"rb")
    train_generater = cPickle.load(read_f)
    read_f.close()
    read_f = file("./data/emb","rb")
    embedding_matrix,_,_ = cPickle.load(read_f)
    read_f.close()

    #test_generater = DataGnerater("test",256)
    read_f = file("./data/test_data"+fix,"rb")
    test_generater = cPickle.load(read_f)
    read_f.close()
 
    print "Building torch model"
    model = Network(nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix,nnargs["hidden_dimention"]).cuda()

    this_lr = nnargs["lr"]

    optimizer = optim.Adam(model.parameters(), lr=this_lr)
 
    for echo in range(nnargs["epoch"]):
        cost = 0.0
        np_cost = 0.0
        zp_cost = 0.0
        can_find = 0
        print >> sys.stderr, "Begin epoch",echo
        for doc in train_generater.generate_data(shuffle=True):
        #for doc in train_generater.generate_data(shuffle=False):
            if sum(doc.sentence_len) >= 4000:
                continue
            start,end,zp_indexs,np_indexs,np_pair_score,pair_score,zp_score,np_score,zp_x_score = model.forward(doc,dropout=nnargs["dropout"],zp_lamda=0.1,np_lamda=0.2,train=True)
            loss = 0.0
            optimizer.zero_grad()

            gold_azp = torch.tensor(doc.gold_azp).type(torch.cuda.FloatTensor)
            if sum(doc.gold_azp) >= 1:
                zp_loss = -1.0*torch.log(torch.sum(F.softmax(torch.squeeze(zp_score),0)*gold_azp)+1e-12)
                #zp_loss = -1.0*( torch.sum( (torch.log( (torch.squeeze(zp_score)+1.0)/2.0+1e-12 ) *gold_azp) + (torch.log( (1-(torch.squeeze(zp_score)+1.0)/2.0)+1e-12 ) *(1-gold_azp) ), 0) )
                #zp_loss = -1.0*( torch.sum( (torch.log( (torch.squeeze(F.sigmoid(zp_score)))+1e-12 ) *gold_azp) + (torch.log( (1-(torch.squeeze(F.sigmoid(zp_score))))+1e-12 ) *(1-gold_azp) ), 0) )
                loss += zp_loss
                zp_cost += zp_loss.item()

            #gold_np = torch.tensor(doc.gold_ante).type(torch.cuda.FloatTensor)
            gold_np = torch.tensor(doc.gold_np).type(torch.cuda.FloatTensor)
            np_loss = -1.0*torch.log(torch.sum(F.softmax(torch.squeeze(np_score),0)*gold_np)+1e-12)
            #np_loss = -1.0*( torch.sum( (torch.log( (torch.squeeze(np_score)+1.0)/2.0+1e-12 ) *gold_np) + (torch.log( (1-(torch.squeeze(np_score)+1.0)/2.0)+1e-12 ) *(1-gold_np) ), 0) )
            #np_loss = -1.0*( torch.sum( (torch.log( (torch.squeeze(F.sigmoid(np_score)))+1e-12 ) *gold_np) + (torch.log( (1-(torch.squeeze(F.sigmoid(np_score))))+1e-12 ) *(1-gold_np) ), 0) )
            loss += np_loss
            np_cost += np_loss.item()

            azp_num = 0 
            random_list = range(len(start))
            random.shuffle(random_list)
            #for s,e in zip(start,end):
            for ii in random_list:
                s = start[ii]
                e = end[ii]
                if s == e:
                    continue
                this_zp_indexs = zp_indexs[s:e][0]
                this_np_indexs = np_indexs[s:e]
                #this_not_zp_score = torch.zeros(1,1).cuda()
                #this_not_zp_score = (1-zp_score[this_zp_indexs]).view(1,1)
                this_not_zp_score = zp_x_score[this_zp_indexs].view(1,1)

                this_pair_score = np_pair_score[s:e]+pair_score[s:e]+zp_score[this_zp_indexs]
                #this_pair_score = this_pair_score/3.0
                #this_pair_score = pair_score[s:e]
                this_score = F.softmax(torch.squeeze(torch.cat( [this_pair_score,this_not_zp_score] )),0)
                gold = numpy.append(doc.zp_candi_coref[this_zp_indexs][this_np_indexs],doc.zp_candi_coref[this_zp_indexs][-1])
                gold = torch.tensor(gold).type(torch.cuda.FloatTensor)
                if doc.zp_candi_coref[this_zp_indexs][-1] == 1:
                    if azp_num > 0:
                        loss += (-1.0*torch.log(torch.sum(gold*this_score)+1e-12))
                        azp_num -= 1
                else:
                    #if numpy.sum(doc.zp_candi_coref[this_zp_indexs][this_np_indexs]) >= 1:
                    loss += (-1.0*torch.log(torch.sum(gold*this_score)+1e-12))
                    azp_num += 1
                    can_find += 1

            if loss == 0.0:
                continue

            cost += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)
            optimizer.step()
        print >> sys.stderr, "End epoch",echo,"Cost:", cost,"NP cost",np_cost,"ZP cost",zp_cost, "ha",can_find
        torch.save(model, "./model/model") 
       
        print "Epoch",echo
        with torch.no_grad():
            result = evaluate.evaluate(model,train_generater,dev=True)
            #result = evaluate.evaluate(model,train_generater,dev=False)
        print "Dev:",result["all_pos"],result["predict_pos"],result["hit"],result["f"]

        with torch.no_grad():
            result = evaluate.evaluate(model,test_generater)
        print "Test:",result["all_pos"],result["predict_pos"],result["hit"],result["f"]

 
if __name__ == "__main__":
    main()
